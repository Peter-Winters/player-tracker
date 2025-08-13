import numpy as np
import cv2
from collections import defaultdict
from scipy.optimize import linear_sum_assignment
from scipy.spatial import Voronoi, ConvexHull

# ──────────────────────────────
# 0) Tiny trajectory “store”
# ──────────────────────────────

def create_trajectory_store(fps: float):
    """Return a dict that will hold per-track samples."""
    return {"fps": float(fps), "data": defaultdict(list)}  # track_id -> [ (t,x,y,team) ]

def add_frame(store: dict, frame_idx: int, ids, pts_m, team_ids):
    """Append one frame of metric positions."""
    t = frame_idx / store["fps"]
    for tid, (x, y), team in zip(ids, pts_m, team_ids):
        store["data"][int(tid)].append((float(t), float(x), float(y), int(team)))

def to_numpy(store: dict, track_id: int) -> np.ndarray:
    """Return Nx4 array [t,x,y,team] for a given track (or empty)."""
    rows = store["data"].get(int(track_id), [])
    return np.array(rows, dtype=float) if rows else np.empty((0,4), dtype=float)

def all_tracks(store: dict):
    return list(store["data"].keys())

def per_player_speed(store: dict, track_id: int) -> np.ndarray:
    """Finite-difference speed (m/s) for one track."""
    arr = to_numpy(store, track_id)  # [t,x,y,team]
    if len(arr) < 2: return np.zeros((0,), dtype=float)
    dt = np.diff(arr[:,0]); dx = np.diff(arr[:,1]); dy = np.diff(arr[:,2])
    v = np.sqrt(dx*dx + dy*dy) / np.maximum(dt, 1e-6)
    return v

# ──────────────────────────────
# 1) Marker distance (nearest-opponent matching per frame)
# ──────────────────────────────

def init_marker_state():
    """State dicts to accumulate distances across frames."""
    return {"per_player": defaultdict(list), "team_avgs": defaultdict(list)}

def marker_update_frame(state: dict, ids, team_ids, pts_m):
    """Update per-player & per-team marker distances for this frame."""
    ids = np.asarray(ids)
    teams = np.asarray(team_ids)
    P = np.asarray(pts_m, dtype=float)

    uniq = np.unique(teams)
    if len(uniq) != 2:
        return  # need exactly two teams present

    A_mask = teams == uniq[0]
    B_mask = ~A_mask
    if P[A_mask].size == 0 or P[B_mask].size == 0:
        return

    A_ids, A_pts = ids[A_mask], P[A_mask]  # (Na,), (Na,2)
    B_ids, B_pts = ids[B_mask], P[B_mask]  # (Nb,), (Nb,2)

    # Pairwise distances (Na x Nb)
    D = np.sqrt(((A_pts[:, None, :] - B_pts[None, :, :])**2).sum(axis=2))
    # Hungarian works with rectangular matrices; returns min(Na, Nb) pairs
    r, c = linear_sum_assignment(D)

    pairs = [(int(A_ids[i]), int(B_ids[j]), float(D[i, j])) for i, j in zip(r, c)]

    # Log per-player distances for both players in each pair
    for a_id, b_id, d in pairs:
        state["per_player"][a_id].append(d)
        state["per_player"][b_id].append(d)

    # Framewise averages per team (same mean for both)
    if pairs:
        m = float(np.mean([d for *_ , d in pairs]))
        state["team_avgs"][int(uniq[0])].append(m)
        state["team_avgs"][int(uniq[1])].append(m)


def marker_player_average(state: dict, track_id: int):
    vals = state["per_player"].get(int(track_id))
    return float(np.mean(vals)) if vals else None

def marker_team_average(state: dict, team_id: int):
    vals = state["team_avgs"].get(int(team_id))
    return float(np.mean(vals)) if vals else None

# ──────────────────────────────
# 2) Voronoi cells bounded to a rectangular pitch
# ──────────────────────────────

def _voronoi_finite_polygons_2d(vor, radius=None):
    """Reconstruct infinite regions of a 2D Voronoi diagram to finite polygons."""
    assert vor.points.shape[1] == 2
    new_regions = []
    new_vertices = vor.vertices.tolist()
    center = vor.points.mean(axis=0)
    if radius is None:
        #radius = vor.points.ptp().max() * 2
        radius = float(np.ptp(vor.points, axis=0).max()) * 2
        if not np.isfinite(radius) or radius <= 0:
            # Fallback based on spread; ensures positive radius
            center = vor.points.mean(axis=0)
            spread = float(np.linalg.norm(vor.points - center, axis=1).max())
            radius = (spread * 2) if spread > 0 else 1.0

    # map from ridge point -> list of (other_point, v1, v2)
    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))

    for p1, region_idx in enumerate(vor.point_region):
        vertices = vor.regions[region_idx]
        if -1 not in vertices:
            new_regions.append(vertices)
            continue

        ridges = all_ridges[p1]
        new_region = [v for v in vertices if v >= 0]

        for p2, v1, v2 in ridges:
            if v1 >= 0 and v2 >= 0:
                continue
            # direction of ray
            t = vor.points[p2] - vor.points[p1]
            t /= np.linalg.norm(t) + 1e-12
            n = np.array([-t[1], t[0]])
            midpoint = (vor.points[p1] + vor.points[p2]) / 2
            direction = np.sign(np.dot(midpoint - center, n)) * n
            # pick the finite vertex and extend in 'direction'
            v_finite = vor.vertices[v1 if v1 >= 0 else v2]
            far_point = v_finite + direction * radius
            new_vertices.append(far_point.tolist())
            new_region.append(len(new_vertices) - 1)

        # order CCW
        vs = np.asarray([new_vertices[v] for v in new_region])
        c = vs.mean(axis=0)
        angles = np.arctan2(vs[:,1] - c[1], vs[:,0] - c[0])
        new_regions.append([v for _, v in sorted(zip(angles, new_region))])

    return new_regions, np.asarray(new_vertices)

def _clip_poly_to_rect(poly, x0, y0, x1, y1):
    """Sutherland–Hodgman clipping to an axis-aligned rectangle."""
    S = [np.array(p, float) for p in poly]

    def clip(edge_fn, intersect_fn):
        out = []
        for i in range(len(S)):
            P, Q = S[i-1], S[i]
            pin, qin = edge_fn(P), edge_fn(Q)
            if pin and qin:
                out.append(Q)
            elif pin and not qin:
                I = intersect_fn(P, Q)
                if I is not None: out.append(I)
            elif not pin and qin:
                I = intersect_fn(P, Q)
                if I is not None: out.append(I)
                out.append(Q)
        return out

    def ix_left(P, Q):
        xA, yA = P; xB, yB = Q; dx = xB-xA; 
        if dx == 0: return None
        t = (x0-xA)/dx
        if 0 <= t <= 1:
            y = yA + t*(yB-yA)
            if y0-1e-6 <= y <= y1+1e-6: return np.array([x0, y])
        return None

    def ix_right(P, Q):
        xA, yA = P; xB, yB = Q; dx = xB-xA;
        if dx == 0: return None
        t = (x1-xA)/dx
        if 0 <= t <= 1:
            y = yA + t*(yB-yA)
            if y0-1e-6 <= y <= y1+1e-6: return np.array([x1, y])
        return None

    def ix_bottom(P, Q):
        xA, yA = P; xB, yB = Q; dy = yB-yA;
        if dy == 0: return None
        t = (y0-yA)/dy
        if 0 <= t <= 1:
            x = xA + t*(xB-xA)
            if x0-1e-6 <= x <= x1+1e-6: return np.array([x, y0])
        return None

    def ix_top(P, Q):
        xA, yA = P; xB, yB = Q; dy = yB-yA;
        if dy == 0: return None
        t = (y1-yA)/dy
        if 0 <= t <= 1:
            x = xA + t*(xB-xA)
            if x0-1e-6 <= x <= x1+1e-6: return np.array([x, y1])
        return None

    for edge_fn, ix in (
        (lambda P: P[0] >= x0, ix_left),
        (lambda P: P[0] <= x1, ix_right),
        (lambda P: P[1] >= y0, ix_bottom),
        (lambda P: P[1] <= y1, ix_top),
    ):
        S = clip(edge_fn, ix)
        if not S: break
    return np.array(S, float)

def voronoi_cells_bounded(pts_m, pitch_w_m, pitch_h_m):
    pts_m = np.asarray(pts_m, float)
    n = len(pts_m)

    # Always return one polygon per point to match team_ids length
    if n == 0:
        return [], []
    if n < 3:
        # Not enough points for a 2D Voronoi; return empty cells
        return [np.empty((0,2), float) for _ in range(n)], [0.0] * n

    # Jitter if duplicates would break Qhull
    if np.unique(pts_m, axis=0).shape[0] != n:
        eps = max(pitch_w_m, pitch_h_m) * 1e-6
        pts_m = pts_m + np.random.uniform(-eps, eps, size=pts_m.shape)

    vor = Voronoi(pts_m)
    regions, vertices = _voronoi_finite_polygons_2d(vor)

    polys, areas = [], []
    for reg in regions:
        poly = vertices[reg]
        poly = _clip_poly_to_rect(poly, 0.0, 0.0, pitch_w_m, pitch_h_m)
        if len(poly) >= 3:
            x, y = poly[:,0], poly[:,1]
            area = 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
            polys.append(poly)
            areas.append(float(area))
        else:
            polys.append(np.empty((0,2), float))
            areas.append(0.0)

    # regions are in the same order as input points, so polys/areas length == n
    return polys, areas


def draw_voronoi_on_pitch(pitch_img, polys, team_ids, metric_to_pitch_px_fn, team_colours, alpha=0.35):
    """Overlay filled Voronoi cells on the mini-pitch image (in-place)."""
    overlay = pitch_img.copy()
    for poly, team in zip(polys, team_ids):
        if poly.size == 0: continue
        pts_px = np.array([metric_to_pitch_px_fn(x, y) for x, y in poly], dtype=np.int32)
        cv2.fillPoly(overlay, [pts_px], team_colours[int(team)+2])
    cv2.addWeighted(overlay, alpha, pitch_img, 1-alpha, 0, dst=pitch_img)

# ──────────────────────────────
# 3) Useful team shape extras
# ──────────────────────────────

def team_centroid_and_spread(pts_m, team_ids):
    """
    Return {team_id: (centroid_xy, mean_dist_to_centroid, hull_area_m2)}.
    ConvexHull.volume is the polygon area in 2D.
    """
    out = {}
    ids = np.asarray(team_ids); P = np.asarray(pts_m)
    for t in np.unique(ids):
        Q = P[ids == t]
        if len(Q) == 0: 
            continue
        c = Q.mean(axis=0)
        spread = float(np.mean(np.linalg.norm(Q - c, axis=1)))
        if len(Q) >= 3:
            try:
                hull_area = float(ConvexHull(Q).volume)
            except Exception:
                hull_area = 0.0
        else:
            hull_area = 0.0
        out[int(t)] = (tuple(map(float, c)), spread, hull_area)
    return out
