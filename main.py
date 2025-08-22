import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO

# DeepSORT imports
from deep_sort.tracker import Tracker
from deep_sort.detection import Detection
from deep_sort import nn_matching

from collections import defaultdict, deque

# ─── file paths ───────────────────────────────────────────────────────────────
VIDEO_PATH        = "clip5.mp4"
OUTPUT_VIDEO      = "out\\clip5_analysis.mp4"

PLAYER_WEIGHTS    = "models\\player_detector.pt"
PITCH_KP_WEIGHTS  = "models\\pitch_keypoint_detector.pt"
CLASSIFIER_WEIGHTS = "models\\team_classifier.pt"

PITCH_IMAGE       = "pitch.png"

# ─── metric template in the model’s 19-keypoint order ─────────────────────────
TEMPLATE_METRIC = np.array([
    (  0.0, 90.0),(  0.0,  0.0),(145.0, 90.0),(145.0,  0.0),
    (100.0,  0.0),(100.0, 90.0),( 45.0, 90.0),( 45.0,  0.0),
    ( 72.5,  0.0),( 20.0, 58.0),( 20.0, 32.0),(125.0, 58.0),
    (125.0, 32.0),( 72.5, 90.0),( 72.5, 45.0),(  0.0, 48.25),
    (  0.0, 41.75),(145.0, 48.25),(145.0, 41.75)
], dtype=np.float32)

# ─── pitch & template image sizes ─────────────────────────────────────────────
PITCH_W_M, PITCH_H_M = 145.0, 85.0
#PITCH_W_PX, PITCH_H_PX = 412, 253
PITCH_W_PX, PITCH_H_PX = 618, 380

HOMO_EVERY_N = 1          # refresh homography every N frames
TAIL_LEN = 120  # keep last 120 positions

# ─── helpers ──────────────────────────────────────────────────────────────────
def build_pitch_template(width_m: float,
                         height_m: float,
                         template_ref: np.ndarray = TEMPLATE_METRIC
                        ) -> np.ndarray:
    """Return a pitch template scaled to the given metric dimensions."""
    # Reference dimensions
    REF_W, REF_H = 145.0, 90.0

    # Scale factors for each axis
    sx, sy = width_m / REF_W, height_m / REF_H

    # Apply anisotropic scaling and preserve dtype
    return (template_ref * np.array([sx, sy], dtype=np.float32)).astype(np.float32)

def detect_pitch_keypoints(model: YOLO, frame: np.ndarray, template) -> np.ndarray | None:
    """Return (19,2) pixel coords or None if detection fails."""
    res = model(frame, verbose=False)
    if not res or res[0].keypoints is None:
        return None
    kps = res[0].keypoints.xy[0].cpu().numpy().astype(np.float32)
    if kps.shape[0] < template.shape[0]:
        return None
    return kps[:template.shape[0]]

def fit_homography(src_pts_px: np.ndarray, template) -> np.ndarray | None:
    H, _ = cv2.findHomography(src_pts_px, template, cv2.RANSAC, 3.0, maxIters=10000)
    return H

def metric_to_pitch_px(x_m, y_m,
                       w_px=PITCH_W_PX, h_px=PITCH_H_PX,
                       w_m=PITCH_W_M,  h_m=PITCH_H_M):
    u = x_m / w_m * w_px
    v = (h_m - y_m) / h_m * h_px      # flip vertical axis
    return int(round(u)), int(round(v))

def overlay_image(background: np.ndarray, overlay: np.ndarray,
                  top_left: tuple[int, int] = (0, 0),
                  opacity: float = 1.0) -> None:
    x, y = top_left
    h, w = overlay.shape[:2]
    roi = background[y:y+h, x:x+w]

    if overlay.shape[2] == 4:  # BGRA with per-pixel alpha
        # scale existing alpha by requested opacity
        alpha = (overlay[:, :, 3:4].astype(np.float32) / 255.0) * float(opacity)
        roi[:] = (alpha * overlay[:, :, :3].astype(np.float32) +
                  (1.0 - alpha) * roi.astype(np.float32)).astype(np.uint8)
    else:  # BGR: uniform opacity
        cv2.addWeighted(roi, 1.0 - float(opacity), overlay, float(opacity), 0, dst=roi)

def init_tracker() -> Tracker:
    metric = nn_matching.NearestNeighborDistanceMetric("euclidean", 0.7, 100)
    return Tracker(metric)

def safe_crop(im, x, y, w, h):
    """Clamp box to image bounds and return None if it is empty."""
    H, W, _ = im.shape

    # round and clamp
    x1 = max(0, int(round(x)))
    y1 = max(0, int(round(y)))
    x2 = min(W, int(round(x + w)))
    y2 = min(H, int(round(y + h)))

    if x2 - x1 < 2 or y2 - y1 < 2:      # too small → skip
        return None
    return im[y1:y2, x1:x2]

def _build_heatmap_image(points_uv, width, height, sigma=17, colormap=cv2.COLORMAP_TURBO):
    """
    points_uv: list of (u, v) pitch-pixel coords
    returns BGR heatmap image (H x W x 3) with zeros where there is no density
    """
    heat = np.zeros((height, width), dtype=np.float32)
    if not points_uv:
        return np.zeros((height, width, 3), dtype=np.uint8)

    # accumulate hits
    for u, v in points_uv:
        if 0 <= u < width and 0 <= v < height:
            heat[v, u] += 1.0

    # smooth & normalize
    heat = cv2.GaussianBlur(heat, (0, 0), sigmaX=sigma, sigmaY=sigma)
    if heat.max() > 0:
        heat_norm = (heat / heat.max() * 255).astype(np.uint8)
    else:
        heat_norm = heat.astype(np.uint8)

    # colorize (prebuilt OpenCV colormap)
    heat_color = cv2.applyColorMap(heat_norm, colormap)

    # zero out areas with no density so blend doesn't tint whole image
    heat_color[heat_norm == 0] = 0
    return heat_color

def _overlay_heatmap_on_pitch(pitch_img_bgr, heat_bgr, alpha=0.65):
    """Blend heat over pitch only where heat != 0."""
    out = pitch_img_bgr.copy()
    mask = (heat_bgr[:,:,0] + heat_bgr[:,:,1] + heat_bgr[:,:,2]) > 0
    out[mask] = cv2.addWeighted(pitch_img_bgr[mask], 1 - alpha, heat_bgr[mask], alpha, 0)
    return out


# ─── main ─────────────────────────────────────────────────────────────────────
def main():
    # models
    player_model = YOLO(PLAYER_WEIGHTS)
    kp_model     = YOLO(PITCH_KP_WEIGHTS)
    cls_model = YOLO(CLASSIFIER_WEIGHTS)

    tracker = init_tracker()
    dummy_feature = np.ones((1,), dtype=np.float32)

    pitch_trails_px = defaultdict(lambda: deque(maxlen=TAIL_LEN))  # 2D pitch view
    foot_trails_world = defaultdict(lambda: deque(maxlen=TAIL_LEN)) # world coordinates (deal with cam movement)
    track_team      = {}  # remember each track's last seen team colour (0/1)
                  
    COLOUR = {0: (0, 90, 0), # green
            1: (0,   0, 60),  # maroon
            2: (0, 140, 0), # light green
            3: (0,   0, 120),}  # light maroon

    # video I/O
    cap = cv2.VideoCapture(str(Path(VIDEO_PATH)))
    height, width = (int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                     int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
    fps = cap.get(cv2.CAP_PROP_FPS)
    out = cv2.VideoWriter(OUTPUT_VIDEO,
                          cv2.VideoWriter_fourcc(*'mp4v'),
                          fps, (width, height))

    pitch_base = cv2.imread(str(Path(PITCH_IMAGE)))
    if pitch_base is None:
        raise IOError("Cannot load PITCH_IMAGE.")
    
    pitch_template = build_pitch_template(PITCH_W_M, PITCH_H_M)

    # ── get an initial homography ───────────────────────────────
    H = None
    while H is None:
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError("Could not find a frame with detectable key-points.")
        kps = detect_pitch_keypoints(kp_model, frame, pitch_template)
        if kps is not None:
            H = fit_homography(kps, pitch_template)

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)             # rewind to start

    all_pitch_points = []                # [(u, v)]
    team_pitch_points = defaultdict(list)  # team_id -> [(u, v)]

    # ── main loop ───────────────────────────────────────────────────────────
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        valid_H = False

        # refresh homography periodically
        if frame_idx % HOMO_EVERY_N == 0:
            kps = detect_pitch_keypoints(kp_model, frame, pitch_template)
            if kps is not None:
                H_new = fit_homography(kps, pitch_template)
                if H_new is not None:
                    H = H_new
                    valid_H = True

        # player detections
        result = player_model(frame, verbose=False)[0]
        boxes = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        clss  = result.boxes.cls.cpu().numpy()

        detections = [
            Detection([x1, y1, x2 - x1, y2 - y1], cf, dummy_feature)
            for (x1, y1, x2, y2), cf, cls in zip(boxes, confs, clss)
            if cf >= 0.3 and cls != 2 and cls != 0
        ]

        # DeepSORT update
        tracker.predict()
        tracker.update(detections)

        # Remove trails for tracks that no longer exist
        active_ids = {t.track_id for t in tracker.tracks}  # current DeepSORT tracks

        for store in (pitch_trails_px, foot_trails_world):
            for tid in list(store.keys()):
                if tid not in active_ids:
                    store.pop(tid, None)

        for tid in list(track_team.keys()):
            if tid not in active_ids:
                track_team.pop(tid, None)

        pitch_bg = pitch_base                    # pitch image
        minimap_draw_bgr  = np.zeros_like(pitch_base, dtype=np.uint8)           # colors of dots/trails
        minimap_alpha     = np.zeros((PITCH_H_PX, PITCH_W_PX), dtype=np.uint8)   # per-pixel alpha for dots/trails

        pitch_positions = []     # (x_px, y_px, team_id, track_id)

        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 0:
                continue
            x, y, w, h = track.to_tlwh()
            foot = (x + w / 2, y + h)
            foot_x = int(x + w / 2)
            foot_y = int(y + h)
            crop = safe_crop(frame, x, y, w, h)
            if crop is None:          # box too small or off-frame
                continue

            team_id = int(cls_model(crop, verbose=False)[0].probs.top1)
            track_team[track.track_id] = team_id
            pitch_positions.append((foot[0], foot[1], team_id, track.track_id)) # (x_px, y_px, team_id, track_id)
            colour = COLOUR[team_id]

            axes = (int(w), int(h * 0.1))       # (major, minor) radii
            cv2.ellipse(frame,
                (foot_x, foot_y),          # centre
                axes,                      # axes lengths
                0,                         # rotation angle
                -45, 235,                  # full ellipse
                colour, 2,                 # colour, thickness 
                lineType=cv2.LINE_4)       

        trail_overlay = np.zeros_like(frame, dtype=np.uint8)
        current_minimap_dots = []  # (u, v, team_id)

        if valid_H:
            # project to metric & draw on 2d pitch
            if pitch_positions:
                pts_px = np.array([(p[0], p[1]) for p in pitch_positions],
                                np.float32).reshape(-1, 1, 2)
                pts_m  = cv2.perspectiveTransform(pts_px, H).reshape(-1, 2)

                for (x_m, y_m), (_, _, team_id, track_id) in zip(pts_m, pitch_positions):
                    # guard bad homographies
                    if not np.isfinite(x_m) or not np.isfinite(y_m):
                        continue
                    foot_trails_world[track_id].append((float(x_m), float(y_m)))
                    u, v = metric_to_pitch_px(x_m, y_m)
                    if 0 <= u < PITCH_W_PX and 0 <= v < PITCH_H_PX:
    
                        current_minimap_dots.append((u, v, team_id)) # save positions for current frame


                    # store for tail
                    pitch_trails_px[track_id].append((u, v))
                    all_pitch_points.append((u, v))
                    team_pitch_points[team_id].append((u, v))
                
            H_inv = None if H is None else np.linalg.inv(H)

            if H_inv is not None:
                for tid, pts_world in foot_trails_world.items():
                    if len(pts_world) < 2:
                        continue

                    # project world to current image
                    pts_world_np = np.array(pts_world, np.float32).reshape(-1, 1, 2)
                    pts_img = cv2.perspectiveTransform(pts_world_np, H_inv).reshape(-1, 2)

                    base_col = COLOUR.get(track_team.get(tid, 0), (255, 255, 255))
                    n = len(pts_img)

                    # draw fading polyline
                    for i in range(1, n):
                        a = i / n  # older segments fainter
                        col = (int(base_col[0]*a), int(base_col[1]*a), int(base_col[2]*a))

                        x1, y1 = np.rint(pts_img[i-1]).astype(int)
                        x2, y2 = np.rint(pts_img[i]).astype(int)

                        # clamp to frame to avoid OpenCV errors
                        x1 = max(0, min(x1, width-1)); y1 = max(0, min(y1, height-1))
                        x2 = max(0, min(x2, width-1)); y2 = max(0, min(y2, height-1))

                        cv2.line(trail_overlay, (x1, y1), (x2, y2), col, 2, lineType=cv2.LINE_AA)

            # single blend after drawing all trails
            cv2.addWeighted(trail_overlay, 1.0, frame, 1.0, 0, frame)

            # output minimap
            TRAIL_THICK = 2
            for tid, pts in pitch_trails_px.items():
                if len(pts) < 2:
                    continue
                team_id = track_team.get(tid, 0)
                base_col = COLOUR.get(team_id, (255, 255, 255))
                n = len(pts)
                for i in range(1, n):
                    a = i / n                        # 0..1 (older => lighter)
                    alpha_val = int(max(1, min(255, round(255 * a))))
                    cv2.line(minimap_draw_bgr, pts[i-1], pts[i], base_col, TRAIL_THICK, lineType=cv2.LINE_AA)
                    cv2.line(minimap_alpha,    pts[i-1], pts[i], alpha_val, TRAIL_THICK, lineType=cv2.LINE_AA)

                # draw current dots last, fully opaque, to sit above trails
                for (u, v, team_id) in current_minimap_dots:
                    cv2.circle(minimap_draw_bgr, (u, v), 6, COLOUR[team_id], -1, lineType=cv2.LINE_AA)
                    cv2.circle(minimap_alpha,    (u, v), 6, 255,            -1, lineType=cv2.LINE_AA)


            panel_pos = ((width - PITCH_W_PX) // 2, 0)
            h, w = pitch_bg.shape[:2]

            # Build a single BGRA panel
            panel_bgr = pitch_bg.copy()

            # wherever there are drawn graphics, replace the colors with the drawn colors
            mask = minimap_alpha > 0
            panel_bgr[mask] = minimap_draw_bgr[mask]

            # per-pixel alpha: 0.65 for background, but graphics use their own alpha (up to 255)
            panel_alpha = np.full((h, w), int(1.0 * 255), dtype=np.uint8)
            panel_alpha[mask] = minimap_alpha[mask]

            # convert to BGRA and overlay once
            panel_bgra = cv2.cvtColor(panel_bgr, cv2.COLOR_BGR2BGRA)
            panel_bgra[:, :, 3] = panel_alpha
            overlay_image(frame, panel_bgra, panel_pos, opacity=1.0)

        out.write(frame)

        frame_idx += 1

        cv2.imshow("overlay", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    try:
        # overall heatmap
        heat_all = _build_heatmap_image(all_pitch_points, PITCH_W_PX, PITCH_H_PX, sigma=9, colormap=cv2.COLORMAP_TURBO)
        heat_all_over = _overlay_heatmap_on_pitch(pitch_base, heat_all, alpha=0.65)

        base = Path(OUTPUT_VIDEO)
        out_dir = base.parent
        stem = base.stem

        overall_path = out_dir / f"{stem}_heatmap_all.png"
        cv2.imwrite(str(overall_path), heat_all_over)

        # per-team heatmaps
        for team_id, pts in team_pitch_points.items():
            if not pts:
                continue
            heat_t = _build_heatmap_image(pts, PITCH_W_PX, PITCH_H_PX, sigma=9, colormap=cv2.COLORMAP_TURBO)
            heat_t_over = _overlay_heatmap_on_pitch(pitch_base, heat_t, alpha=0.65)
            team_path = out_dir / f"{stem}_heatmap_team{team_id}.png"
            cv2.imwrite(str(team_path), heat_t_over)

        print(f"[heatmap] saved: {overall_path}")
    except Exception as e:
        print(f"[heatmap] skipped due to error: {e}")


    cap.release(); out.release(); cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
