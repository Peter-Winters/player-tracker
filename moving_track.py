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
VIDEO_PATH        = r"data\vids\clip3.mp4"
OUTPUT_VIDEO      = r"C:\Users\pfwin\Project Code\clip3__.mp4"
PLAYER_WEIGHTS    = r"G:\My Drive\data\player_imgs\runs\detect\train4\weights\best.pt"
PITCH_KP_WEIGHTS  = r"G:\My Drive\data\pitch_kpt_run\train2\weights\best.pt"
PITCH_IMAGE       = r"deep_sort\pitch.png"

# classification model
cls_model_path = r"G:\My Drive\train2\weights\best.pt"

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
PITCH_W_PX, PITCH_H_PX = 412, 253

HOMO_EVERY_N = 1          # refresh homography every N frames
TAIL_LEN = 70  # keep last ~40 positions

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
                  top_left: tuple[int, int] = (0, 0)) -> None:
    x, y = top_left; h, w = overlay.shape[:2]
    roi = background[y:y+h, x:x+w]
    if overlay.shape[2] == 4:                       # BGRA
        alpha = overlay[:, :, 3:4] / 255.0
        roi[:] = (alpha * overlay[:, :, :3] + (1 - alpha) * roi).astype(np.uint8)
    else:                                           # BGR
        roi[:] = overlay

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

def big_h_jump(H_prev, H_cur, thr=0.12):
    if H_prev is None or H_cur is None: return False
    A = H_prev / H_prev[2,2]; B = H_cur / H_cur[2,2]
    return np.linalg.norm(A - B, ord='fro') > thr

# ─── main ─────────────────────────────────────────────────────────────────────
def main():
    # models
    player_model = YOLO(PLAYER_WEIGHTS)
    kp_model     = YOLO(PITCH_KP_WEIGHTS)
    cls_model = YOLO(cls_model_path)

    tracker = init_tracker()
    dummy_feature = np.ones((1,), dtype=np.float32)

    foot_trails_px  = defaultdict(lambda: deque(maxlen=TAIL_LEN))  # broadcast view
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

    # ── main loop ───────────────────────────────────────────────────────────
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # refresh homography periodically
        if frame_idx % HOMO_EVERY_N == 0:
            kps = detect_pitch_keypoints(kp_model, frame, pitch_template)
            if kps is not None:
                H_new = fit_homography(kps, pitch_template)
                if H_new is not None:
                    H = H_new

        # player detections
        result = player_model(frame, verbose=False)[0]
        boxes = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        clss  = result.boxes.cls.cpu().numpy()

        detections = [
            Detection([x1, y1, x2 - x1, y2 - y1], cf, dummy_feature)
            for (x1, y1, x2, y2), cf, cls in zip(boxes, confs, clss)
            if cf >= 0.3 and cls != 2
        ]

        # DeepSORT update
        tracker.predict()
        tracker.update(detections)

        pitch_img = pitch_base.copy()
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
            #foot_trails_px[track.track_id].append((foot_x, foot_y))
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
        # for tid, pts in foot_trails_px.items():
        #     if len(pts) < 2:
        #         continue
        #     base_col = COLOUR.get(track_team.get(tid, 0), (255, 255, 255))
        #     n = len(pts)
        #     for i in range(1, n):
        #         a = i / n  # older segments are fainter
        #         col = (int(base_col[0]*a), int(base_col[1]*a), int(base_col[2]*a))
        #         cv2.line(trail_overlay, pts[i-1], pts[i], col, 1, lineType=cv2.LINE_AA)

            # # blend once
            # cv2.addWeighted(trail_overlay, 1.0, frame, 1.0, 0, frame)


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
                    cv2.circle(pitch_img, (u, v), 4, COLOUR[team_id+2], -1)
                # store for tail
                pitch_trails_px[track_id].append((u, v))

        trail_overlay = np.zeros_like(frame, dtype=np.uint8)

        for tid, pts in pitch_trails_px.items():
            if len(pts) < 2:
                continue
            team_id = track_team.get(tid, 0)
            base_col = COLOUR.get(team_id + 2, (200, 200, 200))

            # optional fading
            for i in range(1, len(pts)):
                a = i / len(pts)  # older segments fainter
                col = (int(base_col[0]*a), int(base_col[1]*a), int(base_col[2]*a))
                cv2.line(pitch_img, pts[i-1], pts[i], col, 1, lineType=cv2.LINE_AA)

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

                    cv2.line(trail_overlay, (x1, y1), (x2, y2), col, 1, lineType=cv2.LINE_AA)

        # single blend after drawing all trails
        cv2.addWeighted(trail_overlay, 1.0, frame, 1.0, 0, frame)

        # output
        overlay_image(frame, pitch_img, ((width - PITCH_W_PX) // 2, 0))
        out.write(frame)

        frame_idx += 1

        cv2.imshow("overlay", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release(); out.release(); cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
