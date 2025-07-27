import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO

# DeepSORT imports
from deep_sort.tracker import Tracker
from deep_sort.detection import Detection
from deep_sort import nn_matching

from team_assigner.team_assigner import TeamAssigner

# ─── file paths ───────────────────────────────────────────────────────────────
VIDEO_PATH        = r"C:\Users\pfwin\Project Code\data\vids\test_vid2.mp4"
PLAYER_WEIGHTS    = r"C:\Users\pfwin\Project Code\data processing\fine_tuned_run\train\weights\best.pt"
PITCH_KP_WEIGHTS  = r"G:\My Drive\data\pitch_kpt_run\train2\weights\best.pt"    
PITCH_IMAGE       = r"C:\Users\pfwin\Project Code\homography\pitch.jpg"
OUTPUT_VIDEO      = r"C:\Users\pfwin\Project Code\data\vids\moving_out.mp4"

# ─── metric template in the model’s 19-keypoint order ─────────────────────────
TEMPLATE_METRIC = np.array([
    (  0.0, 90.0),    # 0  LTC
    (  0.0,  0.0),    # 1  LBC
    (145.0, 90.0),    # 2  RTC
    (145.0,  0.0),    # 3  RBC
    (100.0,  0.0),    # 4  RB45
    (100.0, 90.0),    # 5  RT45
    ( 45.0, 90.0),    # 6  LT45
    ( 45.0,  0.0),    # 7  LB45
    ( 72.5,  0.0),    # 8  BM
    ( 20.0, 58.0),    # 9  LTA
    ( 20.0, 32.0),    # 10 LBA
    (125.0, 58.0),    # 11 RTA
    (125.0, 32.0),    # 12 RBA
    ( 72.5, 90.0),    # 13 TM
    ( 72.5, 45.0),    # 14 MM
    (  0.0, 48.25),   # 15 LTG
    (  0.0, 41.75),   # 16 LBG
    (145.0, 48.25),   # 17 RTG
    (145.0, 41.75)    # 18 RBG
], dtype=np.float32)

# ─── pitch & template image sizes ─────────────────────────────────────────────
PITCH_W_M, PITCH_H_M = 145.0, 90.0
PITCH_W_PX, PITCH_H_PX = 412, 253

# ─── helpers ──────────────────────────────────────────────────────────────────
def detect_pitch_keypoints(model: YOLO, frame: np.ndarray) -> np.ndarray:
    """Return (19,2) array of key-points from first detection in *frame*."""
    res = model(frame, verbose=False)
    if not res or res[0].keypoints is None:
        raise RuntimeError("Pitch key-points not detected in first frame.")
    kps = res[0].keypoints.xy[0].cpu().numpy().astype(np.float32)
    if kps.shape[0] < TEMPLATE_METRIC.shape[0]:
        raise ValueError("Model returned fewer key-points than expected.")
    return kps[:TEMPLATE_METRIC.shape[0]]           # (19, 2)

def fit_homography(src_pts_px: np.ndarray) -> np.ndarray:
    """Pixel points → metric TEMPLATE_METRIC."""
    H, _ = cv2.findHomography(src_pts_px, TEMPLATE_METRIC, cv2.RANSAC, 2.0)
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
        alpha = overlay[:,:,3:4] / 255.0
        roi[:] = (alpha * overlay[:,:,:3] + (1-alpha) * roi).astype(np.uint8)
    else:                                           # BGR
        roi[:] = overlay

def init_tracker() -> Tracker:
    metric = nn_matching.NearestNeighborDistanceMetric("euclidean", 0.7, 100)
    return Tracker(metric)

# ─── main ─────────────────────────────────────────────────────────────────────
def main():
    # models
    player_model = YOLO(PLAYER_WEIGHTS)
    kp_model     = YOLO(PITCH_KP_WEIGHTS)

    tracker = init_tracker()
    dummy_feature = np.ones((1,), dtype=np.float32)   # placeholder appearance vec
    team_assigner = TeamAssigner()

    # video I/O
    cap = cv2.VideoCapture(str(Path(VIDEO_PATH)))
    height, width = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fps = cap.get(cv2.CAP_PROP_FPS)
    out = cv2.VideoWriter(OUTPUT_VIDEO, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    # first frame → homography
    ok, first = cap.read()
    if not ok:
        raise IOError("Cannot read first frame from video.")
    src_pts_px = detect_pitch_keypoints(kp_model, first)
    H = fit_homography(src_pts_px)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)              # rewind

    pitch_base = cv2.imread(str(Path(PITCH_IMAGE)))
    if pitch_base is None:
        raise IOError("Cannot load PITCH_IMAGE.")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # ── 1. player detections ──────────────────────────────────────────
        result = player_model(frame, verbose=False)[0]
        boxes = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        clss  = result.boxes.cls.cpu().numpy()

        detections = [Detection([x1, y1, x2-x1, y2-y1], cf, dummy_feature)
                      for (x1,y1,x2,y2), cf, cls in zip(boxes, confs, clss)
                      if cf >= 0.3 and cls != 2]

        # ── 2. team-colour k-means init ───────────────────────────────────
        if team_assigner.kmeans is None and len(detections) >= 5:
            det_np = [[d.tlwh[0], d.tlwh[1], d.tlwh[2], d.tlwh[3], 1.0, 0] for d in detections]
            team_assigner.assign_team_color(frame, det_np)

        # ── 3. DeepSORT predict/update ────────────────────────────────────
        tracker.predict()
        tracker.update(detections)

        pitch_img = pitch_base.copy()
        pitch_positions = []     # (x_px, y_px, team_id)

        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 0:
                continue
            x, y, w, h = track.to_tlwh()
            foot = (x + w/2, y + h)

            team_id = team_assigner.get_player_team(frame, (x, y, w, h), track.track_id)
            colour  = tuple(map(int, team_assigner.team_colors[team_id]))

            cv2.rectangle(frame, (int(x), int(y)), (int(x+w), int(y+h)), colour, 2)
            cv2.putText(frame, f"ID {track.track_id}", (int(x), int(y)-6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 2)

            pitch_positions.append((foot[0], foot[1], team_id))

        # ── 4. project to metric & draw on pitch overlay ──────────────────
        if pitch_positions:
            pts_px = np.array([(p[0], p[1]) for p in pitch_positions], np.float32).reshape(-1,1,2)
            pts_m  = cv2.perspectiveTransform(pts_px, H).reshape(-1,2)
            for (x_m, y_m), (_, _, team_id) in zip(pts_m, pitch_positions):
                u,v = metric_to_pitch_px(x_m, y_m)
                if 0 <= u < PITCH_W_PX and 0 <= v < PITCH_H_PX:
                    col = (0,0,255) if team_id == 2 else (0,0,0)
                    cv2.circle(pitch_img, (u,v), 4, col, -1)

        # ── 5. composite & output ─────────────────────────────────────────
        overlay_image(frame, pitch_img, ((width - PITCH_W_PX)//2, 0))
        out.write(frame)

        #-- debug windows (optional) --
        # cv2.imshow("overlay", frame)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

    cap.release(); out.release(); cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
