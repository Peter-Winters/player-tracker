import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO

# DeepSORT imports
from deep_sort.tracker import Tracker
from deep_sort.detection import Detection
from deep_sort import nn_matching

from team_assigner.team_assigner import TeamAssigner

VIDEO_PATH   = r"C:\Users\pfwin\Project Code\data\vids\fixed_left.mp4"
YOLO_WEIGHTS = r"C:\Users\pfwin\Project Code\data processing\fine_tuned_run\train\weights\best.pt"
PITCH_IMAGE  = r"C:\Users\pfwin\Project Code\homography\pitch.jpg"
OUTPUT_VIDEO = r"C:\Users\pfwin\Project Code\data\vids\output.mp4"
# Left‑half pitch template (metres)
TEMPLATE = {
    0: (0, 0),    1: (0, 90),   2: (0, 41.75), 3: (0, 48.25),
    4: (13, 0),   5: (13, 90),  6: (20, 0),    7: (20, 90),
    8: (45, 0),   9: (45, 90),
}

LANDMARK_IDS = {
    "left_corner_flag": 0,       "right_corner_flag": 1,
    "left_goal_post_bottom": 2, "right_goal_post_bottom": 3,
    "left_13_intersection": 4,  "right_13_intersection": 5,
    "left_20_intersection": 6,  "right_20_intersection": 7,
    "left_45_intersection": 8,  "right_45_intersection": 9,
}

LANDMARK_ORDER = list(LANDMARK_IDS.keys())

# Pitch dimensions (metres) and template image size (pixels)
PITCH_W_M, PITCH_H_M = 145.0, 90.0
PITCH_W_PX, PITCH_H_PX = 412, 253


def collect_landmarks(video: Path):
    """Interactively pick landmark correspondences on the first frame."""
    cap = cv2.VideoCapture(str(video))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise IOError(f"Cannot read first frame from {video}")

    vis, idx, wait = frame.copy(), 0, True
    picked = {n: None for n in LANDMARK_ORDER}

    def mouse(ev, x, y, *_):
        nonlocal wait, idx, vis
        if ev == cv2.EVENT_LBUTTONDOWN and wait:
            picked[LANDMARK_ORDER[idx]] = (x, y)
            cv2.circle(vis, (x, y), 6, (0, 255, 0), 2)
            cv2.putText(
                vis,
                LANDMARK_ORDER[idx],
                (x + 10, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )
            wait = False

    cv2.namedWindow("Pick landmarks", cv2.WINDOW_GUI_NORMAL)
    cv2.setMouseCallback("Pick landmarks", mouse)
    while idx < len(LANDMARK_ORDER):
        disp = vis.copy()
        cv2.putText(
            disp,
            f"Click {LANDMARK_ORDER[idx]}  (s=skip,q=quit)",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )
        cv2.imshow("Pick landmarks", disp)
        k = cv2.waitKey(30) & 0xFF
        if not wait and k != ord("s"):
            idx += 1
            wait = True
        elif k in (ord("s"), ord("S")):
            idx += 1
            wait = True
        elif k in (ord("q"), ord("Q"), 27):
            break
    cv2.destroyWindow("Pick landmarks")
    return picked, frame.shape[:2]

def overlay_image(
    background: np.ndarray,
    overlay: np.ndarray,
    top_left: tuple[int, int] = (0, 0)
) -> None:
    """
    Alpha-composite *overlay* onto *background* in-place.

    Parameters
    ----------
    background : H×W×3   uint8  (BGR)
    overlay    : h×w×3/4 uint8  (BGR or BGRA)
    top_left   : (x, y)  = pixel where overlay’s top-left corner will land
    """
    x, y = top_left
    h, w = overlay.shape[:2]

    # Region of interest on the background where we’ll draw the overlay
    roi = background[y : y + h, x : x + w]

    if overlay.shape[2] == 4:          # BGRA → alpha-blend
        alpha = overlay[:, :, 3] / 255.0
        alpha = alpha[..., None]       # (h, w, 1) for broadcasting
        roi[:] = (alpha * overlay[:, :, :3] + (1 - alpha) * roi).astype(np.uint8)
    else:                              # BGR → direct copy
        roi[:] = overlay


def fit_homography(picked: dict[str, tuple[int, int]]):
    """Estimate homography H: image pixel → metric template coordinates."""
    img_pts, world_pts = [], []
    for n in LANDMARK_ORDER:
        if picked[n] is not None:
            img_pts.append(picked[n])
            world_pts.append(TEMPLATE[LANDMARK_IDS[n]])
    if len(img_pts) < 4:
        raise ValueError("Need at least four landmarks to compute homography")
    H, _ = cv2.findHomography(
        np.array(img_pts, np.float32),
        np.array(world_pts, np.float32),
        cv2.RANSAC,
        2.0,
    )
    return H


def metric_to_pitch_px(x_m: float, y_m: float,
                       w_px: int = PITCH_W_PX, h_px: int = PITCH_H_PX,
                       w_m: float = PITCH_W_M, h_m: float = PITCH_H_M):
    """Convert metric pitch coords → pixel coords on the pitch image."""
    u = x_m / w_m * w_px
    v = (h_m - y_m) / h_m * h_px  # flip vertical axis
    return int(round(u)), int(round(v))


def init_tracker():
    """Initialise a DeepSORT tracker with Euclidean appearance metric."""
    max_euclidean_distance = 0.7
    nn_budget = 100
    metric = nn_matching.NearestNeighborDistanceMetric(
        "euclidean", max_euclidean_distance, nn_budget
    )
    return Tracker(metric)


def main():
    # Load models
    model = YOLO(YOLO_WEIGHTS)
    tracker = init_tracker()
    dummy_feature = np.ones((1,), dtype=np.float32)  # placeholder appearance vec

    # Team colour assigner
    team_assigner = TeamAssigner()

    # Video
    video_path = Path(VIDEO_PATH)
    cap = cv2.VideoCapture(str(video_path))



    height, width = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)
    out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (width, height))

    # Homography
    picked, _ = collect_landmarks(video_path)
    H = fit_homography(picked)

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # -------------------------- 1. Run YOLO --------------------------- #
        results = model(frame)
        detections_np = []  # [x, y, w, h, conf, cls]
        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()
            confidences = result.boxes.conf.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy()
            for box, conf, cls in zip(boxes, confidences, classes):
                x1, y1, x2, y2 = box
                w, h = x2 - x1, y2 - y1
                detections_np.append([x1, y1, w, h, conf, cls])

        # --------------------- 2. Initialise k‑means ---------------------- #
        if team_assigner.kmeans is None and len(detections_np) >= 5:
            team_assigner.assign_team_color(frame, detections_np)

        # ------------------- 3. DeepSORT: predict/update ------------------ #
        detections_ds = [
            Detection([x, y, w, h], conf, dummy_feature)
            for x, y, w, h, conf, cls in detections_np
            if conf >= 0.3 and cls != 2  # filter low‑conf & non‑player
        ]

        tracker.predict()
        tracker.update(detections_ds)

        # --------------------- 4. Gather confirmed tracks ----------------- #
        pitch_positions = []  # (metric_x, metric_y, team_id)
        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 0:
                continue

            x, y, w, h = track.to_tlwh()  # pixel top‑left, width, height
            bottom_center = (x + w / 2, y + h)

            # Determine team
            team_id = team_assigner.get_player_team(
                frame, (x, y, w, h), track.track_id
            )
            color = tuple(map(int, team_assigner.team_colors[team_id]))

            # Draw bounding box & track ID
            cv2.rectangle(
                frame,
                (int(x), int(y)),
                (int(x + w), int(y + h)),
                color,
                2,
            )
            cv2.putText(
                frame,
                f"ID {track.track_id}",
                (int(x), int(y) - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
            )

            pitch_positions.append((bottom_center[0], bottom_center[1], team_id))

        # ---------------- 5. Project to metric & draw pitch --------------- #
        pitch_img = cv2.imread(PITCH_IMAGE)

        if pitch_positions:
            pts = (
                np.array([(p[0], p[1]) for p in pitch_positions], dtype=np.float32)
                .reshape(-1, 1, 2)
            )
            world_pts = cv2.perspectiveTransform(pts, H).reshape(-1, 2)

            for (x_m, y_m), (_, _, team_id) in zip(world_pts, pitch_positions):
                u, v = metric_to_pitch_px(x_m, y_m)
                if 0 <= u < PITCH_W_PX and 0 <= v < PITCH_H_PX:
                    col = (0, 0, 255) if team_id == 2 else (0, 0, 0)
                    cv2.circle(pitch_img, (u, v), 4, col, -1)

        # ---------------------------- 6. Display --------------------------- #
        pos = ((frame.shape[1] - pitch_img.shape[1]) // 2, 0)
        overlay_image(frame, pitch_img, pos)

        #cv2.imshow("combined.jpg", frame)
        out.write(frame)

        #cv2.imshow("Detections & Tracks", frame)
        #cv2.imshow("Pitch Detections", pitch_img)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
