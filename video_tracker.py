import cv2
import numpy as np
from deep_sort.tracker import Tracker
from deep_sort.detection import Detection
from deep_sort import nn_matching
from ultralytics import YOLO
from application_util.visualization import Visualization

def main():
    # DeepSORT config
    max_euclidean_distance = 0.7
    nn_budget = 100
    metric = nn_matching.NearestNeighborDistanceMetric("euclidean", max_euclidean_distance, nn_budget)
    tracker = Tracker(metric)
    dummy_feature = np.ones((1,), dtype=np.float32)

    # Video input path
    video_path = r"C:\Users\pfwin\Project Code\data\vids\test_vid2.mp4"
    cap = cv2.VideoCapture(video_path)

    # YOLO model
    custom_model_path = r"C:\Users\pfwin\Project Code\data processing\fine_tuned_run\train\weights\best.pt"
    model = YOLO(custom_model_path)

    # Output setup
    ret, first_frame = cap.read()
    if not ret:
        print("Failed to read video.")
        return

    height, width = first_frame.shape[:2]
    output_video_path = "test.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # Visualization
    seq_info = {
        "sequence_name": "test_clip_video",
        "image_size": (height, width),
        "min_frame_idx": 0,
        "max_frame_idx": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
    }
    viz = Visualization(seq_info, update_ms=int(1000 / fps))

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)
        detections_np = []
        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()
            confidences = result.boxes.conf.cpu().numpy()
            for box, conf in zip(boxes, confidences):
                x1, y1, x2, y2 = box
                w = x2 - x1
                h = y2 - y1
                detections_np.append([x1, y1, w, h, conf])

        detections = []
        for det in detections_np:
            x, y, w, h, conf = det
            if conf < 0.3:
                continue
            detections.append(Detection([x, y, w, h], conf, dummy_feature))

        tracker.predict()
        tracker.update(detections)

        viz.set_image(frame)
        viz.draw_trackers_ellipse(tracker.tracks)

        out.write(viz.viewer.image)
        frame_idx += 1

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Tracking output saved to {output_video_path}")

if __name__ == "__main__":
    main()