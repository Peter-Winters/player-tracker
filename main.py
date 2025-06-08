import cv2 
import numpy as np
import os
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

    # frame path and YOLO model path
    frames_path = r"C:\Users\pfwin\Project Code\data\test_vid_frames"
    custom_model_path = r"C:\Users\pfwin\Project Code\data processing\fine_tuned_run\train\weights\best.pt"
    model = YOLO(custom_model_path)

    # Setup for Visualization
    frame_files = sorted([f for f in os.listdir(frames_path) if f.endswith('.jpg')])
    num_frames = len(frame_files) - 1
    first_img = cv2.imread(os.path.join(frames_path, f"{1:06d}.jpg"))
    height, width = first_img.shape[:2]
    seq_info = {
        "sequence_name": "test_clip_frames",
        "image_size": (height, width),
        "min_frame_idx": 1,
        "max_frame_idx": num_frames
    }

    output_video_path = "test.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = 30
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    def frame_callback(viz, frame_idx):
        img_path = os.path.join(frames_path, f"{frame_idx:06d}.jpg")
        image = cv2.imread(img_path)
        if image is None:
            print(f"Could not read {img_path}")
            return

        # Run YOLO detection
        results = model(image)
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
            if conf < 0.3: continue
            detections.append(Detection([x, y, w, h], conf, dummy_feature))

        # DeepSORT update
        tracker.predict()
        tracker.update(detections)

        # Visualization
        viz.set_image(image)
        #viz.draw_trackers(tracker.tracks)
        #viz.draw_detections(detections)
        viz.draw_trackers_ellipse(tracker.tracks)
        

        out.write(viz.viewer.image)

    viz = Visualization(seq_info, update_ms=30)
    viz.run(frame_callback)
    out.release()

    print(f"Tracking output saved to {output_video_path}")
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()