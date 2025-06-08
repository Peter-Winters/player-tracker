# vim: expandtab:ts=4:sw=4
import numpy as np
import colorsys
from .image_viewer import ImageViewer
import cv2

def create_unique_color_float(tag, hue_step=0.41):
    """Create a unique RGB color code for a given track id (tag).

    The color code is generated in HSV color space by moving along the
    hue angle and gradually changing the saturation.

    Parameters
    ----------
    tag : int
        The unique target identifying tag.
    hue_step : float
        Difference between two neighboring color codes in HSV space (more
        specifically, the distance in hue channel).

    Returns
    -------
    (float, float, float)
        RGB color code in range [0, 1]

    """
    h, v = (tag * hue_step) % 1, 1. - (int(tag * hue_step) % 4) / 5.
    r, g, b = colorsys.hsv_to_rgb(h, 1., v)
    return r, g, b


def create_unique_color_uchar(tag, hue_step=0.41):
    """Create a unique RGB color code for a given track id (tag).

    The color code is generated in HSV color space by moving along the
    hue angle and gradually changing the saturation.

    Parameters
    ----------
    tag : int
        The unique target identifying tag.
    hue_step : float
        Difference between two neighboring color codes in HSV space (more
        specifically, the distance in hue channel).

    Returns
    -------
    (int, int, int)
        RGB color code in range [0, 255]

    """
    r, g, b = create_unique_color_float(tag, hue_step)
    return int(255*r), int(255*g), int(255*b)


class NoVisualization(object):
    """
    A dummy visualization object that loops through all frames in a given
    sequence to update the tracker without performing any visualization.
    """

    def __init__(self, seq_info):
        self.frame_idx = seq_info["min_frame_idx"]
        self.last_idx = seq_info["max_frame_idx"]

    def set_image(self, image):
        pass

    def draw_groundtruth(self, track_ids, boxes):
        pass

    def draw_detections(self, detections):
        pass

    def draw_trackers(self, trackers):
        pass

    def run(self, frame_callback):
        while self.frame_idx <= self.last_idx:
            frame_callback(self, self.frame_idx)
            self.frame_idx += 1


class Visualization(object):
    """
    This class shows tracking output in an OpenCV image viewer.
    """

    def __init__(self, seq_info, update_ms):
        image_shape = seq_info["image_size"][::-1]
        aspect_ratio = float(image_shape[1]) / image_shape[0]
        image_shape = 1024, int(aspect_ratio * 1024)
        self.viewer = ImageViewer(
            update_ms, image_shape, "Figure %s" % seq_info["sequence_name"])
        self.viewer.thickness = 2
        self.frame_idx = seq_info["min_frame_idx"]
        self.last_idx = seq_info["max_frame_idx"]

    def run(self, frame_callback):
        self.viewer.run(lambda: self._update_fun(frame_callback))

    def _update_fun(self, frame_callback):
        if self.frame_idx > self.last_idx:
            return False  # Terminate
        frame_callback(self, self.frame_idx)
        self.frame_idx += 1
        return True

    def set_image(self, image):
        self.viewer.image = image

    def draw_groundtruth(self, track_ids, boxes):
        self.viewer.thickness = 2
        for track_id, box in zip(track_ids, boxes):
            self.viewer.color = create_unique_color_uchar(track_id)
            self.viewer.rectangle(*box.astype(np.int64), label=str(track_id))

    def draw_detections(self, detections):
        self.viewer.thickness = 2
        self.viewer.color = 0, 0, 255
        for i, detection in enumerate(detections):
            self.viewer.rectangle(*detection.tlwh)

    def draw_trackers(self, tracks):
        self.viewer.thickness = 2
        for track in tracks:
            if not track.is_confirmed() or track.time_since_update > 0:
                continue
            self.viewer.color = create_unique_color_uchar(track.track_id)
            self.viewer.rectangle(
                *track.to_tlwh().astype(np.int64), label=str(track.track_id))
            # self.viewer.gaussian(track.mean[:2], track.covariance[:2, :2],
            #                      label="%d" % track.track_id)
#

    def draw_trackers_ellipse(self, tracks):
        self.viewer.thickness = 2
        for track in tracks:
            if not track.is_confirmed() or track.time_since_update > 0:
                continue
            self.viewer.color = create_unique_color_uchar(track.track_id)
            
            x, y, w, h = track.to_tlwh().astype(np.int32)
            
            # Feet position: bottom center of bbox
            feet_x = x + w // 2
            feet_y = y + h

            
            axis_length = (int(w), int(h * 0.15))
            
            # Angle: 0 means ellipse aligned with image axes
            angle = 0
            
            # Draw the ellipse (thickness=-1 for filled, or any int for outline)
            cv2.ellipse(
                self.viewer.image,                # Image to draw on
                (feet_x, feet_y),                 # Center of ellipse
                axis_length,                      # Axes lengths
                angle,                            # Angle of rotation
                -45, 235,                           # Start and end angle
                self.viewer.color,                # Color
                self.viewer.thickness,             # Thickness
                lineType=cv2.LINE_4             
            )


            # Draw track number under ellipse
            label = str(track.track_id)
            font = cv2.FONT_HERSHEY_SIMPLEX
            min_font_scale = 0.4
            max_font_scale = 0.6

            base_font_scale = h / 60.0  # 60 is an empirically chosen constant; tune as needed
            font_scale = max(base_font_scale, min_font_scale)
            if max_font_scale is not None:
                font_scale = min(font_scale, max_font_scale)

            #font_scale = 0.8
            font_thickness = 1
            label_size, _ = cv2.getTextSize(label, font, font_scale, font_thickness)
            label_x = feet_x - label_size[0] // 2
            #label_y = feet_y + axis_length[1] + label_size[1] + 6  # Just below the ellipse

            offset = int(h * 0.1)  # adjust based on player size
            label_y = y - offset

            cv2.putText(
                self.viewer.image,
                label,
                (label_x, label_y),
                font,
                font_scale,
                self.viewer.color,
                font_thickness,
                cv2.LINE_AA
            )