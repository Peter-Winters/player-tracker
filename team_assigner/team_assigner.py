from sklearn.cluster import KMeans
import numpy as np
import cv2

# class TeamAssigner:
#     def __init__(self):
#         self.team_colors = {}
#         self.player_team_dict = {}
#         self.kmeans = None

#     def get_clustering_model(self, image):
#         image_2d = image.reshape(-1, 3)
#         kmeans = KMeans(n_clusters=2, init="k-means++", n_init=10)
#         kmeans.fit(image_2d)
#         return kmeans
    
#     def get_player_color(self, frame, bbox):
#         """
#         Return the dominant BGR jersey colour inside `bbox` using 2-cluster K-means
#         on the cropped top-half region (head excluded).  No corner-pixel heuristic.
#         """
#         x1, y1, w, h = map(int, bbox)
#         x2, y2 = x1 + w, y1 + h
#         image  = frame[y1:y2, x1:x2]

#         if image.size == 0:
#             return np.array([0, 0, 0], dtype=np.float32)   # fallback
        
#         #colour_bbox = (int(x1+(0.25*w)), int(y1+(0.6*h)), int(x1 + (w*0.85)), int(y1 + (h*0.25)))

#         # ── focus on the jersey band (upper half) ─────────────────────────────
#         top_half = image[: int(image.shape[0] * 0.6)]
#         if top_half.size == 0:
#             return np.array([0, 0, 0], dtype=np.float32)
#         #crop top 10% to avoid the head
#         top_half = top_half[int(0.25 * top_half.shape[0]):, :]

#         # reduce bbox size to try and isolate jersey color
#         left   = int(0.25 * w)
#         right  = int(0.75 * w)

#         top_half = top_half[:, left:right]

#         # ── 2-cluster K-means on colour pixels ────────────────────────────────
#         pixels = top_half.reshape(-1, 3).astype(np.float32)
#         kmeans = KMeans(n_clusters=2, n_init=5).fit(pixels)

#         # pick the cluster that has *more pixels* (most frequent label)
#         labels, counts = np.unique(kmeans.labels_, return_counts=True)
#         dominant_label = labels[np.argmax(counts)]

#         return kmeans.cluster_centers_[dominant_label].astype(np.float32)
    
#     def assign_team_color(self, frame, detections):
#         player_colors = []
#         bboxes = []

#         for det in detections:
#             x, y, w, h, conf, cls = det
#             if conf < 0.3:
#                 continue
#             bboxes.append((x, y, w, h))
#             player_colors.append(self.get_player_color(frame, (x, y, w, h)))

#         if not player_colors:
#             return

#         kmeans = KMeans(n_clusters=2, init="k-means++", n_init=10)
#         kmeans.fit(player_colors)

#         self.kmeans = kmeans
#         self.team_colors[1] = kmeans.cluster_centers_[0]
#         self.team_colors[2] = kmeans.cluster_centers_[1]

#     def get_player_team(self, frame, bbox, player_id):
#         if player_id in self.player_team_dict:
#             return self.player_team_dict[player_id]

#         color = self.get_player_color(frame, bbox)
#         color = color.astype(self.kmeans.cluster_centers_.dtype, copy=False)
#         team_id = self.kmeans.predict(color.reshape(1, -1))[0] + 1

#         self.player_team_dict[player_id] = team_id
#         return team_id


class TeamAssigner:
    def __init__(self):
        self.team_colors_bgr = {}       # for rectangles, dots, etc.
        self.player_team_dict = {}
        self.kmeans = None              # trained in Lab space

    # ──────────────────────────────────────────────────────────────────────
    # 1.  jersey-colour extractor  →  returns a single Lab vector
    # ──────────────────────────────────────────────────────────────────────
    def get_player_color(self, frame, bbox):
        """
        Extract the dominant jersey colour inside `bbox` and return it in
        CIELAB space (float32, shape (3,)).

        Crop logic is unchanged; only BGR→Lab + Lab-KMeans is new.
        """
        x1, y1, w, h = map(int, bbox)
        x2, y2 = x1 + w, y1 + h
        image = frame[y1:y2, x1:x2]
        
        # show the cropped bbox image
        cv2.imshow("Cropped Image", image)
        

        if image.size == 0:
            return np.zeros(3, dtype=np.float32)  # fallback Lab (black)

        # ── focus on jersey band ─────────────────────────────────────────
        top_half = image[: int(image.shape[0] * 0.6)]
        if top_half.size == 0:
            return np.zeros(3, dtype=np.float32)

        top_half = top_half[int(0.25 * top_half.shape[0]):, :]     # trim head
        left, right = int(0.25 * w), int(0.75 * w)                 # squeeze sides
        top_half = top_half[:, left:right]
        if top_half.size == 0:
            return np.zeros(3, dtype=np.float32)

        # ── NEW: convert to Lab & cluster ───────────────────────────────
        lab = cv2.cvtColor(top_half, cv2.COLOR_BGR2LAB).astype(np.float32)
        pixels_lab = lab.reshape(-1, 3)
        kmeans = KMeans(n_clusters=2, n_init=5).fit(pixels_lab)

        labels, counts = np.unique(kmeans.labels_, return_counts=True)
        dominant = labels[np.argmax(counts)]
        return kmeans.cluster_centers_[dominant].astype(np.float32)   # ← Lab

    # ──────────────────────────────────────────────────────────────────────
    # 2.  compute two team colours (fit in Lab, store Lab + BGR)
    # ──────────────────────────────────────────────────────────────────────
    def assign_team_color(self, frame, detections):
        player_colors_lab = []

        for det in detections:
            x, y, w, h, conf, cls = det
            if conf < 0.30:
                continue
            player_colors_lab.append(self.get_player_color(frame, (x, y, w, h)))

        if not player_colors_lab:
            return  # nothing to learn from

        self.kmeans = KMeans(n_clusters=2, init="k-means++", n_init=10)
        self.kmeans.fit(player_colors_lab)           # ← Lab space

        # store centroids both ways
        for idx, lab_centroid in enumerate(self.kmeans.cluster_centers_, start=1):
            self.team_colors_bgr[idx] = cv2.cvtColor(
                lab_centroid.reshape(1, 1, 3).astype(np.uint8),
                cv2.COLOR_Lab2BGR
            ).reshape(3).astype(np.float32)

    # ──────────────────────────────────────────────────────────────────────
    # 3.  predict a player's team using Lab colour
    # ──────────────────────────────────────────────────────────────────────
    def get_player_team(self, frame, bbox, player_id):
        if player_id in self.player_team_dict:
            return self.player_team_dict[player_id]

        color_lab = self.get_player_color(frame, bbox)
        color_lab = color_lab.astype(self.kmeans.cluster_centers_.dtype, copy=False)
        team_id = int(self.kmeans.predict(color_lab.reshape(1, -1))[0]) + 1

        self.player_team_dict[player_id] = team_id
        return team_id