# from sklearn.cluster import KMeans
# import numpy as np
# import cv2

# # class TeamAssigner:
# #     def __init__(self):
# #         self.team_colors = {}
# #         self.player_team_dict = {}
# #         self.kmeans = None

# #     def get_clustering_model(self, image):
# #         image_2d = image.reshape(-1, 3)
# #         kmeans = KMeans(n_clusters=2, init="k-means++", n_init=10)
# #         kmeans.fit(image_2d)
# #         return kmeans
    
# #     def get_player_color(self, frame, bbox):
# #         """
# #         Return the dominant BGR jersey colour inside `bbox` using 2-cluster K-means
# #         on the cropped top-half region (head excluded).  No corner-pixel heuristic.
# #         """
# #         x1, y1, w, h = map(int, bbox)
# #         x2, y2 = x1 + w, y1 + h
# #         image  = frame[y1:y2, x1:x2]

# #         if image.size == 0:
# #             return np.array([0, 0, 0], dtype=np.float32)   # fallback
        
# #         #colour_bbox = (int(x1+(0.25*w)), int(y1+(0.6*h)), int(x1 + (w*0.85)), int(y1 + (h*0.25)))

# #         # ── focus on the jersey band (upper half) ─────────────────────────────
# #         top_half = image[: int(image.shape[0] * 0.6)]
# #         if top_half.size == 0:
# #             return np.array([0, 0, 0], dtype=np.float32)
# #         #crop top 10% to avoid the head
# #         top_half = top_half[int(0.25 * top_half.shape[0]):, :]

# #         # reduce bbox size to try and isolate jersey color
# #         left   = int(0.25 * w)
# #         right  = int(0.75 * w)

# #         top_half = top_half[:, left:right]

# #         # ── 2-cluster K-means on colour pixels ────────────────────────────────
# #         pixels = top_half.reshape(-1, 3).astype(np.float32)
# #         kmeans = KMeans(n_clusters=2, n_init=5).fit(pixels)

# #         # pick the cluster that has *more pixels* (most frequent label)
# #         labels, counts = np.unique(kmeans.labels_, return_counts=True)
# #         dominant_label = labels[np.argmax(counts)]

# #         return kmeans.cluster_centers_[dominant_label].astype(np.float32)
    
# #     def assign_team_color(self, frame, detections):
# #         player_colors = []
# #         bboxes = []

# #         for det in detections:
# #             x, y, w, h, conf, cls = det
# #             if conf < 0.3:
# #                 continue
# #             bboxes.append((x, y, w, h))
# #             player_colors.append(self.get_player_color(frame, (x, y, w, h)))

# #         if not player_colors:
# #             return

# #         kmeans = KMeans(n_clusters=2, init="k-means++", n_init=10)
# #         kmeans.fit(player_colors)

# #         self.kmeans = kmeans
# #         self.team_colors[1] = kmeans.cluster_centers_[0]
# #         self.team_colors[2] = kmeans.cluster_centers_[1]

# #     def get_player_team(self, frame, bbox, player_id):
# #         if player_id in self.player_team_dict:
# #             return self.player_team_dict[player_id]

# #         color = self.get_player_color(frame, bbox)
# #         color = color.astype(self.kmeans.cluster_centers_.dtype, copy=False)
# #         team_id = self.kmeans.predict(color.reshape(1, -1))[0] + 1

# #         self.player_team_dict[player_id] = team_id
# #         return team_id


# class TeamAssigner:
#     def __init__(self):
#         self.team_colors_bgr = {}       # for rectangles, dots, etc.
#         self.player_team_dict = {}
#         self.kmeans = None              # trained in Lab space

#     # ──────────────────────────────────────────────────────────────────────
#     # 1.  jersey-colour extractor  →  returns a single Lab vector
#     # ──────────────────────────────────────────────────────────────────────
#     def get_player_color(self, frame, bbox):
#         """
#         Extract the dominant jersey colour inside `bbox` and return it in
#         CIELAB space (float32, shape (3,)).

#         Crop logic is unchanged; only BGR→Lab + Lab-KMeans is new.
#         """
#         x1, y1, w, h = map(int, bbox)
#         x2, y2 = x1 + w, y1 + h
#         image = frame[y1:y2, x1:x2]
        
#         # show the cropped bbox image
#         cv2.imshow("Cropped Image", image)
        

#         if image.size == 0:
#             return np.zeros(3, dtype=np.float32)  # fallback Lab (black)

#         # ── focus on jersey band ─────────────────────────────────────────
#         top_half = image[: int(image.shape[0] * 0.6)]
#         if top_half.size == 0:
#             return np.zeros(3, dtype=np.float32)

#         top_half = top_half[int(0.25 * top_half.shape[0]):, :]     # trim head
#         left, right = int(0.25 * w), int(0.75 * w)                 # squeeze sides
#         top_half = top_half[:, left:right]
#         if top_half.size == 0:
#             return np.zeros(3, dtype=np.float32)

#         # ── NEW: convert to Lab & cluster ───────────────────────────────
#         lab = cv2.cvtColor(top_half, cv2.COLOR_BGR2LAB).astype(np.float32)
#         pixels_lab = lab.reshape(-1, 3)
#         kmeans = KMeans(n_clusters=2, n_init=5).fit(pixels_lab)

#         labels, counts = np.unique(kmeans.labels_, return_counts=True)
#         dominant = labels[np.argmax(counts)]
#         return kmeans.cluster_centers_[dominant].astype(np.float32)   # ← Lab

#     # ──────────────────────────────────────────────────────────────────────
#     # 2.  compute two team colours (fit in Lab, store Lab + BGR)
#     # ──────────────────────────────────────────────────────────────────────
#     def assign_team_color(self, frame, detections):
#         player_colors_lab = []

#         for det in detections:
#             x, y, w, h, conf, cls = det
#             if conf < 0.30:
#                 continue
#             player_colors_lab.append(self.get_player_color(frame, (x, y, w, h)))

#         if not player_colors_lab:
#             return  # nothing to learn from

#         self.kmeans = KMeans(n_clusters=2, init="k-means++", n_init=10)
#         self.kmeans.fit(player_colors_lab)           # ← Lab space

#         # store centroids both ways
#         for idx, lab_centroid in enumerate(self.kmeans.cluster_centers_, start=1):
#             self.team_colors_bgr[idx] = cv2.cvtColor(
#                 lab_centroid.reshape(1, 1, 3).astype(np.uint8),
#                 cv2.COLOR_Lab2BGR
#             ).reshape(3).astype(np.float32)

#     # ──────────────────────────────────────────────────────────────────────
#     # 3.  predict a player's team using Lab colour
#     # ──────────────────────────────────────────────────────────────────────
#     def get_player_team(self, frame, bbox, player_id):
#         if player_id in self.player_team_dict:
#             return self.player_team_dict[player_id]

#         color_lab = self.get_player_color(frame, bbox)
#         color_lab = color_lab.astype(self.kmeans.cluster_centers_.dtype, copy=False)
#         team_id = int(self.kmeans.predict(color_lab.reshape(1, -1))[0]) + 1

#         self.player_team_dict[player_id] = team_id
#         return team_id


import cv2
import numpy as np
import skimage.color
from PIL import Image


class TeamAssigner:
    """
    Assigns a stable team-id (1, 2, …) to each DeepSORT track_id
    using a predefined palette of kit colours and ΔE voting.
    """

    def __init__(self, team_palette_bgr: dict[int, list[list[int]]]):
        """
        Parameters
        ----------
        team_palette_bgr : dict
            { team_id : [ [R,G,B], [R,G,B], … ] }  — 1-to-N reference colours per kit.
        """
        # convert reference palettes to Lab once
        self.team_colors_lab = {
            t: [skimage.color.rgb2lab(np.array(c, np.float32)[None] / 255.0)[0]
                for c in palette]
            for t, palette in team_palette_bgr.items()
        }
        # average BGR per team for drawing rectangles / dots later
        self.team_colors_bgr = {
            t: np.mean(palette, axis=0).astype(np.float32)
            for t, palette in team_palette_bgr.items()
        }

        self.player_team_dict: dict[int, int] = {}          # track_id → team_id


    # palette extractor
    def _palette(self, frame_rgb: np.ndarray, bbox, interval=(0, 5)) -> list[list[int]]:
        """
        Crop a centre-torso patch, quantise to a 216-colour web-palette,
        and return the n°1→n°5 most frequent RGB triples.
        """
        x1, y1, w, h = map(int, bbox)
        crop = frame_rgb[y1:y1 + h, x1:x1 + w]
        if crop.size == 0:
            return []

        # centre-torso filter
        h_c, w_c = crop.shape[:2]
        cfx1, cfx2 = max((w_c // 2) - (w_c // 5), 1), (w_c // 2) + (w_c // 5)
        cfy1, cfy2 = max((h_c // 3) - (h_c // 5), 1), (h_c // 3) + (h_c // 5)
        torso = crop[cfy1:cfy2, cfx1:cfx2]

        pil = Image.fromarray(torso)
        reduced = pil.convert("P", palette=Image.Palette.WEB)
        pal = reduced.getpalette()                         # flat list  [r,g,b,r,g,b,…]
        pal = [pal[3 * n:3 * n + 3] for n in range(256)]   # → [[r,g,b], …]
        counts = [(cnt, pal[idx]) for cnt, idx in reduced.getcolors()]
        counts.sort(key=lambda x: x[0], reverse=True)      # freq-sorted
        return [rgb for _, rgb in counts[interval[0]:interval[1]]]


    # pick a team from one palette
    def _team_by_palette(self, palette_rgb: list[list[int]]) -> int:
        """
        Majority vote over ΔE-nearest matches.
        """
        if not palette_rgb:
            return 1  # default / fallback team

        palette_lab = [
            skimage.color.rgb2lab(np.array(c, np.float32)[None] / 255.0)[0]
            for c in palette_rgb
        ]
        votes = []
        for lab in palette_lab:
            dists = {
                t: min(skimage.color.deltaE_cie76(lab, ref) for ref in refs)
                for t, refs in self.team_colors_lab.items()
            }
            votes.append(min(dists, key=dists.get))        # team with smallest ΔE
        return max(votes, key=votes.count)                 # majority vote


    # public API used by tracker
    def get_player_team(self, frame_bgr, bbox, track_id):
        """
        Returns a stable team-id (int ≥ 1) for the given track.
        Caches the result so each player is classified only once.
        """
        if track_id in self.player_team_dict:
            return self.player_team_dict[track_id]

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        palette_rgb = self._palette(frame_rgb, bbox)
        team_id = self._team_by_palette(palette_rgb)

        self.player_team_dict[track_id] = team_id
        return team_id