from sklearn.cluster import KMeans
import numpy as np

class TeamAssigner:
    def __init__(self):
        self.team_colors = {}
        self.player_team_dict = {}
        self.kmeans = None

    def get_clustering_model(self, image):
        image_2d = image.reshape(-1, 3)
        kmeans = KMeans(n_clusters=2, init="k-means++", n_init=10)
        kmeans.fit(image_2d)
        return kmeans

    def get_player_color(self, frame, bbox):
        x1, y1, w, h = bbox
        x1, y1 = int(x1), int(y1)
        x2, y2 = int(x1 + w), int(y1 + h)
        image = frame[y1:y2, x1:x2]

        if image.size == 0:
            return np.array([0, 0, 0])  # fallback

        top_half = image[0:int(image.shape[0] / 2), :]

        # reduce bbox size to try and isolate jersey color
        left   = int(0.10 * w)
        right  = int(0.90 * w)

        top_half = top_half[:, left:right]

        if top_half.size == 0:
            return np.array([0, 0, 0])  # fallback
        

        kmeans = self.get_clustering_model(top_half)
        labels = kmeans.labels_
        clustered = labels.reshape(top_half.shape[:2])

        corners = [clustered[0, 0], clustered[0, -1], clustered[-1, 0], clustered[-1, -1]]
        non_player_cluster = max(set(corners), key=corners.count)
        player_cluster = 1 - non_player_cluster

        return kmeans.cluster_centers_[player_cluster]

    def assign_team_color(self, frame, detections):
        player_colors = []
        bboxes = []

        for det in detections:
            x, y, w, h, conf, cls = det
            if conf < 0.3:
                continue
            bboxes.append((x, y, w, h))
            player_colors.append(self.get_player_color(frame, (x, y, w, h)))

        if not player_colors:
            return

        kmeans = KMeans(n_clusters=2, init="k-means++", n_init=10)
        kmeans.fit(player_colors)

        self.kmeans = kmeans
        # self.team_colors[1] = kmeans.cluster_centers_[0]
        # self.team_colors[2] = kmeans.cluster_centers_[1]

        # manually assigned colours for now
        self.team_colors[2] = np.array([0, 0, 255], dtype=np.float32)   # red
        self.team_colors[1] = np.array([0, 0,   0], dtype=np.float32)   # black

    def get_player_team(self, frame, bbox, player_id):
        if player_id in self.player_team_dict:
            return self.player_team_dict[player_id]

        color = self.get_player_color(frame, bbox)
        team_id = self.kmeans.predict(color.reshape(1, -1))[0] + 1

        self.player_team_dict[player_id] = team_id
        return team_id
