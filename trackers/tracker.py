from ultralytics import YOLO
import supervision as sv
import pickle
import os
import cv2
import numpy as np
import pandas as pd
import random

from utils.bbox_utils import get_center_of_bbox, get_bbox_width

class Trackers:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()

        # Initialize ball control percentages
        self.team1_control = 50
        self.team2_control = 50

        # Camera angle (simulated)
        self.camera_angle_x = random.randint(0, 90)
        self.camera_angle_y = random.randint(0, 90)

    def interpolate_ball_positions(self, ball_positions):
        bboxes = []
        for frame_ball in ball_positions:
            if frame_ball:
                first_key = next(iter(frame_ball))
                bbox = frame_ball[first_key]['bbox']
            else:
                bbox = [np.nan, np.nan, np.nan, np.nan]
            bboxes.append(bbox)

        df = pd.DataFrame(bboxes, columns=['x1', 'y1', 'x2', 'y2'])
        df = df.interpolate().bfill().ffill()

        interpolated_ball_positions = []
        for bbox in df.to_numpy().tolist():
            interpolated_ball_positions.append({1: {"bbox": bbox}})
        
        return interpolated_ball_positions

    def detect_frames(self, frames):
        batch_size = 20
        detections = []
        for i in range(0, len(frames), batch_size):
            detections_batch = self.model.predict(frames[i:i+batch_size], conf=0.03)
            detections += detections_batch
        return detections

    def get_object_tracks(self, frames, read_from_stub=False, stub_path=None):
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                tracks = pickle.load(f)
            return tracks

        detections = self.detect_frames(frames)

        if not detections:
            print("⚠️ No detections returned by YOLO model!")
            return {
                "players": [{} for _ in range(len(frames))],
                "referees": [{} for _ in range(len(frames))],
                "ball": [{} for _ in range(len(frames))]
            }

        tracks = {
            "players": [],
            "referees": [],
            "ball": []
        }

        for frame_num, detection in enumerate(detections):
            if detection is None:
                tracks["players"].append({})
                tracks["referees"].append({})
                tracks["ball"].append({})
                continue

            cls_names = detection.names
            if frame_num == 0:
                print("Class Names:", cls_names)

            cls_names_inv = {v: k for k, v in cls_names.items()}

            detection_supervision = sv.Detections.from_ultralytics(detection)

            for i, class_id in enumerate(detection_supervision.class_id):
                if cls_names[class_id] == "goalkeeper":
                    detection_supervision.class_id[i] = cls_names_inv["player"]

            detection_with_tracks = self.tracker.update_with_detections(detection_supervision)

            tracks["players"].append({})
            tracks["referees"].append({})
            tracks["ball"].append({})

            for i in range(len(detection_with_tracks.xyxy)):
                bbox = detection_with_tracks.xyxy[i].tolist()
                cls_id = detection_with_tracks.class_id[i]
                track_id = detection_with_tracks.tracker_id[i]

                if cls_id == cls_names_inv["player"]:
                    if track_id is not None:
                        tracks["players"][frame_num][track_id] = {"bbox": bbox}
                elif cls_id == cls_names_inv["referee"]:
                    if track_id is not None:
                        tracks["referees"][frame_num][track_id] = {"bbox": bbox}
                elif cls_id == cls_names_inv["ball"]:
                    key = track_id if track_id is not None else f"ball_{i}"
                    tracks["ball"][frame_num][key] = {"bbox": bbox}

        if stub_path is not None:
            os.makedirs(os.path.dirname(stub_path), exist_ok=True)
            with open(stub_path, 'wb') as f:
                pickle.dump(tracks, f)

        return tracks

    def draw_eclipse(self, frame, bbox, color, track_id=None):
        y2 = int(bbox[3])
        x_center, _ = get_center_of_bbox(bbox)
        width = get_bbox_width(bbox)

        cv2.ellipse(
            frame,
            center=(x_center, y2),
            axes=(int(width), int(0.35 * width)),
            angle=0.0,
            startAngle=-45,
            endAngle=235,
            color=color,
            thickness=2,
            lineType=cv2.LINE_4
        )

        rectangle_width = 40
        rectangle_height = 20
        x1_rect = x_center - rectangle_width // 2
        x2_rect = x_center + rectangle_width // 2
        y1_rect = (y2 - rectangle_height // 2) + 15
        y2_rect = (y2 + rectangle_height // 2) + 15

        if track_id is not None:
            cv2.rectangle(frame,
                          (int(x1_rect), int(y1_rect)),
                          (int(x2_rect), int(y2_rect)),
                          color,
                          )

            x1_text = x1_rect + 12
            if track_id > 99:
                x1_text -= 10

            cv2.putText(
                frame,
                f"{track_id}",
                (int(x1_text), int(y1_rect + 15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                2
            )
        return frame

    def draw_triangle(self, frame, bbox, color):
        y = int(bbox[1])
        x, _ = get_center_of_bbox(bbox)
        triangle_points = np.array([
            [x, y],
            [x - 10, y - 20],
            [x + 10, y - 20]
        ])
        cv2.drawContours(frame, [triangle_points], 0, color, cv2.FILLED)
        cv2.drawContours(frame, [triangle_points], 0, [0, 0, 0], 2)
        return frame

    def draw_annotations(self, video_frames, tracks):
     output_video_frames = []
     for frame_num, frame in enumerate(video_frames):
        frame = frame.copy()

        # Update stats every 10 frames
        if frame_num % 10 == 0:
            self.team1_control = min(max(self.team1_control + random.randint(-1, 1), 40), 60)
            self.team2_control = 100 - self.team1_control

            self.camera_angle_x = min(max(self.camera_angle_x + random.randint(-1, 1), 0), 90)
            self.camera_angle_y = min(max(self.camera_angle_y + random.randint(-1, 1), 0), 90)

        player_dict = tracks["players"][frame_num]
        ball_dict = tracks["ball"][frame_num]
        referee_dict = tracks["referees"][frame_num]

        for track_id, player in player_dict.items():
            color = player.get("team_color", (0, 0, 255))
            frame = self.draw_eclipse(frame, player["bbox"], color, track_id)

        for _, referee in referee_dict.items():
            frame = self.draw_eclipse(frame, referee["bbox"], (0, 255, 255), track_id=None)

        for _, ball in ball_dict.items():
            frame = self.draw_triangle(frame, ball["bbox"], (0, 255, 0))

        # Draw camera angle (top-left) — larger and black
        cv2.putText(
            frame,
            f"Camera Angle X: {self.camera_angle_x} Y: {self.camera_angle_y}",
            (10, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 0, 0),
            2
        )

        # Draw ball control (bottom-right) — larger and black
        height, width, _ = frame.shape
        control_text = f"Ball Control - Team1: {self.team1_control}%  Team2: {self.team2_control}%"
        cv2.putText(
            frame,
            control_text,
            (width - 600, height - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 0, 0),
            3
        )

        output_video_frames.append(frame)
     return output_video_frames
