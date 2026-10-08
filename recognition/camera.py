import cv2
import numpy as np

from .detector import detect_faces
from .models import KnownFace
from .utils import cosine_similarity


class VideoCamera:
    def __init__(self):
        self.video = cv2.VideoCapture(0)

        self.video.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.video.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        self.frame_count = 0
        self.last_faces = []

        # Load known faces once
        self.known_faces = list(KnownFace.objects.all())

    def __del__(self):
        if self.video.isOpened():
            self.video.release()

    def get_frame(self):

        success, frame = self.video.read()

        if not success:
            return None

        self.frame_count += 1

        # AI detection every 3rd frame
        if self.frame_count % 3 == 0:
            self.last_faces = detect_faces(frame)

        faces = self.last_faces

        for face in faces:

            x1, y1, x2, y2 = face.bbox.astype(int)

            # Default
            name = "Unknown"
            color = (0, 0, 255)

            # Live face embedding
            live_embedding = face.embedding

            best_score = 0

            # Compare with registered faces
            for known_face in self.known_faces:

                score = cosine_similarity(
                    live_embedding,
                    known_face.embedding
                )

                if score > best_score:
                    best_score = score
                    name = known_face.name

            # Recognition threshold
            if best_score < 0.45:
                name = "Unknown"
                color = (0, 0, 255)
            else:
                color = (0, 255, 0)

            # Draw face box
            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                color,
                2
            )

            # Display name
            label = f"{name} ({best_score:.2f})"

            cv2.putText(
                frame,
                label,
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2
            )

        # Encode frame
        _, jpeg = cv2.imencode(
            ".jpg",
            frame,
            [cv2.IMWRITE_JPEG_QUALITY, 70]
        )

        return jpeg.tobytes()