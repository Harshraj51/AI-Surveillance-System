from django.shortcuts import render, redirect
from django.http import StreamingHttpResponse

from .camera import VideoCamera
from .detector import detect_faces
from .models import KnownFace
from .utils import get_face_embedding

import cv2
import numpy as np


def gen(camera):
    while True:
        frame = camera.get_frame()

        if frame is None:
            continue

        yield (
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' +
            frame +
            b'\r\n'
        )


def video_feed(request):
    return StreamingHttpResponse(
        gen(VideoCamera()),
        content_type='multipart/x-mixed-replace; boundary=frame'
    )


def live_camera(request):
    return render(request, "recognition/live.html")


def register_face(request):

    if request.method == "POST":

        name = request.POST.get("name")
        image_file = request.FILES.get("image")

        if not name or not image_file:
            return render(
                request,
                "recognition/register.html",
                {"error": "Name and image are required."}
            )

        # Uploaded image ko OpenCV format me convert karo
        image_bytes = image_file.read()
        image_array = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

        if frame is None:
            return render(
                request,
                "recognition/register.html",
                {"error": "Invalid image."}
            )

        # Face detect karo
        faces = detect_faces(frame)

        if len(faces) == 0:
            return render(
                request,
                "recognition/register.html",
                {"error": "No face detected in the image."}
            )

        if len(faces) > 1:
            return render(
                request,
                "recognition/register.html",
                {"error": "Please upload an image containing only one face."}
            )

        # First/only face
        face = faces[0]

        # Face embedding generate karo
        embedding = get_face_embedding(face)

        if embedding is None:
            return render(
                request,
                "recognition/register.html",
                {"error": "Could not generate face embedding."}
            )

        # File pointer ko beginning par le jao
        image_file.seek(0)

        # Database me save
        KnownFace.objects.create(
            name=name,
            image=image_file,
            embedding=embedding
        )

        return redirect("live_camera")

    return render(request, "recognition/register.html")