from django.urls import path
from . import views

urlpatterns = [
    path("", views.live_camera, name="live_camera"),
    path("video/", views.video_feed, name="video_feed"),
    path("register/", views.register_face, name="register_face"),
]