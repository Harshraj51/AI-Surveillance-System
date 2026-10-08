from django.db import models


class KnownFace(models.Model):
    name = models.CharField(max_length=100)

    image = models.ImageField(
        upload_to="known_faces/",
        blank=True,
        null=True
    )

    embedding = models.JSONField()

    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.name