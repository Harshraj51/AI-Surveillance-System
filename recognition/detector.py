from insightface.app import FaceAnalysis

# Load InsightFace model (runs only once)
app = FaceAnalysis(
    name="buffalo_l",
    providers=["CPUExecutionProvider"]
)

app.prepare(ctx_id=0, det_size=(320, 320))


def detect_faces(frame):
    """
    Detect faces in a frame.
    Returns a list of face objects.
    """
    return app.get(frame)