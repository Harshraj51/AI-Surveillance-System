import numpy as np


def get_face_embedding(face):
    """
    Extract embedding from an InsightFace face object.
    """

    if face is None:
        return None

    embedding = face.embedding

    if embedding is None:
        return None

    # Convert numpy array to normal Python list
    return embedding.astype(float).tolist()


def cosine_similarity(embedding1, embedding2):
    """
    Calculate cosine similarity between two face embeddings.
    """

    a = np.array(embedding1, dtype=np.float32)
    b = np.array(embedding2, dtype=np.float32)

    denominator = np.linalg.norm(a) * np.linalg.norm(b)

    if denominator == 0:
        return 0.0

    return float(np.dot(a, b) / denominator)