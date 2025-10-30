import os
import numpy as np
from PIL import Image
from mtcnn import MTCNN
from keras_facenet import FaceNet
from sklearn.metrics.pairwise import cosine_similarity
import argparse
import pickle
from tqdm import tqdm

# -------------------------------------------------------------
# Block: Load the pre-trained FaceNet model
# -------------------------------------------------------------
def load_facenet_model():
    embedder = FaceNet()
    return embedder

# -------------------------------------------------------------
# Block: Extract face from an image file using MTCNN
# -------------------------------------------------------------
def extract_face(image_path, required_size=(160, 160)):
    image = Image.open(image_path)
    image = image.convert('RGB')
    pixels = np.asarray(image)
    detector = MTCNN()
    results = detector.detect_faces(pixels)
    if not results:
        return None
    x1, y1, width, height = results[0]['box']
    x1, y1 = abs(x1), abs(y1)
    x2, y2 = x1 + width, y1 + height
    face = pixels[y1:y2, x1:x2]
    face_image = Image.fromarray(face)
    face_image = face_image.resize(required_size)
    return np.asarray(face_image)

# -------------------------------------------------------------
# Block: Get FaceNet embedding vector for a cropped face
# -------------------------------------------------------------
def get_embedding(model, face_pixels):
    embeddings = model.embeddings([face_pixels])
    return embeddings[0]

# -------------------------------------------------------------
# Block: Compare two embedding vectors using cosine similarity
# -------------------------------------------------------------
def compare_embeddings(emb1, emb2):
    """
    Computes the cosine similarity between two embeddings.
    Args:
        emb1, emb2: np.ndarray 1D feature vectors.
    Returns:
        float: similarity score (higher = more similar; range [-1, 1])
    """
    score = cosine_similarity([emb1], [emb2])[0][0]
    return score

# -------------------------------------------------------------
# Block: Build and load gallery embeddings (for recognition)
# -------------------------------------------------------------

def build_gallery_embeddings(dataset_dir, cache_path=None, required_size=(160, 160), limit_per_person=None):
    """
    Build embedding list (person_name, embedding, image_path) for all images in dataset_dir.
    Optionally limit number of images per person. Optionally cache to pickle (cache_path).
    """
    detector = MTCNN()
    model = FaceNet()
    data = []
    people = sorted(os.listdir(dataset_dir))
    for person in tqdm(people, desc="Processing Identities"):
        person_dir = os.path.join(dataset_dir, person)
        if not os.path.isdir(person_dir):
            continue
        image_files = [f for f in os.listdir(person_dir) if f.lower().endswith('.jpg')]
        if limit_per_person:
            image_files = image_files[:limit_per_person]
        for img_name in image_files:
            img_path = os.path.join(person_dir, img_name)
            try:
                image = Image.open(img_path).convert('RGB')
                pixels = np.asarray(image)
                results = detector.detect_faces(pixels)
                if not results:
                    continue
                x1, y1, width, height = results[0]['box']
                x1, y1 = abs(x1), abs(y1)
                x2, y2 = x1 + width, y1 + height
                face = pixels[y1:y2, x1:x2]
                face_image = Image.fromarray(face).resize(required_size)
                face_array = np.asarray(face_image)
                embedding = model.embeddings([face_array])[0]
                data.append({'person': person, 'embedding': embedding, 'image_path': img_path})
            except Exception:
                continue
    if cache_path:
        with open(cache_path, 'wb') as f:
            pickle.dump(data, f)
    return data


def load_gallery_embeddings(cache_path):
    """
    Load gallery embeddings from pickle cache file.
    Returns list of dicts: {'person', 'embedding', 'image_path'}
    """
    with open(cache_path, 'rb') as f:
        data = pickle.load(f)
    return data

# -------------------------------------------------------------
# Block: Command-line interface for verification or gallery build
# -------------------------------------------------------------

def main():
    """
    CLI entry-point.
    - Default: Face verification between two images.
    - With --build-gallery: Build and cache embeddings for a dataset directory.
    """
    parser = argparse.ArgumentParser(description='FaceNet Utilities')
    subparsers = parser.add_subparsers(dest='command')

    # Verification command (default)
    verify = subparsers.add_parser('verify', help='Verify if two images are the same person')
    verify.add_argument('--img1', type=str, required=True, help='Path to first image')
    verify.add_argument('--img2', type=str, required=True, help='Path to second image')

    # Build gallery
    build = subparsers.add_parser('build-gallery', help='Build gallery embeddings cache from dataset directory')
    build.add_argument('--dataset_dir', type=str, required=True, help='Dataset root (folders per person)')
    build.add_argument('--cache_path', type=str, default='lfw_gallery_embeddings.pkl', help='Output pickle path')
    build.add_argument('--limit_per_person', type=int, default=None, help='Optional limit per identity')

    args = parser.parse_args()

    if args.command == 'build-gallery':
        print(f"Building gallery from: {args.dataset_dir}")
        data = build_gallery_embeddings(args.dataset_dir, cache_path=args.cache_path, limit_per_person=args.limit_per_person)
        print(f"Saved {len(data)} embeddings to {args.cache_path}")
        return

    if args.command == 'verify' or args.command is None:
        model = load_facenet_model()
        face1 = extract_face(args.img1)
        face2 = extract_face(args.img2)
        if face1 is None:
            print(f"No face detected in {args.img1}")
            return
        if face2 is None:
            print(f"No face detected in {args.img2}")
            return
        emb1 = get_embedding(model, face1)
        emb2 = get_embedding(model, face2)
        score = compare_embeddings(emb1, emb2)
        print(f"\nImage 1: {os.path.basename(args.img1)}")
        print(f"Image 2: {os.path.basename(args.img2)}")
        print(f"Cosine similarity: {score:.4f}")
        threshold = 0.5
        if score > threshold:
            print("Result: Same person ✅")
        else:
            print("Result: Different persons ❌")

if __name__ == "__main__":
    main()
