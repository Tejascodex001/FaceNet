import streamlit as st
import numpy as np
from PIL import Image, ImageDraw
import tempfile
import os
import time
import pickle
from facenet_assignment import (
    load_facenet_model, extract_face, get_embedding, compare_embeddings,
    build_gallery_embeddings, load_gallery_embeddings
)
from mtcnn import MTCNN

st.set_page_config(page_title="FaceNet Face Recognition Demo", layout="wide")

# ----- Simple theming + cursor improvements -----
st.markdown(
    """
    <style>
    .app-title { font-size: 28px; font-weight: 700; margin-bottom: 6px; }
    .app-sub { color: #6b7280; margin-bottom: 20px; }
    .badge { display: inline-block; padding: 2px 8px; border-radius: 999px; font-size: 12px; }
    .badge-ok { background: #DCFCE7; color: #166534; }
    .badge-warn { background: #FEF9C3; color: #854D0E; }
    .section { padding: 12px 16px; background: #F9FAFB; border: 1px solid #E5E7EB; border-radius: 8px; }
    /* Make interactive elements show pointer cursor */
    button, [role=tab], .stSelectbox, .stRadio, .stSlider, .stFileUploader, .stCameraInput, label { cursor: pointer !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="app-title">FaceNet: Face Verification and Recognition</div>', unsafe_allow_html=True)
st.markdown('<div class="app-sub">Upload images or use your webcam to verify or recognize faces using FaceNet embeddings.</div>', unsafe_allow_html=True)

MODE_VERIFICATION = "Face Verification (1:1)"
MODE_RECOGNITION = "Face Recognition (Who is this?)"

GALLERY_DIR = "extracted_archive/lfw_funneled/"
DEFAULT_CACHE_PATH = "lfw_gallery_embeddings.pkl"
GALLERY_LOAD_ERROR = None
CUSTOM_DEFAULT_PATH = "/home/tejas/Projects/Facenet/custom_gallery_embeddings.pkl"

# ---------- KNOWN FACES GALLERY CACHE ----------
@st.cache_resource(show_spinner=False)
def get_model_detector():
    return load_facenet_model(), MTCNN()

@st.cache_resource(show_spinner=False)
def get_cached_gallery(cache_path: str):
    global GALLERY_LOAD_ERROR
    GALLERY_LOAD_ERROR = None
    if cache_path and os.path.exists(cache_path):
        try:
            return load_gallery_embeddings(cache_path)
        except Exception as e:
            GALLERY_LOAD_ERROR = f"Failed to load '{cache_path}': {e}"
            return None
    return None

# Sidebar: menu + settings + gallery source
gallery_ready_placeholder = st.sidebar.empty()
with st.sidebar:
    st.markdown("### App Menu")
    mode = st.selectbox("Mode", [MODE_VERIFICATION, MODE_RECOGNITION])
    st.markdown("---")
    st.markdown("### Recognition Settings")
    sim_threshold = st.slider("Similarity threshold", 0.0, 1.0, 0.5, 0.01)
    topn = st.slider("Top-N matches to display", 1, 5, 3)
    st.markdown("---")
    st.markdown("### Gallery Source")
    default_index = 1 if os.path.exists(CUSTOM_DEFAULT_PATH) else 0
    gallery_choice = st.radio("Pick gallery", ["LFW cache", "Custom cache (.pkl)"], index=default_index)
    cache_path = DEFAULT_CACHE_PATH
    if gallery_choice == "Custom cache (.pkl)":
        st.caption("Use an existing local .pkl or upload one.")
        custom_source = st.radio("Custom source", ["Local path", "Upload"], horizontal=True)
        if custom_source == "Local path":
            local_path = st.text_input(
                "Path to embeddings .pkl",
                value=CUSTOM_DEFAULT_PATH,
            )
            cache_path = local_path if local_path else None
        else:
            uploaded = st.file_uploader("Upload embeddings pickle", type=["pkl"], accept_multiple_files=False)
            if uploaded is not None:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pkl") as tfp:
                    tfp.write(uploaded.read())
                    cache_path = tfp.name
            else:
                cache_path = None

# Load selected gallery (no building in UI)
_gallery_data = get_cached_gallery(cache_path)
if _gallery_data is not None:
    gallery_ready_placeholder.markdown('<span class="badge badge-ok">Gallery cache: Ready</span>', unsafe_allow_html=True)
else:
    gallery_ready_placeholder.markdown('<span class="badge badge-warn">Gallery cache: Missing</span>', unsafe_allow_html=True)
    if 'cache_path' in locals() and cache_path and os.path.exists(str(cache_path)) and GALLERY_LOAD_ERROR:
        st.sidebar.error(GALLERY_LOAD_ERROR)


def recognize_faces_in_img(img, gallery_data, topn_show=1, required_size=(160, 160), threshold=0.5):
    draw = ImageDraw.Draw(img)
    model, detector = get_model_detector()
    results = detector.detect_faces(np.asarray(img))
    faces = []
    for result in results:
        x1, y1, w, h = result['box']
        x1, y1 = abs(x1), abs(y1)
        x2, y2 = x1 + w, y1 + h
        face_pixels = np.asarray(img)[y1:y2, x1:x2]
        if face_pixels.size == 0:
            continue
        face_img = Image.fromarray(face_pixels).resize(required_size)
        face_arr = np.asarray(face_img)
        emb = get_embedding(model, face_arr)
        similarities = np.array([
            compare_embeddings(emb, gal['embedding']) for gal in gallery_data
        ])
        sort_idx = np.argsort(similarities)[::-1]
        best_idx = sort_idx[0]
        best_sim = similarities[best_idx]
        best_name = gallery_data[best_idx]['person'] if best_sim > threshold else 'Unknown'
        top_matches = [(gallery_data[ix]['person'], float(similarities[ix])) for ix in sort_idx[:topn_show]]
        faces.append({
            'bbox': (x1, y1, x2, y2),
            'name': best_name,
            'score': float(best_sim),
            'top_matches': top_matches
        })
        draw.rectangle([(x1, y1), (x2, y2)], outline='#ef4444', width=3)
        label = f"{best_name} ({best_sim:.2f})"
        draw.text((x1, max(0, y1 - 14)), label, fill='#ef4444')
    return img, faces

# ---------- Face Verification (Original) ----------

def run_verification():
    st.subheader("Face Verification")
    st.markdown("Compare two face images: Are they of the same person?")
    with st.container():
        cols = st.columns(2)
        with cols[0]:
            img1 = st.file_uploader("Upload Image 1", type=["jpg", "jpeg", "png"], key="img1")
        with cols[1]:
            img2 = st.file_uploader("Upload Image 2", type=["jpg", "jpeg", "png"], key="img2")
    run_btn = st.button("Compare Faces", type="primary")
    if run_btn and img1 and img2:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tf1:
            tf1.write(img1.read())
            path1 = tf1.name
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tf2:
            tf2.write(img2.read())
            path2 = tf2.name
        face1 = extract_face(path1)
        face2 = extract_face(path2)
        cols = st.columns(2)
        with cols[0]:
            st.caption("Detected Face 1")
            st.image(face1 if face1 is not None else np.zeros((160,160,3), dtype=np.uint8), width=180)
            if face1 is None:
                st.warning("No face detected in Image 1.")
        with cols[1]:
            st.caption("Detected Face 2")
            st.image(face2 if face2 is not None else np.zeros((160,160,3), dtype=np.uint8), width=180)
            if face2 is None:
                st.warning("No face detected in Image 2.")
        if face1 is not None and face2 is not None:
            model, _ = get_model_detector()
            emb1 = get_embedding(model, face1)
            emb2 = get_embedding(model, face2)
            score = compare_embeddings(emb1, emb2)
            m1, m2 = st.columns(2)
            with m1:
                st.metric("Cosine Similarity", f"{score:.4f}")
            with m2:
                st.metric("Decision Threshold", "0.50")
            if score > 0.5:
                st.success("Result: Same person ✅")
            else:
                st.error("Result: Different persons ❌")
        os.remove(path1)
        os.remove(path2)

# ---------- Face Recognition (Cache-only in UI) ----------

def run_recognition():
    st.subheader("Face Recognition")
    if _gallery_data is None:
        st.error("Gallery cache not found. Please pre-build or upload a pickle before using recognition.")
        st.code(
            """python facenet_assignment.py build-gallery \
--dataset_dir extracted_archive/lfw_funneled \
--cache_path lfw_gallery_embeddings.pkl"""
        )
        return

    st.markdown("Recognize who appears in an image or webcam snapshot using the selected embeddings gallery.")
    src = st.tabs(["Image Upload", "Webcam Snapshot"])

    with src[0]:
        uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"], key="rec_img")
        if uploaded:
            img = Image.open(uploaded).convert('RGB')
            st.image(img, caption="Original input", width=480)
            with st.spinner("Detecting and recognizing faces..."):
                t0 = time.time()
                out_img, faces = recognize_faces_in_img(img.copy(), _gallery_data, topn_show=topn, threshold=sim_threshold)
                t1 = time.time()
            st.image(out_img, caption="Recognized faces", width=480)
            cols = st.columns(3)
            cols[0].metric("Detected Faces", f"{len(faces)}")
            cols[1].metric("Top-N", f"{topn}")
            cols[2].metric("Time", f"{t1 - t0:.2f}s")
            for i, face in enumerate(faces):
                st.markdown(f"**Face {i+1}:** {face['name']} — similarity: {face['score']:.4f}")
                st.caption("Top matches: " + ", ".join([f"{n} ({s:.2f})" for n, s in face['top_matches']]))
            if len(faces) == 0:
                st.warning("No faces detected.")

    with src[1]:
        cap = st.camera_input("Take a photo")
        if cap:
            img = Image.open(cap).convert('RGB')
            st.image(img, caption="Captured", width=480)
            with st.spinner("Detecting and recognizing faces..."):
                t0 = time.time()
                out_img, faces = recognize_faces_in_img(img.copy(), _gallery_data, topn_show=topn, threshold=sim_threshold)
                t1 = time.time()
            st.image(out_img, caption="Recognized faces", width=480)
            cols = st.columns(3)
            cols[0].metric("Detected Faces", f"{len(faces)}")
            cols[1].metric("Top-N", f"{topn}")
            cols[2].metric("Time", f"{t1 - t0:.2f}s")
            for i, face in enumerate(faces):
                st.markdown(f"**Face {i+1}:** {face['name']} — similarity: {face['score']:.4f}")
                st.caption("Top matches: " + ", ".join([f"{n} ({s:.2f})" for n, s in face['top_matches']]))
            if len(faces) == 0:
                st.warning("No faces detected.")

# ---------- App Mode Selector ----------
if mode == MODE_VERIFICATION:
    run_verification()
else:
    run_recognition()
