# FaceNet Face Recognition Assignment

## Overview

This project demonstrates face recognition using a pre-trained FaceNet model. You will:
- Detect faces in images using MTCNN
- Extract facial embeddings using FaceNet
- Compare embeddings to determine if two images belong to the same person

---

## Environment Setup

### Using Conda (Recommended)
1. **Create environment:**
    ```bash
    conda create -n facenet_env python=3.11 numpy matplotlib pillow tensorflow keras scikit-learn opencv
    ```
2. **Activate environment:**
    ```bash
    conda activate facenet_env
    ```
3. **Install additional packages via pip:**
    ```bash
    pip install keras-facenet opencv-python mtcnn
    ```

### Using requirements.txt with pip (Alternative)
1. (Optional) Create a Python virtual environment:
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```
2. **Install requirements:**
    ```bash
    pip install -r requirements.txt
    ```

---

## Downloading Sample Images

You will need publicly available face images to test the model.
- [Labeled Faces in the Wild (LFW) dataset](http://vis-www.cs.umass.edu/lfw/)
    - Download a few images to a local `images/` folder for testing.

---

## Prebuilding Gallery Embeddings (Required for Recognition Mode)

Recognition mode uses a precomputed gallery of FaceNet embeddings. Build it once from the LFW dataset (or your own dataset) and the UI will load it.

Assuming you extracted LFW to `extracted_archive/lfw_funneled/`:

```bash
python facenet_assignment.py build-gallery \
  --dataset_dir extracted_archive/lfw_funneled \
  --cache_path lfw_gallery_embeddings.pkl
```

- This creates `lfw_gallery_embeddings.pkl` in the project root.
- The Streamlit UI will not auto-build; it requires this file to exist.

(Optional) Limit images per identity for faster builds:
```bash
python facenet_assignment.py build-gallery \
  --dataset_dir extracted_archive/lfw_funneled \
  --cache_path lfw_gallery_embeddings.pkl \
  --limit_per_person 5
```

---

## Running the Examples

### Command-Line Interface (CLI)

Compare two images (verification):
```bash
python facenet_assignment.py verify --img1 images/person1.jpg --img2 images/person2.jpg
```
**Output:**
- Prints cosine similarity score
- Displays if images are of the "same person" or "different persons"

### Streamlit Web App (Recommended UI)

Start the UI:
```bash
streamlit run streamlit_app.py
```
- Verification works out of the box (compare two uploads).
- Recognition requires the gallery pickle (`lfw_gallery_embeddings.pkl`). If missing, the UI shows a command to build it.

---

## Files Included
| File                  | Purpose                                                      |
|-----------------------|-------------------------------------------------------------|
| README.md             | Project documentation, setup, usage, and references         |
| requirements.txt      | Python dependencies for pip users                           |
| facenet_assignment.py | Main logic, CLI (verify/build-gallery)                      |
| streamlit_app.py      | Streamlit UI for verification and recognition                |
| (screenshots/)        | Optional: result screenshots for grading/report             |

---

## Further Feature Ideas (Bonus or Exploration)
- **Face clustering:** Group multiple images by identity using embeddings.
- **Face search:** Find the most similar face from a database given a query image.
- **Batch processing:** Compare all pairs in a folder or compute similarity matrix.
- **Threshold tuning:** Interactive threshold slider to visualize how matches vary.
- **Heatmap visualization:** Show similarity matrix for many faces.
- **Export embeddings:** Save/load embedding vectors for later fast inference.

---

## References
* [FaceNet Paper](https://arxiv.org/abs/1503.03832)
* [keras-facenet](https://github.com/nyoki-mtl/keras-facenet)
* [MTCNN](https://github.com/ipazc/mtcnn)

---

