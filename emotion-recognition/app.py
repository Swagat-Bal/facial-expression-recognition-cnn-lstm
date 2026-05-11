"""
Streamlit UI for static-image facial expression prediction.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import streamlit as st
from PIL import Image

from predict import DEFAULT_MODEL_PATH, load_expression_model, predict_pil_image
from preprocess import display_labels


def inject_dark_theme() -> None:
    st.markdown(
        """
        <style>
        .stApp {
            background: #101318;
            color: #f5f7fb;
        }
        [data-testid="stSidebar"] {
            background: #161b22;
        }
        .result-panel {
            border: 1px solid #2b3340;
            border-radius: 8px;
            padding: 16px;
            background: #151a21;
        }
        .prediction-label {
            font-size: 1.5rem;
            font-weight: 800;
            margin-bottom: 4px;
        }
        .prob-row {
            display: grid;
            grid-template-columns: 86px 1fr 58px;
            align-items: center;
            gap: 10px;
            margin: 9px 0;
            font-size: 0.92rem;
        }
        .prob-track {
            height: 11px;
            background: #2a313d;
            border-radius: 999px;
            overflow: hidden;
        }
        .prob-fill {
            height: 11px;
            background: linear-gradient(90deg, #2dd4bf, #38bdf8);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource(show_spinner=False)
def get_model(model_path: str):
    return load_expression_model(Path(model_path))


def render_probability_bars(probabilities: Dict[str, float]) -> None:
    rows = []
    for label in display_labels():
        value = probabilities.get(label, 0.0)
        rows.append(
            f"""
            <div class="prob-row">
                <span>{label}</span>
                <div class="prob-track"><div class="prob-fill" style="width:{value * 100:.1f}%"></div></div>
                <span>{value * 100:.1f}%</span>
            </div>
            """
        )
    st.markdown("".join(rows), unsafe_allow_html=True)


def main() -> None:
    st.set_page_config(
        page_title="Facial Expression Recognition",
        page_icon=":material/sentiment_satisfied:",
        layout="wide",
    )
    inject_dark_theme()

    st.title("Facial Expression Recognition")
    st.caption("CNN spatial feature extraction + LSTM sequence classifier for dataset images")

    with st.sidebar:
        st.header("Model Settings")
        model_path = st.text_input("Model path", str(DEFAULT_MODEL_PATH))
        image_size = st.selectbox("Image size", options=[48, 64, 96], index=0)
        model_exists = Path(model_path).exists()
        st.status(
            "Model ready" if model_exists else "Train the model before prediction",
            state="complete" if model_exists else "error",
        )

    left, right = st.columns([1.1, 1])

    with left:
        st.subheader("Input Image")
        uploaded_file = st.file_uploader(
            "Upload a face image",
            type=["jpg", "jpeg", "png", "bmp", "webp"],
            accept_multiple_files=False,
        )
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption=uploaded_file.name, use_container_width=True)
        else:
            image = None
            st.info("Upload a cropped face image from the dataset or any similar image.")

    with right:
        st.subheader("Prediction")
        if image is None:
            st.markdown('<div class="result-panel">Waiting for an image.</div>', unsafe_allow_html=True)
        elif not model_exists:
            st.error("Model file was not found. Run `python train.py` first.")
        else:
            with st.spinner("Predicting expression..."):
                model = get_model(model_path)
                result = predict_pil_image(
                    model,
                    image,
                    image_size=image_size,
                )

            st.markdown('<div class="result-panel">', unsafe_allow_html=True)
            st.markdown(f'<div class="prediction-label">{result.label}</div>', unsafe_allow_html=True)
            st.metric("Confidence", f"{result.confidence * 100:.1f}%")
            render_probability_bars(result.probabilities)
            st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
