
import json
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image, UnidentifiedImageError
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# ---------------- Configuration ----------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "smartsort_model.keras"
CLASS_NAMES_PATH = BASE_DIR / "class_names.json"

CONFIDENCE_THRESHOLD = 0.60
IMAGE_SIZE = (224, 224)

st.set_page_config(
    page_title="SmartSort | Waste Classification",
    page_icon="♻️",
    layout="wide"
)

# ---------------- Load model ----------------
@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    return tf.keras.models.load_model(MODEL_PATH)

@st.cache_data
def load_class_names():
    if not CLASS_NAMES_PATH.exists():
        raise FileNotFoundError(
            f"Class names file not found: {CLASS_NAMES_PATH}"
        )
    with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as f:
        names = json.load(f)

    if not isinstance(names, list) or len(names) != 6:
        raise ValueError("class_names.json must contain six class names.")

    return names

# ---------------- Disposal guidance ----------------
def get_guidance(category, clean, empty):
    if category.lower() == "trash":
        return (
            "This item was classified as general trash. "
            "Check whether it needs a special collection service "
            "before putting it in general waste."
        )

    if category.lower() == "plastic":
        advice = (
            "Plastic recycling depends on the type of plastic and "
            "whether your local service accepts it. Check local rules."
        )
    elif category.lower() == "cardboard":
        advice = (
            "Cardboard is often recyclable when dry and reasonably clean. "
            "Keep it dry and flatten it if your local service accepts it."
        )
    elif category.lower() == "paper":
        advice = (
            "Paper may be recyclable if accepted locally. "
            "Greasy, wet, or heavily contaminated paper may not be accepted."
        )
    elif category.lower() == "metal":
        advice = (
            "Metal cans are often recyclable through accepted collection "
            "services. Check local rules for the specific item."
        )
    elif category.lower() == "glass":
        advice = (
            "Glass acceptance varies by type and location. "
            "Check whether your local service accepts this kind of glass."
        )
    else:
        advice = (
            "Check the item's material and local disposal instructions."
        )

    if not empty:
        advice += " Empty the item if it is safe to do so."
    if not clean:
        advice += (
            " If appropriate, remove leftover contents safely; "
            "do not wash or handle hazardous substances."
        )

    return advice

# ---------------- User interface ----------------
st.title("♻️ SmartSort")
st.subheader("AI-Based Waste Classification and Recycling Guidance")

st.write(
    "Upload a waste image to receive an AI prediction and general "
    "disposal guidance."
)

st.info(
    "This is an educational prototype. Predictions may be incorrect, "
    "and local recycling rules always take priority."
)

try:
    model = load_model()
    class_names = load_class_names()
except Exception as e:
    st.error(f"Could not load the model or class names: {e}")
    st.stop()

uploaded_file = st.file_uploader(
    "Upload a waste image",
    type=["jpg", "jpeg", "png", "webp"]
)

col1, col2 = st.columns(2)

with col1:
    is_empty = st.radio(
        "Is the item empty?",
        ["Yes", "No", "Not sure"],
        horizontal=True
    )

with col2:
    is_clean = st.radio(
        "Is the item reasonably clean?",
        ["Yes", "No", "Not sure"],
        horizontal=True
    )

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded waste image", width="stretch")

        if st.button("Classify Waste", type="primary"):
            with st.spinner("Analyzing your image..."):
                resized = image.resize(IMAGE_SIZE)
                array = np.asarray(resized, dtype=np.float32)
                array = preprocess_input(array)
                batch = np.expand_dims(array, axis=0)

                scores = model.predict(batch, verbose=0)[0]

                if len(scores) != len(class_names):
                    st.error(
                        "Model output and class names do not match."
                    )
                    st.stop()

                index = int(np.argmax(scores))
                category = class_names[index]
                confidence = float(scores[index])

            st.divider()
            st.header("Classification Results")

            result_col, score_col = st.columns(2)

            with result_col:
                st.metric("Predicted Category", category.capitalize())

            with score_col:
                st.metric("Highest Model Score", f"{confidence:.1%}")

            if confidence < CONFIDENCE_THRESHOLD:
                st.warning(
                    "Low-confidence prediction. Please verify the waste "
                    "category manually. The 60% threshold is an initial "
                    "prototype setting, not a validated universal rule."
                )

            if is_empty == "Not sure" or is_clean == "Not sure":
                st.info(
                    "Since the item's condition is uncertain, inspect it "
                    "carefully before deciding how to dispose of it."
                )

            st.subheader("Disposal Guidance")

            if category.lower() == "trash":
                st.write(get_guidance(category, False, False))
            else:
                st.write(
                    get_guidance(
                        category,
                        is_clean == "Yes",
                        is_empty == "Yes"
                    )
                )

            st.caption(
                "The model classifies the image; it does not verify "
                "contamination or guarantee recyclability. Batteries, "
                "electronics, chemicals, medical waste, and other special "
                "waste require appropriate local collection guidance."
            )

            with st.expander("View scores for all six categories"):
                for name, score in zip(class_names, scores):
                    st.write(f"**{name.capitalize()}** — {score:.1%}")
                    st.progress(float(np.clip(score, 0.0, 1.0)))

    except (UnidentifiedImageError, OSError, ValueError) as e:
        st.error(f"Unable to process this image: {e}")
    except Exception as e:
        st.error(f"Prediction failed: {e}")

st.divider()
st.caption(
    "SmartSort student project | Always check your local recycling rules."
)
