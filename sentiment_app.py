import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
from tensorflow.keras.datasets import imdb
import zipfile
import json
import io
import tempfile
import os

# --- Page Config ---
st.set_page_config(
    page_title="Movie Review Sentiment Analysis",
    layout="centered",
    page_icon="🎬",
)

# --- Model Options ---
MODEL_OPTIONS = {
    "GRU Model": "gru_model.keras",
    "LSTM Model": "lstm_model.keras",
    "Simple RNN Model": "simple_rnn_model.keras",
}

# --- Sidebar: Model Selection ---
st.sidebar.title("⚙️ Settings")
selected_model_name = st.sidebar.selectbox(
    "Choose a Model:",
    list(MODEL_OPTIONS.keys()),
    index=0,
    help="Select the deep learning model you want to use for sentiment prediction.",
)
selected_model_file = MODEL_OPTIONS[selected_model_name]

st.sidebar.markdown("---")
st.sidebar.info(
    "**Models Available:**\n"
    "- 🔵 GRU Model\n"
    "- 🟢 LSTM Model\n"
    "- 🟠 Simple RNN Model"
)

# ---------------------------------------------------------------------------
# Helper: strip fields that newer Keras saves but older Keras doesn't know
# ---------------------------------------------------------------------------
_STRIP_KEYS = {"quantization_config"}

def _strip_unknown_fields(obj):
    """Recursively remove keys in _STRIP_KEYS from every dict in the config."""
    if isinstance(obj, dict):
        # Replace DTypePolicy objects with plain dtype string
        if obj.get("class_name") == "DTypePolicy" and "config" in obj:
            return obj["config"].get("name", "float32")
        cleaned = {}
        for k, v in obj.items():
            if k in _STRIP_KEYS:
                continue
            cleaned[k] = _strip_unknown_fields(v)
        return cleaned
    if isinstance(obj, list):
        return [_strip_unknown_fields(item) for item in obj]
    return obj


@st.cache_resource(show_spinner=False)
def load_model_patched(model_path: str):
    """
    Load a .keras file saved with a newer Keras version on an older runtime.

    Strategy:
      1. Open the .keras ZIP in memory.
      2. Read config.json and strip fields the local Keras doesn't understand.
      3. Write a patched .keras file to a temp location.
      4. Load with tf.keras.models.load_model(compile=False).
    """
    with zipfile.ZipFile(model_path, "r") as zin:
        names = zin.namelist()

        # Build patched in-memory ZIP
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zout:
            for name in names:
                data = zin.read(name)
                if name == "config.json":
                    cfg = json.loads(data.decode("utf-8"))
                    cfg = _strip_unknown_fields(cfg)
                    data = json.dumps(cfg).encode("utf-8")
                zout.writestr(name, data)

    # Write to a temp file (tf.keras needs a real path)
    buf.seek(0)
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".keras")
    try:
        with os.fdopen(tmp_fd, "wb") as f:
            f.write(buf.read())
        model = tf.keras.models.load_model(tmp_path, compile=False)
    finally:
        os.remove(tmp_path)

    return model


# --- Load selected model ---
with st.spinner(f"Loading **{selected_model_name}**..."):
    try:
        model = load_model_patched(selected_model_file)
    except Exception as e:
        st.error(f"❌ Error loading model `{selected_model_file}`: {e}")
        st.stop()

# --- Recreate word_index from IMDb dataset ---
@st.cache_resource(show_spinner=False)
def get_word_index():
    word_index = imdb.get_word_index()
    word_index = {k: (v + 3) for k, v in word_index.items()}
    word_index["<PAD>"] = 0
    word_index["<START>"] = 1
    word_index["<UNK>"] = 2
    word_index["<UNUSED>"] = 3
    return word_index

with st.spinner("Preparing vocabulary..."):
    word_index = get_word_index()

MAX_LEN = 256

# --- Preprocess Function ---
def preprocess_review(text, word_index, max_len):
    words = text.lower().split()
    encoded_review = [word_index.get(word, word_index["<UNK>"]) for word in words]
    encoded_review = [word_index["<START>"]] + encoded_review
    padded_review = pad_sequences(
        [encoded_review], value=word_index["<PAD>"], padding="post", maxlen=max_len
    )
    return padded_review

# --- Main UI ---
st.title("🎬 Movie Review Sentiment Analyzer")
st.markdown(
    f"""
    This app uses a trained **{selected_model_name}** to predict the sentiment
    (**Positive** or **Negative**) of movie reviews.  
    Type your review below and click **Analyze** to get an instant prediction!
    """
)

user_input = st.text_area(
    "✍️ Enter your movie review here:",
    height=200,
    placeholder="e.g., 'This movie was absolutely fantastic! I loved every minute of it.'",
)

if st.button("✨ Analyze Sentiment", use_container_width=True):
    if user_input.strip():
        with st.spinner("Processing your review and predicting sentiment..."):
            processed_input = preprocess_review(user_input, word_index, MAX_LEN)
            prediction = model.predict(processed_input)[0][0]

        st.markdown("---")
        if prediction >= 0.5:
            sentiment = "Positive 😊"
            confidence = float(prediction)
            st.success(f"### Prediction: **{sentiment}**\n**Confidence:** {confidence:.2%}")
            st.balloons()
        else:
            sentiment = "Negative 😔"
            confidence = float(1 - prediction)
            st.error(f"### Prediction: **{sentiment}**\n**Confidence:** {confidence:.2%}")

        st.markdown(f"_Raw model score: `{prediction:.4f}` | Model used: `{selected_model_name}`_")
    else:
        st.warning("💡 Please enter some text into the box to analyze.")

st.markdown("---")
st.caption("Built with TensorFlow/Keras and Streamlit · IMDB Sentiment Analysis")
