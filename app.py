import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ----------------------------
# Page Config (UI only)
# ----------------------------
st.set_page_config(
    page_title="Next Word Predictor",
    page_icon="🔮",
    layout="centered"
)

# ----------------------------
# Load Resources (UNCHANGED LOGIC)
# ----------------------------
@st.cache_resource
def load_lstm_model():
    return load_model("next_word_lstm.h5", compile=False)

@st.cache_resource
def load_tokenizer():
    with open("tokenizer.pickle", "rb") as handle:
        return pickle.load(handle)

model = load_lstm_model()
tokenizer = load_tokenizer()

def predict_next_word(model, tokenizer, text, max_sequence_len):
    token_list = tokenizer.texts_to_sequences([text])[0]

    if len(token_list) == 0:
        return "❓"

    token_list = token_list[-(max_sequence_len - 1):]
    token_list = pad_sequences(
        [token_list], maxlen=max_sequence_len - 1, padding="pre"
    )

    predicted = model.predict(token_list, verbose=0)
    predicted_index = np.argmax(predicted, axis=1)[0]

    return tokenizer.index_word.get(predicted_index, "❓")

# ----------------------------
# UI HEADER
# ----------------------------
st.markdown(
    """
    <h1 style='text-align: center;'>🔮 Next Word Prediction</h1>
    <p style='text-align: center; color: gray;'>
        Powered by LSTM • Deep Learning • NLP
    </p>
    """,
    unsafe_allow_html=True
)

st.write("")  # spacing

# ----------------------------
# Main Card
# ----------------------------
with st.container():
    # st.markdown(
    #     """
    #     <div style="
    #         background-color: #f9f9f9;
    #         padding: 25px;
    #         border-radius: 12px;
    #         box-shadow: 0 4px 12px rgba(0,0,0,0.08);
    #     ">
    #     """,
    #     unsafe_allow_html=True
    # )

    st.subheader("✍️ Enter your text")
    input_text = st.text_input(
        "",
        "To be or not to",
        label_visibility="collapsed"
    )

    st.write("")

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        predict_btn = st.button("✨ Predict Next Word", use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

# ----------------------------
# Prediction Output
# ----------------------------
if predict_btn:
    with st.spinner("🔍 Thinking..."):
        max_sequence_len = model.input_shape[1] + 1
        next_word = predict_next_word(
            model, tokenizer, input_text, max_sequence_len
        )

    st.write("")
    st.markdown(
        f"""
        <div style="
            background-color: #e8f5e9;
            padding: 20px;
            border-radius: 12px;
            text-align: center;
            font-size: 22px;
            color: gray;
        ">
            🧠 <b>Predicted Next Word:</b><br>
            <span style="font-size: 28px; color: #2e7d32;">
                {next_word}
            </span>
        </div>
        """,
        unsafe_allow_html=True
    )

# ----------------------------
# Footer
# ----------------------------
st.write("")
st.markdown(
    """
    <p style='text-align: center; color: gray; font-size: 14px;'>
        Built with ❤️ using TensorFlow & Streamlit
    </p>
    """,
    unsafe_allow_html=True
)
