import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification, BlipProcessor, BlipForConditionalGeneration
import torch
import re
from PIL import Image

st.set_page_config(page_title="Toxic Content Moderation", layout="centered")

# 🔴 Inject custom red-accented CSS
st.markdown("""
    <style>
    .stApp {
        background-color: #0f1117;
        color: #e0e0e0;
        font-family: 'Segoe UI', sans-serif;
    }

    .css-10trblm {
        font-size: 2.5em !important;
        font-weight: bold;
        color: #ef5350;  /* Red title */
        text-align: center;
        padding-bottom: 0.5em;
    }

    .css-1v0mbdj p {
        font-size: 1.2em !important;
        color: #b0bec5;
    }

    .stSelectbox, .stMultiSelect, .stTextInput, .stFileUploader {
        background-color: #1e222b !important;
        border-radius: 10px;
        padding: 10px;
        color: white;
        border: 1px solid #e53935 !important;  /* Red border */
    }

    .result-card {
        background-color: #3b1e22;
        padding: 15px;
        border-radius: 12px;
        border-left: 5px solid #ef5350;
        margin-top: 20px;
        font-size: 1.1em;
        color: white;
    }

    .caption-box {
        color: #ff8a80;
        font-style: italic;
        font-size: 1.1em;
        margin-top: 10px;
    }

    .stButton>button {
        background-color: #e53935 !important;
        color: white !important;
        border: none;
        border-radius: 8px;
        padding: 0.5em 1em;
        
        
    }

    .stButton>button:hover {
        background-color: #ff8a80 !important;
        color: #1e1e1e !important;
    }

    /* Red tags */
    div[data-baseweb="tag"] {
        background-color: #c62828 !important;
        color: white !important;
        border-radius: 8px !important;
        padding: 5px 10px !important;
        font-weight: 500 !important;
        margin: 3px 5px 3px 0 !important;
        transition: background-color 0.3s ease;
    }

    div[data-baseweb="tag"] svg {
        stroke: white !important;
    }

    div[data-baseweb="tag"]:hover {
        background-color: #e53935 !important;
    }

    div[data-baseweb="select"] > div {
        background-color: #1e1e2f !important;
        border: 1px solid #c62828 !important;
        border-radius: 10px !important;
        color: white !important;
    }

    li[role="option"]:hover {
        background-color: #c62828 !important;
        color: white !important;
    }
    </style>
""", unsafe_allow_html=True)

# 🔴 App title and goal
st.markdown("<h1 style='text-align: left; color: #ef5350;'>🛡️ Safe and Responsible Toxic Content Moderation</h1>", unsafe_allow_html=True)
st.write("🎯 **Goal:** A dual-stage, multi-modal AI pipeline for responsible content classification.")

# 🧼 Text cleaner
def dataCleaning(text):
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"[^a-z0-9\s.,!?']+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.lower()

# 🔎 Prediction function
def predict_moderation(input_text):
    input_text = dataCleaning(input_text)
    class_labels = [
        "Safe", "Violent Crimes", "Elections", "Sex-Related Crimes", "Unsafe",
        "Non-Violent Crimes", "Child Sexual Exploitation", "Unknown S-Type", "Suicide & Self-Harm"
    ]
    tokenizer = AutoTokenizer.from_pretrained("my_distilbert_model")
    model = AutoModelForSequenceClassification.from_pretrained("my_distilbert_model", num_labels=len(class_labels))
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, padding=True)
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        predicted_class = torch.argmax(outputs.logits, dim=1).item()
    return class_labels[predicted_class - 1]

# 🧠 BLIP captioning
@st.cache_resource
def load_blip():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return processor, model

# 🎛️ Input selection
st.markdown('<div style="color:white; font-size:16px;">💻 Select input type:</div>', unsafe_allow_html=True)
choice = st.multiselect(label="", options=["image", "query"])

# 🖼️ Image input
if "image" in choice:
    st.markdown('<div style="color:white; font-size:16px;">🖼️ Upload an image:</div>', unsafe_allow_html=True)
    image = st.file_uploader("", type=["jpg", "jpeg", "png"])
    if image:
        st.image(image, caption="📷 Your uploaded image", use_column_width=True)
        processor, blip_model = load_blip()
        img = Image.open(image).convert("RGB")

        with st.spinner("🧠 Generating caption..."):
            inputs = processor(images=img, return_tensors="pt")
            output = blip_model.generate(**inputs)
            caption = processor.decode(output[0], skip_special_tokens=True)

        st.markdown(f"<div class='caption-box'>📝 <strong>Generated Caption:</strong> {caption}</div>", unsafe_allow_html=True)
        prediction = predict_moderation(caption)
        st.markdown(f"<div class='result-card'>🚨 <strong>Predicted Class:</strong> {prediction}</div>", unsafe_allow_html=True)

# 💬 Query input
if "query" in choice:
    st.markdown('<div style="color:white; font-size:16px;">✍️ Enter your query:</div>', unsafe_allow_html=True)
    query = st.text_input("")
    if query:
        st.markdown(f"<div class='caption-box'>🔍 <strong>Your Input:</strong> {query}</div>", unsafe_allow_html=True)
        prediction = predict_moderation(query)
        st.markdown(f"<div class='result-card'>🚨 <strong>Predicted Class:</strong> {prediction}</div>", unsafe_allow_html=True)
