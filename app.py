# app.py

import streamlit as st

# ====================== MUST BE FIRST ==========================
st.set_page_config(
    page_title="🕵️ Crime Scene Investigator AI",
    layout="centered",
    page_icon="🕵️",
)

# ====================== REST IMPORTS ==========================
from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
import google.generativeai as genai

# ====================== CONFIG ==========================
# Load BLIP fine-tuned model from Hugging Face repo
BLIP_REPO = "sufyanbinimran/blip-finetuned"

# Configure Gemini API Key (replace with your key)
GEMINI_API_KEY = "AIzaSyA4mHi9wTVd4ruEEAGT5mwchfbqi6NTSII"

# ====================== CUSTOM STYLES ==========================
custom_css = """
<style>
/* Dark crime background */
body, .stApp {
    background-color: #111111;
    color: #f0f0f0;
}

/* Title styling - like crime scene tape */
h1 {
    background: repeating-linear-gradient(
        45deg,
        #FFD700,
        #FFD700 10px,
        black 10px,
        black 20px
    );
    color: black;
    padding: 0.5rem;
    text-align: center;
    border-radius: 10px;
}

/* Section headings */
h3, .st-subheader {
    color: #FFD700;
}

/* Buttons */
.stButton>button {
    background-color: #FFD700;
    color: black;
    border-radius: 8px;
    padding: 0.5rem 1rem;
    font-weight: bold;
}
.stButton>button:hover {
    background-color: black;
    color: #FFD700;
    border: 2px solid #FFD700;
}

/* Input fields */
input, textarea {
    background-color: #333;
    color: white;
    border: 1px solid #FFD700;
}

/* File uploader */
.css-1um0q1k {
    border: 2px dashed #FFD700;
    color: #FFD700;
}

/* Card-style output */
.report-card {
    border: 2px solid #FFD700;
    padding: 1rem;
    border-radius: 12px;
    background-color: #222;
    margin-top: 1rem;
}

/* Footer */
footer {
    color: gray;
    text-align: center;
}
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# ====================== SETUP ==========================
@st.cache_resource
def load_blip_model():
    processor = BlipProcessor.from_pretrained(BLIP_REPO)
    model = BlipForConditionalGeneration.from_pretrained(BLIP_REPO)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return processor, model.to(device), device

processor, blip_model, device = load_blip_model()

# Configure Gemini
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel('models/gemini-1.5-pro-latest')

# ====================== APP UI ==========================
st.title("🕵️‍♂️ Crime Scene Investigator AI")

st.markdown("Upload a **crime scene image** and ask a question about it. The AI will analyze and provide an investigation report.")

uploaded_image = st.file_uploader("📷 Upload an image", type=["jpg", "jpeg", "png"])
user_question = st.text_input("❓ Enter your investigation question (optional)")

if uploaded_image:
    image = Image.open(uploaded_image).convert("RGB")
    st.image(image, caption="🖼️ Uploaded Crime Scene", use_column_width=True)

    if st.button("🔍 Analyze Crime Scene"):
        with st.spinner("🕵️‍♂️ Analyzing image and generating report..."):

            # ======== Image Captioning ========
            inputs = processor(images=image, return_tensors="pt").to(device)
            out = blip_model.generate(**inputs, max_length=64)
            caption = processor.decode(out[0], skip_special_tokens=True)

            # ======== Gemini Reasoning ========
            investigation_prompt = (
                f"You are a professional crime scene investigator. Carefully analyze the scene description "
                f"to determine if any crime took place, identify possible suspects, evidence, and explain "
                f"your reasoning with clear details.\n\n"
                f"Scene Description: {caption}\n"
            )

            if user_question.strip():
                investigation_prompt += f"Investigation Question: {user_question.strip()}\n"

            investigation_prompt += "\nYour detailed investigation report:"

            response = gemini_model.generate_content(investigation_prompt)
            investigation_solution = response.text

        # ======== OUTPUT ========
        st.subheader("📝 Investigation Results")
        st.markdown(f"<div class='report-card'><strong>🖼️ Image Description:</strong><br>{caption}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='report-card'><strong>🔍 Investigation Report:</strong><br>{investigation_solution}</div>", unsafe_allow_html=True)

# ====================== FOOTER ==========================
st.markdown("---")
st.caption("🔗 Built by Muhammad Sufyan Malik — Powered by Hugging Face & Gemini Pro")
