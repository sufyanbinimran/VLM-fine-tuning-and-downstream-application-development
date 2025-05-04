# app.py

import streamlit as st
from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
import google.generativeai as genai

# ====================== CONFIG ==========================
BLIP_REPO = "sufyanbinimran/blip-finetuned"
GEMINI_API_KEY = "AIzaSyA4mHi9wTVd4ruEEAGT5mwchfbqi6NTSII"

# ====================== CUSTOM CSS ==========================
crime_theme = """
<style>
/* Dark crime theme background and text */
body, .stApp {
    background-color: #111111;
    color: #f1f1f1;
}

/* Yellow tape header style */
h1 {
    background: repeating-linear-gradient(
        -45deg,
        #FFD700,
        #FFD700 10px,
        #111 10px,
        #111 20px
    );
    color: black;
    padding: 12px;
    border-radius: 6px;
    text-align: center;
    font-family: monospace;
}

/* Stylish button */
button[kind="primary"] {
    background-color: #FFD700 !important;
    color: black !important;
    border-radius: 8px !important;
    font-weight: bold;
}

/* Image border effect */
img {
    border: 4px solid #FFD700;
    border-radius: 8px;
    margin-top: 10px;
}

/* Subheaders red tint */
h2, h3 {
    color: #FF4136;
    font-family: monospace;
}

/* Footer */
footer {
    color: gray;
}

/* Inputs dark mode */
input, textarea {
    background-color: #333 !important;
    color: #FFD700 !important;
}
</style>
"""
st.markdown(crime_theme, unsafe_allow_html=True)

# ====================== SETUP ==========================
@st.cache_resource
def load_blip_model():
    processor = BlipProcessor.from_pretrained(BLIP_REPO)
    model = BlipForConditionalGeneration.from_pretrained(BLIP_REPO)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return processor, model.to(device), device

processor, blip_model, device = load_blip_model()

genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel('models/gemini-1.5-pro-latest')

# ====================== APP UI ==========================
st.set_page_config(page_title="🕵️ Crime Scene Investigator AI", layout="centered")

st.title("🚨 CRIME SCENE INVESTIGATION AI 🚨")
st.markdown("Upload a **crime scene image** and ask a question about it. The AI forensic system will analyze and generate a detailed **investigation report**.")

uploaded_image = st.file_uploader("📷 Upload a crime scene image", type=["jpg", "jpeg", "png"])
user_question = st.text_input("❓ Enter your investigation question (optional)")

if uploaded_image:
    image = Image.open(uploaded_image).convert("RGB")
    st.image(image, caption="🖼️ Uploaded Scene Evidence", use_column_width=True)

    if st.button("🔍 Analyze Crime Scene"):
        with st.spinner("🕵️ Analyzing evidence and compiling report..."):

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
        st.subheader("📋 Scene Description")
        st.markdown(f"<div style='background-color:#222;padding:10px;border-radius:5px;color:#FFD700'>{caption}</div>", unsafe_allow_html=True)

        st.subheader("🧩 AI Investigation Report")
        st.markdown(f"<div style='background-color:#222;padding:10px;border-radius:5px;color:#FF4136'>{investigation_solution}</div>", unsafe_allow_html=True)

# ====================== FOOTER ==========================
st.markdown("---")
st.caption("🔗 Built by Muhammad Sufyan Malik — Powered by Hugging Face & Gemini Pro")
