
# 🕵️ Crime Scene Investigator AI

> Fine-tuning Visual-Language Models (VLM) for Crime Scene Investigation 🚨  
> Developed by **Muhammad Sufyan Malik**

---

## 📖 Overview

**CID AI** is an advanced Crime Scene Investigation system powered by deep learning and large language models (LLMs).  
It combines **fine-tuned vision models** and **Gemini LLM** to **analyze crime scene images** and **generate detailed investigation reports**.

🚔 Workflow:  
- Fine-tuned **BLIP** model creates **detailed descriptions** of images.
- Descriptions are passed to **Gemini 1.5 Pro** for **reasoning, evidence analysis, and report generation**.

> 💡 Think of it as an AI-powered **Sherlock Holmes** — analyzing images & generating professional crime scene investigations!

---

## 🔥 Live Demo

> [🔗 Streamlit App](https://vlm-fine-tuning-and-downstream-application-development-mg9s2ff.streamlit.app/)

Upload a crime scene image and get an investigation report instantly!

---

## 📦 Model Files

| Component | Link |
|:----------|:-----|
| Fine-tuned BLIP Model | [HuggingFace: `sufyanbinimran/blip-finetuned`](https://huggingface.co/sufyanbinimran/blip-finetuned/commit/85ef4f8a8816b6bd4a773af421853ca5b1bcacaf) |
| Codebase | [GitHub Repository](https://github.com/sufyanbinimran/VLM-fine-tuning-and-downstream-application-development.git) |

---

## 🛠️ How It Works

1. **Upload an Image** — Any scene, object, or suspected crime scene.
2. **BLIP Fine-tuned Model** generates an **image caption** (description).
3. **Gemini LLM** analyzes the scene description using an investigation prompt.
4. **Investigation Report** is generated detailing:
   - Possible crime evidence
   - Suspects (if any)
   - Logical reasoning steps

---

## 📚 Dataset Used for Fine-Tuning

We fine-tuned BLIP on a **filtered dataset** combining:

- **Flickr8K** — 8,000+ images, 5 captions per image  
  > [Kaggle Flickr8k Dataset](https://www.kaggle.com/datasets/adityajn105/flickr8k)

- **COCO-100K (Subset)** — 10,000+ image-caption pairs  
  > [COCO Dataset Official Site](https://cocodataset.org/#download)

**Purpose of Fine-tuning:**  
- Focus BLIP more on **scene understanding** rather than just objects.
- Prepare captions that are **suitable for forensic analysis**.

---

## 🚀 Local Installation (Optional)

To run it locally:

```bash
# Clone the repo
git clone https://github.com/sufyanbinimran/VLM-fine-tuning-and-downstream-application-development.git
cd VLM-fine-tuning-and-downstream-application-development

# Install dependencies
pip install -r requirements.txt

# Run Streamlit app
streamlit run app.py
```

**Dependencies:**
- `transformers`
- `huggingface_hub`
- `torch`
- `streamlit`
- `Pillow`
- `google-generativeai`

Make sure to set your **Gemini API key** inside `app.py`.

---

## 🖼️ Sample Outputs

| Uploaded Crime Scene Image | Investigation Report |
|:----------------------------|:---------------------|
| ![sample1](https://github.com/sufyanbinimran/VLM-fine-tuning-and-downstream-application-development/blob/main/Screenshot%202025-05-04%20223515.png?raw=true) | **Image Description:** A blood-stained knife lying near a broken window. <br> **LLM Report:** Suspected break-in and assault. Fingerprints and DNA evidence collection recommended. |
| ![sample2](https://github.com/sufyanbinimran/VLM-fine-tuning-and-downstream-application-development/blob/main/Screenshot%202025-05-04%20223556.png?raw=true) | **Image Description:** Scattered jewelry and an open safe. <br> **LLM Report:** Suspected burglary. Investigate entry points and interview possible witnesses. |

(*You can replace placeholders with real sample images once you upload!*)

---

## 🎯 Features

- Fine-tuned Vision-Language Model for **crime understanding**
- Powerful **reasoning** using Gemini LLM
- Easy-to-use **Streamlit Interface**
- **Upload Image** + **Ask Investigation Question**
- **Detailed Crime Scene Reports**

---

## ✨ Future Enhancements

- Add **object detection** to highlight critical evidence.
- Generate **multiple hypotheses** for complex scenes.
- Support **multi-image crime scenes**.
- Integrate with **police report templates**.

---


---

## 📬 Contact

> **Muhammad Sufyan Malik**  
> 📧 [GitHub Profile](https://github.com/sufyanbinimran)

---

# 🕵️ Crime Scene Investigator AI: Solving Crimes with Vision and Language Intelligence!
