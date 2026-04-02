import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from langchain_groq import ChatGroq
from deep_translator import GoogleTranslator
from gtts import gTTS
import os

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(page_title="RetinaScope AI Pro", page_icon="🔬", layout="wide")

# --- 2. ADVANCED PROFESSIONAL CSS ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    :root {
        --bg-main: #0a0f1a;
        --card-bg: rgba(23, 32, 53, 0.8);
        --accent-blue: #3b82f6;
        --accent-emerald: #10b981;
        --accent-red: #ef4444;
        --text-muted: #94a3b8;
    }
    .stApp {
        background-color: var(--bg-main);
        background-image: radial-gradient(at 0% 0%, rgba(59, 130, 246, 0.15) 0, transparent 50%);
        color: #f8fafc;
        font-family: 'Inter', sans-serif;
    }
    .med-card {
        background: var(--card-bg);
        backdrop-filter: blur(16px);
        border-radius: 16px;
        border: 1px solid rgba(255, 255, 255, 0.08);
        padding: 24px;
        margin-bottom: 20px;
    }
    .section-label {
        color: var(--text-muted);
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin-bottom: 12px;
    }
    .recommendation-box { padding: 16px; border-radius: 8px; margin-top: 15px; font-weight: 600; }
    .urgent-alert { background: rgba(239, 68, 68, 0.15); border: 1px solid var(--accent-red); color: #ff8a8a; border-left: 5px solid var(--accent-red); }
    .safe-alert { background: rgba(16, 185, 129, 0.15); border: 1px solid var(--accent-emerald); color: #a7f3d0; border-left: 5px solid var(--accent-emerald); }
    .nav-brand { text-align: center; padding: 20px 0; margin-bottom: 20px; }
    </style>
    """, unsafe_allow_html=True)

# --- 3. CORE LOGIC & MODELS ---
CLASS_NAMES = ['Bulging Eyes', 'Cataracts', 'Crossed Eyes', 'Glaucoma', 'Normal', 'Uveitis']

@st.cache_resource
def load_model():
    model = models.resnet50()
    model.fc = nn.Sequential(nn.Linear(model.fc.in_features, 512), nn.ReLU(), nn.Dropout(0.3), nn.Linear(512, 6))
    model.load_state_dict(torch.load("eye_master_model.pth", map_location='cpu'))
    model.eval()
    return model

model = load_model()

# Translation Helper
def translate_text(text, target_lang_code):
    if target_lang_code == 'en': return text
    return GoogleTranslator(source='auto', target=target_lang_code).translate(text)

# Voice Helper (Short Summary)
def generate_voice_alert(disease_name, urgency_msg, lang_code):
    try:
        # Create a concise sentence for speech
        full_text = f"Analysis suggests {disease_name}. {urgency_msg}"
        translated_voice_text = translate_text(full_text, lang_code)
        
        tts = gTTS(text=translated_voice_text, lang=lang_code, slow=False)
        filename = "alert.mp3"
        tts.save(filename)
        return filename
    except Exception as e:
        return None

# Medical Report Generator (Detailed)
def generate_medical_report(disease, confidence, lang):
    prompt = f"Ophthalmologist Report for '{disease}' ({confidence:.2f}% confidence). Provide Definition, Causes, Symptoms, and Treatment in {lang}."
    # return llm.invoke(prompt).content

# --- 4. APP LAYOUT ---
st.markdown("<div class='nav-brand'><h1><span style='color:#3b82f6;'>RETINA</span>SCOPE <span style='color:#94a3b8; font-weight:300;'>AI PRO</span></h1></div>", unsafe_allow_html=True)

LANG_MAP = {
    "English": "en", "Tamil (தமிழ்)": "ta", "Hindi (हिन्दी)": "hi", 
    "Telugu (తెలుగు)": "te", "Malayalam (മലയാളം)": "ml"
}

with st.sidebar:
    st.markdown("### 🛠️ Global Settings")
    selected_lang = st.selectbox("Select Language", list(LANG_MAP.keys()))
    target_code = LANG_MAP[selected_lang]
    st.markdown("---")
    if st.button("Reset Session"): st.rerun()

# --- INPUT SECTION ---
image_file = None
t1, t2 = st.tabs(["📤 Upload Scan", "📸 Live Capture"])
with t1:
    up = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])
    if up: image_file = up
with t2:
    cw1, cw2, cw3 = st.columns([1, 2, 1])
    with cw2:
        cam = st.camera_input("Scanner")
        if cam: image_file = cam

# --- ANALYSIS ---
if image_file:
    img = Image.open(image_file).convert("RGB")
    tf = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), 
                             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    input_tensor = tf(img).unsqueeze(0)
    
    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.nn.functional.softmax(output[0], dim=0)
        conf, idx = torch.max(probs, 0)
        res_label = CLASS_NAMES[idx]
        confidence_pct = conf.item() * 100

    c1, c2 = st.columns([1.2, 1])
    with c1:
        st.markdown('<div class="med-card"><div class="section-label">📷 Diagnostic Capture</div>', unsafe_allow_html=True)
        st.image(img, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with c2:
        st.markdown('<div class="med-card"><div class="section-label">🩺 Detection Result</div>', unsafe_allow_html=True)
        is_normal = res_label == "Normal"
        status_color = "#10b981" if is_normal else "#ef4444"
        
        # Result and Urgency Strings
        translated_disease = translate_text(res_label, target_code)
        urgency_msg = "Maintain routine checkups." if is_normal else "Immediately consult a doctor."
        
        st.markdown(f"<h1 style='color:{status_color}; margin:0;'>{translated_disease}</h1>", unsafe_allow_html=True)
        
        fig = go.Figure(go.Indicator(mode="gauge+number", value=confidence_pct,
            gauge={'axis': {'range': [None, 100]}, 'bar': {'color': status_color}}))
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', height=180, margin=dict(t=0,b=0,l=10,r=10))
        st.plotly_chart(fig, use_container_width=True)
        
        # Urgency Box + Voice Button
        st.markdown(f"<div class='recommendation-box {'safe-alert' if is_normal else 'urgent-alert'}'>{translate_text(urgency_msg, target_code)}</div>", unsafe_allow_html=True)
        
        # --- SHORT VOICE ALERT BUTTON ---
        if st.button("🔊 Play Voice Alert"):
            with st.spinner("Preparing Audio..."):
                audio_path = generate_voice_alert(res_label, urgency_msg, target_code)
                if audio_path:
                    st.audio(audio_path, format="audio/mp3", autoplay=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Row 3: Multilingual AI Report
    st.markdown('<div class="med-card">', unsafe_allow_html=True)
    st.markdown(f'<div class="section-label">📝 Full Pathological Report ({selected_lang})</div>', unsafe_allow_html=True)
    with st.spinner("Generating Report..."):
        full_report = generate_medical_report(res_label, confidence_pct, selected_lang)
        st.markdown(f"<div style='white-space: pre-wrap; color:#cbd5e1;'>{full_report}</div>", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Row 4: Probability Distribution
    st.markdown('<div class="med-card"><div class="section-label">📊 Probability Analysis</div>', unsafe_allow_html=True)
    df = pd.DataFrame({'Condition': CLASS_NAMES, 'Probability': [float(p) for p in probs]})
    fig_bar = px.bar(df, x='Condition', y='Probability', color='Probability', template='plotly_dark')
    st.plotly_chart(fig_bar, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("<p style='text-align:center; color:#475569; font-size:0.7rem;'>PROTOTYPE v2.8 • SMART VOICE ALERTS • MULTILINGUAL ENABLED</p>", unsafe_allow_html=True)