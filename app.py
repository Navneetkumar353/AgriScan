# app.py — YOLOv8 classification + TTA + optional ensemble + ChatGroq remedy
import os, json, numpy as np
import streamlit as st
from ultralytics import YOLO
from PIL import Image, ImageOps
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()  # reads .env (GROQ_API_KEY=...)

# ---------------------- styling ----------------------
st.set_page_config(page_title=" AgriScan", page_icon="🌿", layout="wide")

CARD_CSS = """
<style>
.block-container {padding-top: 1.2rem; padding-bottom: 2rem;}

.card {border-radius: 16px; padding: 16px 18px; margin: 6px 0 14px 0; box-shadow: 0 6px 18px rgba(0,0,0,0.14);}
.card.success {background: linear-gradient(180deg, rgba(0,160,110,0.12), rgba(0,160,110,0.06)); border: 1px solid rgba(0,160,110,0.35);}
.card.warn {background: linear-gradient(180deg, rgba(220,160,0,0.12), rgba(220,160,0,0.06)); border: 1px solid rgba(220,160,0,0.35);}
.pill {display:inline-block; padding:4px 10px; border-radius:999px; font-size:0.82rem; border:1px solid rgba(255,255,255,0.15); margin-right:6px;}
.codepill {font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, "Liberation Mono";}
.table-compact td, .table-compact th { padding: 6px 8px; }

/* --- anti-overlap: keep big images under control and stack columns on narrower widths --- */
.stImage img {max-height: 540px; width: 100%; object-fit: contain; border-radius: 8px}

/* Stack the two main columns vertically on <=1200px screens (prevents overlap on some laptops) */
@media (max-width: 1200px){
  [data-testid="stHorizontalBlock"] {flex-direction: column !important;}
  [data-testid="stHorizontalBlock"] > div {width: 100% !important;}
}

/* Keep side tables compact so they don't crowd controls */
[data-testid="stDataFrame"] table td, 
[data-testid="stDataFrame"] table th { padding: 6px 8px !important; font-size: 0.92rem; }

/* Ensure result area paints above neighbors if any floaty elements appear */
.result-wrap { position: relative; z-index: 2; }
</style>
"""
st.markdown(CARD_CSS, unsafe_allow_html=True)
st.title("🌿 AgriScan")

# ---------------------- sidebar ----------------------
with st.sidebar:
    st.header("⚙️ Settings")
    weights_a = st.text_input("YOLOv8 best.pt (primary)", "runs/classify/train2/weights/best.pt")
    weights_b = st.text_input("YOLOv8 best.pt (optional ensemble)", "runs/classify/train/weights/best.pt")
    classes_path = st.text_input("class_names.json", "class_names.json")

    conf_thr = st.slider("Low-confidence warn threshold", 0.10, 0.95, 0.55, 0.01)
    img_size = st.selectbox("Inference size", [224, 256, 320], index=2)
    use_tta  = st.toggle("Enable TTA (avg original + flip)", value=True)
    use_ens  = st.toggle("Enable 2-model ensemble (if path B set)", value=False)
    st.caption("Tip: 320 gives a small accuracy bump. TTA/ensemble improve stability without changing data.")

# Safe secrets helper (works locally and on Streamlit Cloud)
def _get_groq_key() -> str:
    k = os.getenv("GROQ_API_KEY", "")
    if k:
        return k
    try:
        return st.secrets["GROQ_API_KEY"]
    except Exception:
        return ""
GROQ_API_KEY = _get_groq_key()

@st.cache_resource
def load_model(w):
    return YOLO(w)

@st.cache_data
def load_classes(p):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def predict_one(model, img, imgsz):
    """Single-view softmax scores."""
    res = model.predict(img, imgsz=imgsz, verbose=False)
    return res[0].probs.data.cpu().numpy()

def predict_tta(model, img, imgsz):
    """Average over original + horizontal flip."""
    views = [img, ImageOps.mirror(img)]
    probs = [predict_one(model, v, imgsz) for v in views]
    return np.mean(probs, axis=0)

# Load models/classes
model_a = load_model(weights_a)
classes = load_classes(classes_path)
model_b = load_model(weights_b) if (use_ens and weights_b and os.path.exists(weights_b)) else None

# ---------------------- layout ----------------------
left, right = st.columns([1.05, 1], gap="large")

with left:
    st.subheader("📸 Upload a leaf image")
    file = st.file_uploader("JPG/PNG up to 200MB", type=["jpg", "jpeg", "png"])
    if file:
        img = Image.open(file).convert("RGB")
        # IMPORTANT: responsive width + max-height via CSS avoids overlap
        st.image(img, caption="Uploaded", use_container_width=True)

with right:
    # Wrap right column in container with higher paint order (belt-and-suspenders)
    with st.container():
        st.markdown('<div class="result-wrap">', unsafe_allow_html=True)
        st.subheader("🔎 Result")
        if file:
            with st.spinner("Predicting…"):
                # primary model (with or without TTA)
                scores_a = predict_tta(model_a, img, img_size) if use_tta else predict_one(model_a, img, img_size)
                scores = scores_a

                # optional ensemble with model_b
                if model_b is not None:
                    scores_b = predict_tta(model_b, img, img_size) if use_tta else predict_one(model_b, img, img_size)
                    scores = (scores_a + scores_b) / 2.0

            idx = int(np.argmax(scores))
            pred_name = classes[idx] if idx < len(classes) else f"class_{idx}"
            conf = float(scores[idx])

            # confidence bar
            st.write("Confidence")
            st.progress(min(max(conf, 0.0), 1.0))

            # result card
            card_class = "warn" if conf < conf_thr else "success"
            st.markdown(
                f'<div class="card {card_class}">'
                f'<div style="font-size:1.05rem;"><b>Prediction:</b> {pred_name}</div>'
                f'<div class="pill codepill">conf {conf:.2f}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

            # top-3
            top3 = np.argsort(scores)[::-1][:3]
            st.write("Top-3")
            rows = [(classes[i] if i < len(classes) else f"class_{i}", float(scores[i])) for i in top3]
            st.dataframe({"Class": [r[0] for r in rows], "Score": [f"{r[1]:.2f}" for r in rows]},
                        hide_index=True, use_container_width=True)

            # tabs
            t1, t2 = st.tabs(["🧪 Remedy", "📊 Raw scores"])
            with t1:
                # Button is more reliable than a small toggle near tables on some devices,
                # but keeping your toggle as requested:
                if st.toggle("Explain remedy with integrated AI", value=False):
                    if not GROQ_API_KEY:
                        st.info("Set GROQ_API_KEY in .env (local) or in Streamlit Cloud → Settings → Secrets.")
                    else:
                        llm = ChatGroq(temperature=0.3, groq_api_key=GROQ_API_KEY,
                                    model_name="llama-3.3-70b-versatile")
                        prompt = (
                            f"The detected plant class is '{pred_name}'. "
                            f"Give a short action plan:\n"
                            f"1) Immediate treatment (dosage & PPE)\n"
                            f"2) Next 7–10 days (monitor/rotation)\n"
                            f"3) Nutrition/irrigation notes\n"
                            f"Be conservative; avoid off-label advice."
                        )
                        with st.spinner("Drafting remedy…"):
                            st.markdown(llm.invoke(prompt).content)
            with t2:
                order = np.argsort(scores)[::-1][:10]
                st.dataframe(
                    {
                        "Rank": list(range(1, len(order)+1)),
                        "Class": [classes[i] if i < len(classes) else f"class_{i}" for i in order],
                        "Score": [float(scores[i]) for i in order],
                    },
                    hide_index=True, use_container_width=True
                )
        else:
            st.info("Upload a leaf image on the left to begin.")
        st.markdown('</div>', unsafe_allow_html=True)
