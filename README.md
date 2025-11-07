# 🌿 AgriScan — AI-Powered Crop Disease Detection & Remedy Recommendation

**AgriScan** is an intelligent crop health monitoring system built to assist farmers and agronomists in detecting plant diseases using **deep learning** and recommending appropriate **fertilizers or treatments** through an integrated **language model (ChatGroq with LLaMA-3)**.  

The system can classify diseases in **tomato, paddy, wheat**, and other Indian crops using drone or manual leaf images.  
It combines **YOLOv8 (Ultralytics)** for image classification and **LangChain + ChatGroq** for generating natural-language treatment suggestions — all hosted in an interactive **Streamlit web app**.

---

## 🧭 Project Overview

| Feature | Description |
|----------|-------------|
| **Disease Detection** | Uses pretrained YOLOv8s classifier trained on the [PlantDoc Dataset]([https://www.kaggle.com/datasets/pratikkayal/plantdoc-dataset](https://www.kaggle.com/datasets/abdulhasibuddin/plant-doc-dataset) |
| **Remedy Suggestions** | Integrates **ChatGroq (LLaMA-3.3-70B)** through LangChain for fertilizer and pesticide recommendations |
| **Web Interface** | Built using **Streamlit**, allowing simple drag-and-drop image uploads |
| **Drone Concept** | Simulates drone-captured imagery for real-time monitoring (conceptual stage) |
| **Model Ensemble + TTA** | Improves classification confidence using ensemble averaging and test-time augmentation |
| **Deployment** | Deployed seamlessly on **Streamlit Cloud** with Groq API key stored in app secrets |

---

## 🧠 Tech Stack

| Layer | Technology |
|-------|-------------|
| **Model** | YOLOv8s / Ultralytics |
| **Deep Learning Framework** | PyTorch |
| **Frontend/UI** | Streamlit |
| **LLM Integration** | LangChain + ChatGroq (LLaMA-3) |
| **Dataset** | PlantDoc (Kaggle) |
| **Deployment Platform** | Streamlit Cloud |
| **Programming Language** | Python 3.12 |

---

## 📁 Folder Structure

```

AgriScan/
├── app1.py                     # Streamlit app (main UI)
├── requirements.txt            # Python dependencies
├── runtime.txt                 # Fixes Python version (3.12)
├── packages.txt                # Adds required system libraries for OpenCV
├── class_names.json            # Label mapping for trained classes
├── .gitignore                  # Excludes data and large files
└── runs/
└── classify/
├── train/weights/best.pt      # Ensemble model 1
└── train2/weights/best.pt     # Primary model

````

---

## 🧩 Dataset

- **Dataset Name:** [PlantDoc Dataset](https://www.kaggle.com/datasets/pratikkayal/plantdoc-dataset)
- **Size:** ~938 MB (Approx.)
- **Structure:**  
  - `train/` and `test/` directories  
  - Each crop has multiple disease folders (e.g., *Tomato Early Blight*, *Tomato Mosaic Virus*, *Tomato Leaf Yellow Virus*).

---

## ⚙️ Model Training

AgriScan uses **YOLOv8s** trained for **15 epochs** on PlantDoc.

**Training Script:** `train_yolo.py`
```bash
python train_yolo.py
````

* **Image Size:** 320
* **Batch Size:** 32
* **Optimizer:** AdamW
* **Epochs:** 15
* **TTA & Ensemble:** Enabled during inference for stability

Best weights are saved at:

```
runs/classify/train2/weights/best.pt
runs/classify/train/weights/best.pt
```

---

## 💻 Local Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/<your-username>/AgriScan.git
cd AgriScan
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv
.\.venv\Scripts\activate       # (Windows)
source .venv/bin/activate      # (Mac/Linux)
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Add Environment Variables

Create a `.env` file in the project root:

```
GROQ_API_KEY=your_api_key_here
```

### 5. Run the App

```bash
streamlit run app1.py
```

---

## 🌐 Streamlit Cloud Deployment

### Push to GitHub

Ensure your repository includes:

* `app1.py`
* `requirements.txt`
* `runtime.txt`
* `packages.txt`
* Two model weights (`runs/classify/train/weights/best.pt` and `runs/classify/train2/weights/best.pt`)

---

## 🧠 How It Works

1. **Upload Image** — User uploads a leaf photo.
2. **Inference (YOLOv8)** — The app classifies the disease and computes confidence scores.
3. **Low-Confidence Warning** — If confidence < threshold, a yellow warning appears.
4. **Remedy Generation** — The app queries ChatGroq (LLaMA-3) via LangChain to produce:

   * Immediate treatment
   * Follow-up plan (7–10 days)
   * Nutrient & irrigation recommendations
5. **Display Results** — A clean, card-style UI shows prediction, confidence bar, and remedy plan.

---

## 🧪 Example Predictions

| Input Leaf                       | Predicted Disease          | Confidence | Suggested Remedy                                               |
| -------------------------------- | -------------------------- | ---------- | -------------------------------------------------------------- |
| Tomato leaf with yellow spots    | *Tomato Leaf Yellow Virus* | 0.89       | Apply imidacloprid, monitor for aphids, maintain field hygiene |
| Tomato with brown circular spots | *Early Blight*             | 0.94       | Use Mancozeb spray, rotate crops, and avoid overhead watering  |
| Healthy leaf                     | *Tomato Leaf*              | 0.97       | No treatment needed                                            |

---

## 🧩 Technologies Used

* **YOLOv8** for classification
* **PyTorch** for model backend
* **LangChain + ChatGroq (LLaMA-3)** for intelligent remedy generation
* **Streamlit** for UI deployment
* **OpenCV (headless)** for image preprocessing

---

## 📦 Deployment Details

| File               | Purpose                                                         |
| ------------------ | --------------------------------------------------------------- |
| `runtime.txt`      | Ensures Python 3.12 environment                                 |
| `packages.txt`     | Installs system libraries (`libgl1`, `libglib2.0-0`) for OpenCV |
| `requirements.txt` | Defines all Python dependencies                                 |
| `.gitignore`       | Excludes `data/`, `.env`, and unnecessary `runs/` files         |

---

## 📈 Model Performance

| Metric             | Score |
| ------------------ | ----- |
| **Top-1 Accuracy** | 68.6% |
| **Top-5 Accuracy** | 97.0% |
| **Epochs**         | 15    |
| **Batch Size**     | 32    |

These results are consistent across test samples of the PlantDoc dataset.

---

## 🧭 Future Enhancements

* Expand to support **Paddy, Cotton, Chilli, Sugarcane** crops.
* Real-time drone image analysis.
* Add **voice-based interaction** in regional languages.
* Introduce weather and soil-based fertilizer recommendations.
* Build a **mobile version** of AgriScan for Android/iOS.

---

## 📜 License

This project is released under the **MIT License**.
You’re free to use, modify, and distribute it for educational or research purposes.

