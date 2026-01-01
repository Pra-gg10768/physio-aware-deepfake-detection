# 🧠 Physio‑Aware Deepfake Detection using rPPG & Multi‑ROI Consistency

A research‑grade deepfake detection system that leverages **physiological signals (remote Photoplethysmography – rPPG)** and **cross‑facial region coherence** to detect manipulated videos.

This project is **explicitly designed to generalize across datasets** and avoid identity, compression, or artifact shortcuts commonly exploited by CNN‑only approaches.

---

## 📌 Problem Statement

Deepfake videos generated using GANs and diffusion models often appear visually convincing but **fail to preserve subtle physiological signals** such as blood‑flow‑induced skin color variations.

Most existing deepfake detectors:

* Overfit to **visual artifacts**
* Fail under **cross‑dataset evaluation**
* Break when compression or resolution changes

🔴 **Key Insight:**

> While appearance can be faked, **physiological coherence across facial regions cannot be perfectly synthesized**.

This project detects deepfakes by modeling:

* rPPG signal quality
* Temporal stability
* Cross‑ROI (Region of Interest) physiological consistency

---

## 🎯 Core Contributions

✔ Multi‑ROI rPPG extraction (face, cheeks, forehead, nose, chin)
✔ Window‑based temporal physiological features
✔ Cross‑ROI correlation modeling (physiological coherence)
✔ Cross‑dataset evaluation (Celeb‑DF → DFD)
✔ Classical ML models (Logistic, SVM, XGBoost) for interpretability

---

## 🧬 What is rPPG?

**Remote Photoplethysmography (rPPG)** measures subtle skin color changes caused by blood volume variations using standard RGB cameras.

In real videos:

* rPPG signals are **stable, periodic, and correlated** across the face

In deepfakes:

* rPPG signals are **noisy, unstable, and spatially inconsistent**

---

## 🧱 Project Architecture

```
Raw Videos (.mp4)
        ↓
MediaPipe FaceMesh
        ↓
Multi‑ROI Mean RGB Signals
        ↓
Temporal Alignment
        ↓
Saved as .npy  (T × ROI × RGB)
        ↓
Physiological Feature Extraction
        ↓
ML Classifiers (SVM / XGBoost)
        ↓
Cross‑Dataset Evaluation
```

---

## 📂 Directory Structure

```
physio-aware-deepfake-detection/
│
├── data/
│   ├── raw_videos/        # Original mp4 videos
│   ├── signals/           # Extracted rPPG signals (.npy)
│   └── signals_dfd_test/  # DFD test signals
│
├── src/
│   ├── preprocessing.py      # Video → rPPG signal extraction
│   ├── features.py           # Physiological feature engineering
│   ├── dataset.py            # Training dataset loader
│   ├── dataset_dfd.py        # DFD test dataset loader
│   └── classifier.py         # ML model definitions
│
├── experiments/
│   ├── run_classification.py # Train & evaluate models
│   ├── run_dfd_test.py       # Cross‑dataset testing
│   
│   
│
├── models/                # Saved trained models
└── README.md
```

---

## 👁 Facial Regions of Interest (ROIs)

| ROI      | Purpose                         |
| -------- | ------------------------------- |
| Face     | Global physiological signal     |
| Forehead | Strong rPPG due to low motion   |
| Cheeks   | 🔥 Highest discriminative power |
| Nose     | Moderate stability              |
| Chin     | Least informative               |

Each ROI produces an independent rPPG signal.

---

## 🧪 Feature Engineering

### 1️⃣ Core rPPG Frequency Features

* **Peak Frequency** – estimated heart rate
* **Peak Sharpness** – signal consistency
* **Band Energy (0.7–3.0 Hz)** – physiological relevance
* **Low/High Band Ratio**

### 2️⃣ Spectral Features

* Spectral Entropy
* Spectral Flatness
* HR Stability

### 3️⃣ Temporal Window Features

* Window‑wise variance
* Temporal instability

### 4️⃣ Signal Quality Features

* Jitter (frame‑to‑frame noise)
* Signal instability

### 5️⃣ Cross‑ROI Correlation (Key Innovation)

* Pearson correlation between all ROI pairs
* Measures **physiological coherence**

➡️ Final feature vector size: **68 features per video**

---

## 🤖 Models Used

| Model               | Why Used                        |
| ------------------- | ------------------------------- |
| Logistic Regression | Baseline interpretability       |
| SVM (RBF)           | Strong boundary modeling        |
| XGBoost             | Non‑linear feature interactions |

All models use:

* StandardScaler
* Probability outputs

---

## 🏋️ Training Dataset

* **Celeb‑DF (Real + Fake)**
* Balanced training split
* Subject‑independent videos

---

## 🧪 Cross‑Dataset Evaluation (Critical)

📌 **No fine‑tuning performed**

Tested on:

* **DFD Original Sequences Dataset**

This evaluates **true generalization**.

---

## 📊 Results

### Training‑Set Performance

| Model    | Accuracy  | ROC‑AUC   |
| -------- | --------- | --------- |
| Logistic | 0.789     | 0.830     |
| SVM‑RBF  | **0.927** | **0.952** |
| XGBoost  | **0.936** | **0.955** |

---

### Cross‑Dataset (DFD) Results

| Model    | Mean Fake Probability | Std       |
| -------- | --------------------- | --------- |
| Logistic | 0.777                 | 0.318     |
| SVM‑RBF  | **0.933**             | **0.098** |
| XGBoost  | 0.921                 | 0.132     |

✔ High confidence without retraining
✔ Low variance → stable predictions

---

## 🔬 ROI Ablation Study

| ROI      | Mean Fake Probability |
| -------- | --------------------- |
| Face     | High                  |
| Cheeks   | 🔥 Highest            |
| Forehead | Moderate              |
| Nose     | Low                   |
| Chin     | Lowest                |

➡️ Confirms **cheeks carry strongest physiological cues**

---

## 📈 Why This Works

Deepfakes:

* Break spatial blood‑flow coherence
* Introduce temporal jitter
* Fail to synchronize physiology across regions

This system detects **what GANs cannot fake well**.

---

## 🚀 How to Run

### 1️⃣ Preprocessing

```bash
python -m src.preprocessing
```

### 2️⃣ Train & Evaluate

```bash
python -m experiments.run_classification
```

### 3️⃣ Cross‑Dataset Test

```bash
python -m experiments.run_dfd_test
```

### 4️⃣ ROI Ablation

```bash
python -m experiments.run_roi_ablation
```

---


