# 🕵️ Two-Branch Deepfake Detector

### Self-Supervised Spatial & Temporal Generalization

실제 얼굴 영상만으로 **공간적(Spatial) 조작 아티팩트**와 **시간적(Temporal) 불일치**를 각각 학습하고,
두 모델을 결합하여 **다양한 딥페이크 기법에 일반화 가능한 탐지 시스템**을 구현했습니다.


---

## 📖 목차

- [소개](#-소개)
- [시스템 아키텍처](#-시스템-아키텍처)
- [주요 특징](#-주요-특징)
- [기술 스택](#-기술-스택)
- [모델 구조](#-모델-구조)
- [프로젝트 구조](#-프로젝트-구조)
- [설치](#-설치)
- [추론 실행](#-추론-실행)
- [성능](#-성능)
- [향후 계획](#-향후-계획)

---

## 📋 소개

기존 딥페이크 탐지 모델은 특정 생성 모델(FaceSwap, Face2Face 등)의 흔적을 암기하는 방식이 많아, 학습에 없던 새로운 조작 기법에는 성능이 크게 떨어지는 한계가 있습니다.

이 프로젝트는 **Zero-shot Generalization**을 목표로, 실제(real) 영상만을 이용해 두 종류의 pseudo-fake를 self-supervised 방식으로 직접 생성하고 학습합니다.

| Branch | 대응 조작 유형 |
|---|---|
| **Spatial (SBI)** | 얼굴 블렌딩 경계를 학습하여 Face Swap 계열 탐지 |
| **Temporal** | 입 움직임의 시간적 불일치를 학습하여 Reenactment / Lip-sync 계열 탐지 |

각 브랜치는 서로 다른 조작 유형에 특화되어 있으며, 최종적으로 두 결과를 융합해 더 강건한 탐지 성능을 제공합니다.

---

## 🏗 시스템 아키텍처

```
Raw Videos
    │
    ▼
Face Detection & Landmark Extraction (Py-Feat)
    │
    ├──────────────────────────┐
    ▼                          ▼
Spatial Pipeline           Temporal Pipeline
   (SBI)                    (Temporal)
    │                          │
Self-Blended Images     Temporal Pseudo-Fake
    │                          │
EfficientNet-B4           CNN + BiGRU
    │                          │
Swap Score                Temporal Score
    └──────────────┬───────────┘
                   ▼
            Score Fusion (Max)
                   ▼
              Real / Fake
```

---

## ✨ 주요 특징

### 🔹 Self-Supervised Spatial Learning
실제 얼굴 이미지로부터 Self-Blended Image(SBI)를 생성하여 다음과 같은 Face Swap 계열의 공통 아티팩트를 학습합니다.
- Blending Boundary
- Color Mismatch
- Compression Artifact

### 🔹 Temporal Consistency Learning
실제 연속 영상에서 다음을 생성하여
- Mouth Temporal Splicing
- Temporal Jitter

Reenactment / Lip-sync / Talking Head 계열에서 나타나는 시간적 불일치를 학습합니다.

### 🔹 On-the-fly Pseudo-Fake Generation
모든 fake 이미지를 미리 저장하지 않고, 학습 중 Self-Blended Image와 Temporal Fake를 실시간으로 생성합니다. 저장 공간을 크게 줄이면서 데이터 다양성을 확보했습니다.

### 🔹 Streaming Data Pipeline
DFDC, FF++ 등 대용량 데이터셋을 아래 방식으로 처리해 디스크 사용량을 최소화했습니다.

```
Download → Process → Feature Extraction → Delete
```

### 🔹 Two-Branch Fusion
Spatial 모델과 Temporal 모델은 서로 다른 조작 유형에 강점을 가지므로, 다음과 같이 융합해 일반화 성능을 높였습니다.

```
Final Score = max(Spatial Score, Temporal Score)
```

---

## 🛠 기술 스택

| 분류 | 사용 기술 |
|---|---|
| Language | Python |
| Deep Learning | PyTorch, timm |
| Computer Vision | OpenCV, Py-Feat |
| Data Processing | NumPy, Albumentations |

---

## 🧠 모델 구조

| Branch | Architecture | 목적 |
|---|---|---|
| Spatial | EfficientNet-B4 + Self-Blended Images | Face Swap 탐지 |
| Temporal | EfficientNet-B0 + BiGRU | Reenactment / Lip-sync 탐지 |
| Fusion | Max Score Fusion | 최종 판단 |

---

## 📂 프로젝트 구조

```
scripts/
├── sbi/
│   ├── train.py
│   ├── dataset.py
│   ├── sbi_gen.py
│   ├── temporal_train.py
│   ├── temporal_dataset.py
│   ├── temporal_gen.py
│   ├── detect.py
│   ├── eval_sbi.py
│   └── temporal_eval.py
│
├── extract_features.py
├── stream_extract.py
├── stream_ffpp_convnext_frames.py
└── dfdc_extract.py

outputs/
├── sbi/
└── temporal/
```

---

## 🚀 설치

```bash
git clone https://github.com/soyyoon/Two-Branch-Deepfake-Detector.git
cd Two-Branch-Deepfake-Detector

conda create -n sbi python=3.10
conda activate sbi
pip install -r requirements.txt
```

> **Py-Feat**는 별도의 conda 환경에서 실행해야 합니다.

```bash
conda create -n pyfeat python=3.10
conda activate pyfeat
pip install py-feat
```

---

## 🚀 추론 실행

**1. Face Crop 추출**
```bash
python local_video_crops.py
```

**2. Temporal Clip 추출**
```bash
python local_video_clips.py
```

**3. 딥페이크 탐지**
```bash
python detect.py
```

---

## 📊 성능

### Spatial Branch (SBI)

| Dataset | AUC |
|---|---|
| CelebDF | 0.92 |
| DFDC | 0.84 |
| FF++ | 0.77 |
| WildDeepfake | 0.69 |
| **Macro AUC** | **0.80** |

### Temporal Branch

Temporal 모델은 **NeuralTextures, Face2Face, Lip-sync** 계열에서 Spatial 모델보다 높은 탐지 성능을 보였습니다.