# Two-Branch Deepfake Detector — SBI + Temporal

일반화 우선(generalization-first) 딥페이크 탐지기. 특정 조작 데이터셋의 "지문"을 암기하는 대신,
**자기지도(self-supervised)로 조작의 보편 아티팩트를 생성·학습**해 미지 도메인/기법에 일반화한다.

서로 다른 조작 유형을 잡는 두 개의 상보 브랜치:

| 브랜치 | 학습 신호 (자기지도, fake 데이터셋 미사용) | 잡는 조작 |
|---|---|---|
| **SBI** (Self-Blended Images) | 실제 얼굴을 자기 자신에 블렌딩 → **공간 블렌딩 경계** | face-swap (Deepfakes, FaceSwap, FaceShifter …) |
| **Temporal** | 실제 연속 클립의 입 영역을 시간축 스플라이스 → **시간 불일치** | reenactment / lip-sync (Face2Face, NeuralTextures …) |

각 브랜치가 **독립적으로 일반화**하므로 융합이 실제로 작동한다.
최종 판정(OR): `SBI(swap) 또는 Temporal(reenact) 중 하나라도 임계값 초과 → FAKE`.

## 결과

4개 도메인 zero-shot 비디오 AUC (단일 SBI):

| CelebDF | FF++ | WildDeepfake | DFDC | **macro** |
|:---:|:---:|:---:|:---:|:---:|
| 0.92 | 0.77 | 0.69 | 0.84 | **0.80** |

- SBI 단일 모델이 4도메인 macro **0.80** — 기존 AU/ConvNeXt 앙상블의 오라클(0.78)을 초월.
- Temporal 브랜치가 FF++ **reenactment**에서 SBI 능가 (NeuralTextures 0.83 vs 0.68).
- 실제 야생 fake 3종(idol face-swap ×2 + Zuckerberg reenactment)을 두 브랜치가 함께 모두 탐지.

근거·실험 상세: [`outputs/sbi/sbi_results.md`](outputs/sbi/sbi_results.md),
[`outputs/temporal/temporal_results.md`](outputs/temporal/temporal_results.md) · 설계: [`docs/`](docs/)

## 설치

두 개의 conda 환경 (자세한 버전은 [`requirements.txt`](requirements.txt)):

```bash
# [1] 학습·추론 (sbi)
conda create -n sbi python=3.10 -y
conda run -n sbi pip install torch==2.6.0 torchvision==0.21.0 \
    --index-url https://download.pytorch.org/whl/cu124
conda run -n sbi pip install -r requirements.txt

# [2] 얼굴 검출·크롭·랜드마크 (pyfeat) — numpy 충돌로 반드시 분리
conda create -n pyfeat python=3.10 -y
conda run -n pyfeat pip install py-feat==0.6.2
```

## 추론 (영상 진위 판별)

```bash
PY=~/anaconda3/envs/pyfeat/bin/python

# video_id,path 형식의 csv 준비 (예: videos.csv)
#   video_id,path
#   myvid,/abs/path/to/video.mp4

# 1) 얼굴 크롭(SBI용) + dense 클립(Temporal용) 추출 — pyfeat env
PYTHONNOUSERSITE=1 $PY scripts/sbi/local_video_crops.py --targets videos.csv --out crops
PYTHONNOUSERSITE=1 $PY scripts/sbi/local_video_clips.py --targets videos.csv --out clips

# 2) 2-브랜치 탐지 — sbi env
PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=0 conda run -n sbi python scripts/sbi/detect.py \
    --crops crops --clips clips --vids myvid
```

출력 예:
```
video         SBI(swap)   Temporal  FUSED(OR)  판정(근거)
myvid            0.253      0.606      0.606  FAKE (reenact)
```

## 학습 (재현)

```bash
# 데이터: 실제 얼굴 영상 -> 크롭+랜드마크(SBI) / 연속 클립(Temporal) 추출 (pyfeat env)
PYTHONNOUSERSITE=1 $PY scripts/sbi/sbi_stream_crops.py --targets <real.csv> --out data_sbi/real
PYTHONNOUSERSITE=1 $PY scripts/sbi/cache_landmarks.py  --root data_sbi/real
PYTHONNOUSERSITE=1 $PY scripts/sbi/stream_clips.py     --targets <real.csv> --out data_sbi/clips

# 학습 (sbi env) — 각 브랜치는 real 데이터만으로 온더플라이 pseudo-fake 생성
conda run -n sbi python scripts/sbi/train.py          --root  data_sbi/real  --out outputs/sbi
conda run -n sbi python scripts/sbi/temporal_train.py --clips data_sbi/clips --out outputs/temporal
```

## 구조

```
scripts/
  stream_extract.py  stream_ffpp_convnext_frames.py  extract_features.py   # 공용 유틸(다운로드/얼굴크롭/pyfeat)
  sbi/
    sbi_gen · dataset · train · eval_sbi.py                       # SBI 브랜치
    temporal_gen · temporal_dataset · temporal_train · temporal_eval.py    # Temporal 브랜치
    detect.py                                                    # ★ 최종 2-브랜치 OR 탐지기
    cache_landmarks.py                                           # 68-pt 랜드마크(pyfeat)
    local_video_crops · local_video_clips.py                     # 로컬 영상 -> 크롭 / 클립 (추론용)
    sbi_stream_crops · stream_clips · dfdc_extract.py            # Kaggle 스트리밍 추출 (학습/평가용)
outputs/
  sbi/best.pt        temporal/best.pt                            # 학습된 모델
  sbi/*.md           temporal/*.md                               # 결과
docs/                                                            # 설계 문서
```

## 참고

- 대용량 데이터(크롭/클립/테스트셋)는 `.gitignore` 처리 — 추출 스크립트로 재생성.
- 모든 pyfeat 실행은 `PYTHONNOUSERSITE=1` 필요 (numpy 충돌 방지).
