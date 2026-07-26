# Phase 1 — SBI Self-Blending 탐지기 착수 플랜

## 목표 & 성공 지표
실제 얼굴만으로 self-blended 가짜를 즉석 생성해 **기법-무관 블렌딩 아티팩트**를 학습하는 탐지기.
성공 = 우리 **LODO cross-domain 프로토콜**에서 baseline(ConvNeXt/AU) 대비 **미지 도메인 AUC 상승**.
목표 야생 성능 현실치 ~0.75-0.85(0.95 아님). 특히 WildDeepfake(현 최고 0.636) / DFDC에서 개선 여부가 핵심.

참고 구현: **mapooon/SelfBlendedImages**(CVPR'22 공식). sbi 생성·마스크·증강 파라미터를 그대로 포팅하는 게
효과의 관건(SBI는 디테일 민감). FSBI(2025, 주파수 강화)는 Phase 2.

## 왜 이게 우리 실패를 겨냥하나
- ConvNeXt는 DFD/CelebDF **지문을 암기** → 미지 기법 붕괴(FF++ 0.53). SBI는 fake 데이터셋을 안 쓰고
  **모든 face-swap 공통의 블렌딩 경계+통계 불일치**만 학습 → 기법 암기 불가.
- "다양성 풀링"은 AU에서 실패했지만, SBI의 다양성은 **실제(real) 얼굴의 도메인 다양성**이라 상충 없음
  (real 크롭을 FF++/CelebDF/WildDeepfake에서 섞음 = 우리가 원하던 좋은 다양성).

## 0. 환경 (신규 `sbi` conda env — au/pyfeat 안 건드림)
- python 3.10, **torch 2.6.0+cu124**(드라이버 550 호환, cuDNN 정상), timm, albumentations, opencv,
  scikit-image, tqdm.
- 랜드마크: **dlib + 81-pt 예측기**(SBI 원본 충실) 또는 설치 쉬운 `face_alignment`(FAN, GPU). 1개 선택.
  → dlib 우선(SBI 마스크가 81-pt 기준), 설치 난망 시 face_alignment 폴백.
- 학습에 cuDNN 필요(속도) → au env의 cudnn-disable 회피 목적으로 별도 env가 정답.

## 1. 데이터 — 실제 얼굴 크롭 풀 + 랜드마크 캐시
- **real 크롭 확장**: 현재 부족(≈770). 스트리밍(`stream_extract`/filelist 재사용)으로 CelebDF real +
  FF++ real 프레임 추가 확보. WildDeepfake real(481 시퀀스) 포함 → **도메인-다양 real 풀**.
  목표 ≈ 수백 영상 / 수만 크롭.
- **랜드마크 사전계산**: real 크롭마다 81(or 68)-pt 1회 검출 → `<crop>.lmk.npy` 캐시(온더플라이 학습 중 재사용).
  실패(무검출) 크롭은 매니페스트에서 제외.
- 산출: `data_sbi/real_manifest.csv`(image_path, lmk_path, source_domain).

## 2. SBI 생성 모듈 `scripts/sbi/sbi_gen.py` (핵심)
공식 repo 포팅. 실제 이미지 I + 랜드마크 L 로부터:
1. **Source-Target Generator**: I를 두 갈래로 증강해 통계차 생성 — source측에 RGBShift/HueSaturationValue/
   RandomBrightnessContrast + downscale(해상도 불일치) + sharpen/blur 중 랜덤.
2. **마스크 M**: L의 convex hull → **랜덤 elastic 변형** → **경계 Gaussian 블러**(소프트 블렌딩 경계).
3. **블렌드**: source 얼굴에 소폭 affine(정합 오차 모사) 후 `I_blend = I_t*(1-M) + I_s*M`.
4. 반환: (I_blend, label=1=fake), (I, label=0=real).
검증: 생성 샘플 몇 개 시각 저장(경계·색 불일치가 보이는지 눈으로 확인).

## 3. Dataset/DataLoader `scripts/sbi/dataset.py`
- `SBIDataset`: real 이미지 1장 입력 → 확률 0.5로 self-blend(fake) or 원본(real) → 온더플라이 균형.
- **분류-시 증강(야생 강건 핵심)**: 압축(JPEG q랜덤)/다운스케일-업스케일/노이즈/블러/색 → **WildDeepfake급
  저화질 시뮬레이션**(우리 모델들이 야생서 무너진 주원인 대응). → resize 380(EffB4) → normalize.

## 4. 모델 & 학습 `scripts/sbi/train.py`
- 백본: **EfficientNet-b4**(timm, SBI 원본) — 대안 ConvNeXt-tiny(기존 자산 재사용). B4 우선.
- 손실 BCE, 옵티마 AdamW(원본 SAM은 Phase 2 개선), lr warmup+cosine, grad clip, AMP.
- val: held-out real+fake 소량(예: CelebDF val 일부)로 AUC 모니터, best 체크포인트 저장.
- 자원: 4×A6000. 수만 iter, 수시간 예상. 로그/체크포인트 `outputs/sbi/`.

## 5. 평가 — 기존 LODO 인프라 재사용
- 이미지-레벨 → **영상당 프레임 평균**(ConvNeXt와 동일 규약)로 비디오 스코어.
- 테스트 도메인: **CelebDF test / FF++ test(기법별) / WildDeepfake / DFDC(미러)**. SBI는 real만 학습 →
  모든 fake가 미지 = 순수 일반화 측정.
- `ensemble_eval.py`/per-method 스크립트 재사용해 baseline(ConvNeXt·AU) 대비 표 + macro.
- **핵심 비교**: SBI vs ConvNeXt vs AU per-domain, 특히 WildDeepfake·DFDC.

## 6. 마일스톤
1. env 구축 + real 풀 확장 + 랜드마크 캐시. (검증: 생성 SBI 샘플 시각 확인)
2. 학습 스모크(소량 iter, 로직·GPU·cuDNN 확인) → 본 학습.
3. LODO 평가 → baseline 대비 개선 확인.
4. (개선 시) Phase 2: 주파수 브랜치(FSBI) + 캘리브레이션/기권.

## 리스크 / 결정포인트
- **랜드마크 품질**: 저화질 야생 real에 검출 실패 가능 → real 풀은 우선 FF++/CelebDF(고화질) 중심,
  WildDeepfake real은 검출되는 것만.
- **SBI 디테일 충실도**: 마스크 변형·블러·source 증강 파라미터가 성능 좌우 → 공식 repo 값 준수.
- **cuDNN/env**: 신규 sbi env로 격리(au/pyfeat 보존).
- **기대치**: cross-dataset은 필드 난제. 개선폭이 작을 수 있음 → 그래도 "지문암기 회피"라는 올바른 축.

## 유지/폐기 (재확인)
- 유지: LODO 프로토콜, 스트리밍 파이프라인, WildDeepfake/DFDC held-out, 비디오-레벨 집계 규약.
- 폐기: AU 주력, 앙상블-라우팅, in-dist AUC 최적화.
