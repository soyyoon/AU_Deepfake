# Phase 3 — Reenactment/Temporal 브랜치 계획

## 동기 (실측 근거)
SBI(블렌딩 탐지)는 face-swap엔 강하나 **reenactment/lip-sync엔 구조적 blind**. 실증:
kakao_3(저커버그 reenactment fake) SBI max 0.44·전프레임 낮음 → 집계로도 못 살림.
FF++ Face2Face 0.667/NeuralTextures 0.676(SBI 약). 별도 브랜치 필요.

## 핵심 원칙 (SBI 교훈 계승)
특정 fake 데이터셋 암기(=ConvNeXt 실패) 대신 **reenactment 보편 아티팩트를 자기지도 생성**.
reenactment = 얼굴 안 바꾸고 일부 영역(주로 입) 시간축 조작 → **공간 블렌딩 없음, 시간 불일치 있음**.

## 설계
1. **자기지도 temporal pseudo-fake:** 실제 연속 클립에서
   - (a) 입 영역을 **같은 사람 다른 시점 프레임**으로 스플라이스(랜드마크 48-67 마스크) → lip-sync식 시간 불일치
   - (b) 입/얼굴 영역 프레임별 미세 워프 → 시간 jitter
   - 원본 클립=real, 조작 클립=fake.
2. **모델:** 공유 2D CNN(EfficientNet-b0/SBI백본) per-frame feature → 시간 모듈(GRU 또는 1D-conv/attention) → 클립 fake 점수. (경량 우선; (2+1)D CNN은 후보)
3. **데이터:** 실제 얼굴 **연속 dense 클립**(예: 16프레임 연속). 기존 data_sbi/real은 uniform 샘플이라 부적합 → 재추출(스트리밍, 연속 구간). + 랜드마크(입 마스크용, pyfeat 68-pt 재사용).
4. **야생 열화 증강**(압축/다운스케일) — 배치 강건성.

## 평가 (기존 LODO 인프라 재사용)
- FF++ reenactment(Face2Face/NeuralTextures) held-out AUC(vs SBI 0.667/0.676).
- kakao_3(reenactment) 잡는지.
- face-swap(Deepfakes/FaceSwap)·CelebDF에서 성능 훼손 없는지(SBI와 상보 확인).
- 최종: **SBI + reenactment 브랜치 융합**(각 독립 일반화 검증 후, LODO)이 단일 SBI를 넘는지.

## 단계
1. temporal pseudo-fake 생성기 + dense 클립 파일럿 → **시각 검증**(입 시간 불일치가 보이는가).
2. dense 클립 대량 추출(실제 영상, 연속 구간) + 랜드마크.
3. 시간 모델 학습(sbi env).
4. LODO 평가 + kakao_3 + 융합.

## 리스크
- 자기지도 temporal 아티팩트가 실제 reenactment와 충분히 유사한가(SBI만큼 통할지 불확실).
- dense 클립 추출 비용(스트리밍 재실행).
- 융합이 face-swap 성능 해치지 않도록(독립 검증 후 게이팅/평균).
관련: [[sbi-redesign]] docs/sbi_phase1_plan.md
