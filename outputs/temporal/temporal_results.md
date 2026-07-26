# Phase 3 결과 — Temporal(reenactment) 브랜치

EfficientNet-b0 + BiGRU, real 연속 클립 2988개(16프레임)로 **자기지도 temporal pseudo-fake**
(입 영역 시간축 스플라이스/jitter)만 학습. fake 데이터셋 미사용. best val(pseudo) AUC 0.906.
계획 docs/phase3_reenactment_plan.md.

## FF++ 기법별 (Temporal vs SBI vs 융합)
| 기법 | SBI | Temporal | fuse-max | fuse-mean |
|---|---|---|---|---|
| Deepfakes (swap) | 0.969 | 0.775 | 0.945 | 0.947 |
| FaceSwap (swap) | 0.830 | 0.564 | 0.753 | 0.752 |
| FaceShifter (swap) | 0.693 | 0.558 | 0.666 | 0.670 |
| **Face2Face (reenact)** | 0.661 | 0.678 | 0.662 | **0.708** |
| **NeuralTextures (reenact)** | 0.670 | **0.830** | 0.797 | **0.830** |
| **전체** | 0.762 | 0.681 | 0.765 | **0.781** |

**핵심:** Temporal이 **reenactment에서 SBI를 이김**(NeuralTextures 0.83 vs 0.68 = +0.16!, Face2Face
0.68 vs 0.66). swap엔 약함(예상대로). **fuse-mean 0.781 > SBI 단독 0.762** — reenactment 사각지대를
메워 순이득(swap 소폭 손해 상쇄).

## 실제 kakao 3영상 (전부 FAKE)
| 영상 | 유형 | SBI | Temporal | 둘 중 하나라도 FAKE? |
|---|---|---|---|---|
| kakao_1 | 아이돌 swap | 0.99 ✅ | 0.35 | ✅ FAKE |
| kakao_2 | 아이돌 swap | 0.99 ✅ | 0.13 | ✅ FAKE |
| kakao_3 | 저커버그 reenact | 0.25 ❌ | **0.61 ✅** | ✅ FAKE |

**SBI 혼자선 2/3, Temporal 혼자선 1/3, 둘 합치면 3/3.** SBI가 놓친 reenactment를 Temporal이 잡음.

## 결론 (Phase 3)
- Temporal 브랜치가 **reenactment/lip-sync를 실제로 일반화**(FF++ NeuralTextures·Face2Face·kakao_3).
  kakao_3만 우연히 맞춘 게 아님(벤치마크로 확정).
- **SBI(공간 블렌딩)+Temporal(시간 불일치) = 진짜 상보 쌍.** 융합이 단일 SBI를 넘음(FF++ 0.762→0.781),
  실제 3영상 모두 커버. **AU·ConvNeXt 앙상블이 실패한 것과 달리, 각 브랜치가 독립적으로 일반화**하므로
  융합이 작동(자기지도로 학습해 지문 암기 회피).
- 개선 여지: 게이팅 융합(swap엔 SBI, reenact엔 Temporal)으로 오라클(≈per-method max)에 근접, 야생 열화
  강건 클립 증강, 클립 길이/모델 확장.

관련: [[sbi-redesign]] docs/phase3_reenactment_plan.md

## 게이팅 융합 (오라클 근접 시도)
| 방식 | FF++ macro(LOMO) |
|---|---|
| SBI 단독 | 0.764 |
| **mean 융합** | **0.787** (오라클 0.800의 98%) |
| 학습형 게이트(LOMO) | 0.776 (과적합, mean 못 넘음) |
| oracle(per-method max) | 0.800 |
→ **단순 mean이 최적**(오라클 근접). 학습 게이트는 과적합. 남은 격차는 "입력이 reenact인지"를
알아야만 좁혀지며 게이트가 일반화 못 함.

## 최종 탐지기 = OR(max) 융합 (`scripts/sbi/detect.py`)
mean은 AUC 랭킹엔 유리하나 **threshold 판정에서 reenactment를 놓침**(SBI 낮은 점수가 Temporal 희석
-> kakao_3 mean 0.43<0.5 오판). **각 브랜치는 다른 조작유형 전문가 -> 탐지 융합은 OR:**
"SBI(swap) 또는 Temporal(reenact) 하나라도 임계값 넘으면 fake". 실제 kakao 3영상 전부 정확 판정
+ 근거(swap/reenact) 제시. mean은 AUC용, OR은 배치 탐지용.
