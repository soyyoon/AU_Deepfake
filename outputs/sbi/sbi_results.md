# SBI Phase 1 결과 — 일반화 우선 단일 탐지기

EfficientNet-b4, real 크롭 29,878개(CelebDF real 699 + FF++ real 800 영상)로 **self-blend만
학습**(fake 데이터셋 미사용), 야생 열화 증강. best val_SBI_auc 0.9932(ep19). 비디오-레벨 평가.
평가 크롭: CelebDF/FF++는 학습과 동일 face-crop 파이프라인(매칭), WildDeepfake는 데이터셋 자체
tight 크롭(불일치 핸디캡).

## 도메인별 AUC (단일 모델, 라우팅 없음)
| 도메인 | **SBI** | ConvNeXt | AU |
|---|---|---|---|
| CelebDF test | **0.919** | 0.959(누수) / 0.922(heldout) | 0.587 |
| FF++ test (미지 기법) | **0.766** | 0.531 | 0.754 |
| WildDeepfake (야생) | **0.693** | 0.636 | 0.449 |
| **macro** | **0.793** | 0.709 | 0.597 |

**★ 핵심: SBI 단일 모델 macro 0.793 > 이전 앙상블의 오라클 상한 0.783.** fake 하나 안 보고,
라우팅 없이, 두 overfit 모델을 도메인마다 골라주는 오라클보다 낫다. 재설계가 옳았음을 실측 증명.

- CelebDF: SBI 0.919 ≈ ConvNeXt 정직치 0.922 (SBI는 CelebDF fake를 전혀 안 봤는데도 대등).
- FF++: SBI 0.766 ≈ AU 0.754, ConvNeXt 0.531 대비 **+0.23** (아티팩트 붕괴를 SBI가 해결).
- WildDeepfake: SBI 0.693 > 둘 다(ConvNeXt 0.636/AU 0.449).

**① 크롭매칭 확인(pad 실험):** tight 크롭에 마진 패딩 추가 시 오히려 하락(pad0.2→0.595,
0.35→0.549, 0.5→0.500). reflect-pad 인공경계가 방해. → **0.693이 진짜 야생 성능, 핸디캡 아님.**

## FF++ 기법별 (SBI)
| 기법 | SBI | ConvNeXt | AU |
|---|---|---|---|
| Deepfakes | **0.964** | 0.853 | 0.889 |
| FaceSwap | **0.828** | 0.337 | 0.532 |
| FaceShifter | 0.694 | 0.411 | 0.643 |
| NeuralTextures | 0.676 | 0.545 | **0.791** |
| Face2Face | 0.667 | 0.510 | **0.913** |

**패턴:** SBI는 **face-swap(블렌딩 경계 있음)에 강함**(Deepfakes 0.96, FaceSwap 0.83), 예상대로
**reenactment(Face2Face/NeuralTextures, 블렌딩 없음)엔 약함**(0.67). 바로 그 사각지대가 AU가
강한 곳(0.91/0.79) → **Phase 3 행동 브랜치의 근거 확보**.

## 결론
1. **재설계 성공.** SBI(기법-무관 블렌딩 학습)가 AU·ConvNeXt·그 앙상블 오라클을 모두 능가.
2. **야생에서도 최고**(0.693), 크롭 매칭하면 더 오를 여지.
3. **남은 사각지대 = reenactment.** 여기에만 behavior/temporal 브랜치를 근거 기반으로 추가(Phase 3).
4. 다음: (a) WildDeepfake 크롭 매칭 재평가, (b) FSBI 주파수 브랜치, (c) reenactment용 temporal 보강,
   (d) DFDC 미러로 4번째 도메인 확인, (e) 캘리브레이션/기권.

관련: [[ensemble-fair-comparison]] [[generalization-axes]] [[wild-dataset-sources]] docs/sbi_phase1_plan.md

## ② DFDC 추가 — 4번째 완전 독립 도메인 (네 모델)
DFDC train_sample 400영상(real77/fake323, 커뮤니티 미러 무폼), 로컬 크롭+AU feature(pyfeat 1회 통과).
| 모델 | DFDC AUC |
|---|---|
| **SBI** | **0.835** |
| ConvNeXt | 0.693 |
| AU | 0.544 |
| 앙상블(SBI+CN+AU rank평균) | 0.788 (AU가 끌어내림) |
| 앙상블(SBI+CN) | 0.837 (SBI와 동률, +0.002) |

## 4-도메인 최종 (SBI vs baseline)
| 도메인 | SBI | ConvNeXt | AU |
|---|---|---|---|
| CelebDF | 0.919 | 0.922 | 0.587 |
| FF++ | 0.766 | 0.531 | 0.754 |
| WildDeepfake | 0.693 | 0.636 | 0.449 |
| DFDC | 0.835 | 0.693 | 0.544 |
| **macro** | **0.803** | 0.696 | 0.583 |

**결론(확정):** SBI 단일 모델이 4개 도메인 중 3개서 최고(CelebDF는 ConvNeXt 정직치와 동률),
macro 0.803으로 압도. **앙상블은 여전히 무의미**(SBI+CN+AU가 SBI 단독보다 나쁨, SBI+CN은 동률).
= 재설계 최종 검증: 일반화 우선 단일 모델(SBI)이 정답, 앙상블/라우팅 불필요. AU는 reenactment
전용 보완으로만 여지(Phase 3).
