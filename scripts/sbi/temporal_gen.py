"""Temporal self-consistency pseudo-fake: 입 영역 시간축 스플라이스 (reenactment/lip-sync 모사).

실제 연속 클립에서 각 프레임의 입을 '다른 시점' 입으로 교체 -> 머리 움직임과 입이 시간적으로
불일치(=lip-sync 아티팩트). 공간 블렌딩은 최소, 시간 불일치가 핵심. SBI의 temporal 버전.
"""
import cv2
import numpy as np


def mouth_mask(lm, shape, dilate=1.5, blur=7):
    pts = lm[48:68].astype(np.float32)
    c = pts.mean(0)
    pts = (c + (pts - c) * dilate).astype(np.int32)
    hull = cv2.convexHull(pts)
    m = np.zeros(shape[:2], np.float32)
    cv2.fillConvexPoly(m, hull, 1.0)
    m = cv2.GaussianBlur(m, (0, 0), blur)
    mx = m.max()
    return (m / mx if mx > 0 else m)[..., None]


def temporal_splice(frames, lms, shift=None):
    """각 프레임 입을 (t+shift) 시점 입으로 교체. 반환: fake 클립."""
    N = len(frames)
    if shift is None:
        shift = np.random.choice([-4, -3, -2, 2, 3, 4])
    fake = frames.copy()
    for t in range(N):
        src = frames[(t + shift) % N]
        m = mouth_mask(lms[t], frames[t].shape)
        fake[t] = np.clip(m * src + (1 - m) * frames[t], 0, 255).astype(np.uint8)
    return fake


def temporal_jitter(frames, lms, amp=0.02):
    """입 영역에 프레임별 랜덤 미세 워프 -> 시간 jitter."""
    N = len(frames); H, W = frames[0].shape[:2]
    fake = frames.copy()
    for t in range(N):
        dx, dy = np.random.uniform(-amp, amp, 2) * [W, H]
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        warped = cv2.warpAffine(frames[t], M, (W, H), borderMode=cv2.BORDER_REFLECT)
        m = mouth_mask(lms[t], frames[t].shape)
        fake[t] = np.clip(m * warped + (1 - m) * frames[t], 0, 255).astype(np.uint8)
    return fake


def make_fake(frames, lms):
    return temporal_splice(frames, lms) if np.random.rand() < 0.6 else temporal_jitter(frames, lms)


if __name__ == "__main__":
    d = np.load("/tmp/real_clip.npz")
    frames, lms = d["frames"], d["lms"]
    fake = temporal_splice(frames, lms, shift=3)
    # 시각화: 연속 6프레임 real vs fake vs diff
    idx = range(2, 14, 2)
    rows = []
    for tag, clip in [("REAL", frames), ("FAKE", fake)]:
        row = np.concatenate([cv2.resize(clip[i], (140, 140)) for i in idx], axis=1)
        rows.append(row)
    diff = np.concatenate([np.clip(cv2.absdiff(frames[i], fake[i]) * 3, 0, 255) for i in idx], axis=1)
    diff = cv2.resize(diff, (rows[0].shape[1], 140))
    grid = np.concatenate([rows[0], rows[1], diff], axis=0)
    cv2.imwrite("/tmp/temporal_check.png", cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
    print("saved /tmp/temporal_check.png (행: REAL / FAKE / diff×3, 연속 프레임)")
    print("mouth diff mean:", round(float(cv2.absdiff(frames, fake).mean()), 3))
