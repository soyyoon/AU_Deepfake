const scanBtn = document.querySelector("#scanBtn");
const clearBtn = document.querySelector("#clearBtn");
const candidateCount = document.querySelector("#candidateCount");
const statusTitle = document.querySelector("#statusTitle");
const statusText = document.querySelector("#statusText");
const videoCount = document.querySelector("#videoCount");
const imageCount = document.querySelector("#imageCount");
const largestType = document.querySelector("#largestType");
const candidateList = document.querySelector("#candidateList");

const INITIAL_STATE = {
  title: "대기 중",
  text: "현재 탭에서 보이는 영상과 이미지를 확인합니다.",
};

const DEFAULT_API_URL = "http://127.0.0.1:8000";
const DEEPFAKE_THRESHOLD = 0.95;
const UNCERTAIN_THRESHOLD = 0.75;
const VIDEO_SEQUENCE_FRAME_COUNT = 3;
const VIDEO_SEQUENCE_FRAME_DELAY_MS = 450;
const SIGNALS = {
  analyzing: {
    label: "Analyzing",
    color: "#ffd6a6",
    tint: "rgba(255, 214, 166, 0.12)",
    shadow: "rgba(255, 214, 166, 0.38)",
  },
  deepfake: {
    label: "Deepfake_Signal",
    color: "#ff9a86",
    tint: "rgba(255, 214, 166, 0.08)",
    shadow: "rgba(255, 179, 153, 0.36)",
  },
  uncertain: {
    label: "Review_Needed",
    color: "#ffd6a6",
    tint: "rgba(255, 214, 166, 0.12)",
    shadow: "rgba(255, 214, 166, 0.34)",
  },
  safe: {
    label: "Safe_Signal",
    color: "#4bb8fa",
    tint: "rgba(75, 184, 250, 0.10)",
    shadow: "rgba(75, 184, 250, 0.34)",
  },
  noface: {
    label: "No_Face",
    color: "#697386",
    tint: "rgba(105, 115, 134, 0.08)",
    shadow: "rgba(105, 115, 134, 0.22)",
  },
  error: {
    label: "Signal_Error",
    color: "#697386",
    tint: "rgba(105, 115, 134, 0.10)",
    shadow: "rgba(105, 115, 134, 0.26)",
  },
};

let activeTabId = null;
let activeWindowId = null;
let activeHoverToken = 0;
const analysisCache = new Map();

renderEmptyState();

scanBtn.addEventListener("click", async () => {
  setLoading(true);
  setStatus("탐색 중", "현재 화면에 보이는 미디어 후보를 확인하고 있습니다.");

  try {
    const result = hasExtensionApi() ? await scanActiveTab() : buildPreviewResult();
    renderScanResult(result);
  } catch (error) {
    activeTabId = null;
    activeWindowId = null;
    analysisCache.clear();
    setStatus("탐색 실패", getFriendlyError(error));
    renderEmptyState({ keepStatus: true });
  } finally {
    setLoading(false);
  }
});

clearBtn.addEventListener("click", async () => {
  await clearPageHighlight({ removeMarkers: true });
  activeTabId = null;
  activeWindowId = null;
  activeHoverToken += 1;
  analysisCache.clear();
  setStatus(INITIAL_STATE.title, INITIAL_STATE.text);
  renderEmptyState();
});

window.addEventListener("pagehide", () => {
  void clearPageHighlight({ removeMarkers: false });
});

async function scanActiveTab() {
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });

  if (!tab?.id) {
    throw new Error("활성 탭을 찾지 못했습니다.");
  }

  const [injection] = await chrome.scripting.executeScript({
    target: { tabId: tab.id },
    func: scanVisibleMedia,
    args: [Date.now(), getPopupOcclusionEstimate()],
  });

  if (!injection?.result) {
    throw new Error("탐색 결과를 가져오지 못했습니다.");
  }

  activeTabId = tab.id;
  activeWindowId = tab.windowId;
  activeHoverToken += 1;
  analysisCache.clear();
  return {
    ...injection.result,
    tabId: tab.id,
  };
}

function scanVisibleMedia(scanId, popupOcclusion) {
  const maxCandidates = 18;
  const markerAttribute = "data-deepfake-signal-candidate-id";
  const viewport = {
    width: window.innerWidth,
    height: window.innerHeight,
  };
  const popupRect = getPopupRect(viewport, popupOcclusion);
  const roots = getSearchRoots();

  deepQueryAll(`[${markerAttribute}]`, roots).forEach((element) => {
    element.removeAttribute(markerAttribute);
  });

  removeHighlightOverlay();

  const rawMediaCandidates = deepQueryAll("video, img", roots)
    .map((element, index) => toCandidateRecord(element, index, "media"))
    .filter(Boolean);
  const backgroundOffset = rawMediaCandidates.length;
  const rawBackgroundCandidates = collectBackgroundCandidates(backgroundOffset);
  const uniqueCandidates = dedupeCandidates([...rawMediaCandidates, ...rawBackgroundCandidates])
    .sort(compareCandidates);
  const candidates = uniqueCandidates.slice(0, maxCandidates);

  candidates.forEach(({ candidate, element }) => {
    element.setAttribute(markerAttribute, candidate.candidateId);
  });

  const visibleCandidates = candidates.map(({ candidate }) => candidate);
  const visibleVideoCount = visibleCandidates.filter((candidate) => candidate.type === "video").length;
  const visibleImageCount = visibleCandidates.filter((candidate) => candidate.type === "image").length;

  return {
    source: "active-tab",
    totalElements: rawMediaCandidates.length + rawBackgroundCandidates.length,
    totalCandidates: uniqueCandidates.length,
    visibleCandidates: visibleCandidates.length,
    counts: {
      video: visibleVideoCount,
      image: visibleImageCount,
    },
    candidates: visibleCandidates,
  };

  function toCandidateRecord(element, index, sourceKind) {
    const candidate = describeMediaElement(element, index, sourceKind);
    return candidate ? { candidate, element } : null;
  }

  function describeMediaElement(element, index, sourceKind) {
    const style = window.getComputedStyle(element);
    const rect = element.getBoundingClientRect();
    const visibleWidth = Math.max(
      0,
      Math.min(rect.right, viewport.width) - Math.max(rect.left, 0),
    );
    const visibleHeight = Math.max(
      0,
      Math.min(rect.bottom, viewport.height) - Math.max(rect.top, 0),
    );
    const pageVisibleRect = {
      left: Math.max(rect.left, 0),
      top: Math.max(rect.top, 0),
      right: Math.min(rect.right, viewport.width),
      bottom: Math.min(rect.bottom, viewport.height),
    };
    const pageVisibleArea = Math.round(visibleWidth * visibleHeight);
    const popupCoveredArea = popupRect ? getIntersectionArea(pageVisibleRect, popupRect) : 0;
    const uncoveredArea = Math.max(0, pageVisibleArea - popupCoveredArea);
    const area = Math.round(rect.width * rect.height);

    if (
      style.display === "none" ||
      style.visibility === "hidden" ||
      Number(style.opacity) === 0 ||
      rect.width < 48 ||
      rect.height < 48 ||
      pageVisibleArea < 4096
    ) {
      return null;
    }

    const tagName = element.tagName.toLowerCase();
    const isVideo = sourceKind === "video" || tagName === "video";
    const mediaWidth = isVideo ? element.videoWidth : element.naturalWidth;
    const mediaHeight = isVideo ? element.videoHeight : element.naturalHeight;
    const source = getElementSource(element, style, sourceKind);
    const rectInfo = {
      left: Math.round(pageVisibleRect.left),
      top: Math.round(pageVisibleRect.top),
      right: Math.round(pageVisibleRect.right),
      bottom: Math.round(pageVisibleRect.bottom),
      width: Math.round(rect.width),
      height: Math.round(rect.height),
    };

    return {
      candidateId: `deepfake-signal-${scanId}-${index}`,
      type: isVideo ? "video" : "image",
      sourceKind,
      viewportWidth: viewport.width,
      viewportHeight: viewport.height,
      displayWidth: Math.round(rect.width),
      displayHeight: Math.round(rect.height),
      mediaWidth: mediaWidth || null,
      mediaHeight: mediaHeight || null,
      visibleArea: uncoveredArea,
      pageVisibleArea,
      visibleRatio: area > 0 ? Math.min(1, pageVisibleArea / area) : 0,
      uncoveredRatio: area > 0 ? Math.min(1, uncoveredArea / area) : 0,
      popupCoveredRatio: pageVisibleArea > 0 ? Math.min(1, popupCoveredArea / pageVisibleArea) : 0,
      sourceHost: getSourceHost(source),
      sourceKey: getSourceKey(source),
      rect: rectInfo,
      isPlaying: isVideo ? !element.paused && !element.ended : null,
      readyState: isVideo ? element.readyState : null,
    };
  }

  function collectBackgroundCandidates(startIndex) {
    const elements = deepQueryAll("a, div, span, ytd-thumbnail, yt-img-shadow, yt-image, [style]", roots);
    const candidates = [];

    elements.forEach((element, index) => {
      const rect = element.getBoundingClientRect();
      if (rect.width < 96 || rect.height < 54) {
        return;
      }

      const style = window.getComputedStyle(element);
      if (!hasBackgroundImage(style.backgroundImage)) {
        return;
      }

      const record = toCandidateRecord(element, startIndex + index, "background");
      if (record) {
        candidates.push(record);
      }
    });

    return candidates;
  }

  function dedupeCandidates(records) {
    const sorted = [...records].sort((first, second) => getCandidateQuality(second.candidate) - getCandidateQuality(first.candidate));
    const unique = [];

    sorted.forEach((record) => {
      const duplicate = unique.some((existing) => areDuplicateCandidates(record.candidate, existing.candidate));
      if (!duplicate) {
        unique.push(record);
      }
    });

    return unique;
  }

  function compareCandidates(first, second) {
    return getCandidateRank(second.candidate) - getCandidateRank(first.candidate);
  }

  function getCandidateQuality(candidate) {
    const kindBonus = {
      video: 500000,
      media: 200000,
      background: 0,
    }[candidate.sourceKind] || 0;
    return candidate.pageVisibleArea + kindBonus;
  }

  function getCandidateRank(candidate) {
    const typeBonus = candidate.type === "video" ? 500000 : 0;
    const sourceBonus = candidate.sourceKind === "media" ? 50000 : 0;
    return candidate.visibleArea + Math.round(candidate.pageVisibleArea * 0.35) + typeBonus + sourceBonus;
  }

  function areDuplicateCandidates(first, second) {
    const overlap = getOverlapRatio(first.rect, second.rect);
    const sameSource = first.sourceKey && first.sourceKey === second.sourceKey;
    const centerDistance = getCenterDistance(first.rect, second.rect);
    const similarSize = Math.abs(first.rect.width - second.rect.width) < 24 &&
      Math.abs(first.rect.height - second.rect.height) < 24;

    return overlap > 0.82 || (sameSource && overlap > 0.25) || (centerDistance < 18 && similarSize);
  }

  function getOverlapRatio(first, second) {
    const intersection = getIntersectionArea(first, second);
    const firstArea = Math.max(0, first.right - first.left) * Math.max(0, first.bottom - first.top);
    const secondArea = Math.max(0, second.right - second.left) * Math.max(0, second.bottom - second.top);
    const smallerArea = Math.min(firstArea, secondArea);

    return smallerArea > 0 ? intersection / smallerArea : 0;
  }

  function getCenterDistance(first, second) {
    const firstX = first.left + (first.right - first.left) / 2;
    const firstY = first.top + (first.bottom - first.top) / 2;
    const secondX = second.left + (second.right - second.left) / 2;
    const secondY = second.top + (second.bottom - second.top) / 2;

    return Math.hypot(firstX - secondX, firstY - secondY);
  }

  function getElementSource(element, style, sourceKind) {
    if (sourceKind === "background") {
      return getBackgroundImageUrl(style.backgroundImage);
    }

    if (element instanceof HTMLVideoElement) {
      return element.currentSrc || element.poster || element.src || "";
    }

    return element.currentSrc || element.src || "";
  }

  function hasBackgroundImage(value) {
    return Boolean(getBackgroundImageUrl(value));
  }

  function getBackgroundImageUrl(value) {
    const match = String(value || "").match(/url\(["']?(.+?)["']?\)/);
    return match?.[1] || "";
  }

  function getSourceHost(source) {
    if (!source) {
      return "inline/local";
    }

    try {
      return new URL(source, window.location.href).hostname || "inline/local";
    } catch {
      return "inline/local";
    }
  }

  function getSourceKey(source) {
    if (!source) {
      return "";
    }

    try {
      const url = new URL(source, window.location.href);
      return `${url.origin}${url.pathname}`;
    } catch {
      return String(source);
    }
  }

  function removeHighlightOverlay() {
    document.getElementById("deepfake-signal-highlight-overlay")?.remove();
  }

  function getPopupRect(currentViewport, occlusion) {
    if (!occlusion?.width || !occlusion?.height) {
      return null;
    }

    const width = Math.min(currentViewport.width, Math.max(0, occlusion.width));
    const height = Math.min(currentViewport.height, Math.max(0, occlusion.height));

    return {
      left: Math.max(0, currentViewport.width - width),
      top: 0,
      right: currentViewport.width,
      bottom: height,
    };
  }

  function getIntersectionArea(first, second) {
    const width = Math.max(0, Math.min(first.right, second.right) - Math.max(first.left, second.left));
    const height = Math.max(0, Math.min(first.bottom, second.bottom) - Math.max(first.top, second.top));
    return Math.round(width * height);
  }

  function deepQueryAll(selector, searchRoots) {
    return searchRoots.flatMap((root) => Array.from(root.querySelectorAll(selector)));
  }

  function getSearchRoots() {
    const searchRoots = [document];
    const queue = [document];

    while (queue.length > 0) {
      const root = queue.shift();
      root.querySelectorAll("*").forEach((element) => {
        if (element.shadowRoot) {
          searchRoots.push(element.shadowRoot);
          queue.push(element.shadowRoot);
        }
      });
    }

    return searchRoots;
  }
}

function highlightCandidateInPage(candidateId, signal = {}, analysis = {}) {
  const markerAttribute = "data-deepfake-signal-candidate-id";
  const target = deepQueryAll(`[${markerAttribute}]`)
    .find((element) => element.getAttribute(markerAttribute) === candidateId);

  document.getElementById("deepfake-signal-highlight-overlay")?.remove();

  if (!target) {
    return { ok: false };
  }

  const rect = target.getBoundingClientRect();
  const visibleLeft = Math.max(0, rect.left);
  const visibleTop = Math.max(0, rect.top);
  const visibleRight = Math.min(window.innerWidth, rect.right);
  const visibleBottom = Math.min(window.innerHeight, rect.bottom);
  const width = Math.max(0, visibleRight - visibleLeft);
  const height = Math.max(0, visibleBottom - visibleTop);

  if (width < 8 || height < 8) {
    return { ok: false };
  }

  const color = signal.color || "#ff9a86";
  const label = signal.label || "Deepfake_Signal";
  const tint = signal.tint || "rgba(255, 214, 166, 0.08)";
  const shadow = signal.shadow || "rgba(255, 179, 153, 0.36)";
  const overlay = document.createElement("div");
  overlay.id = "deepfake-signal-highlight-overlay";
  overlay.setAttribute("aria-hidden", "true");
  Object.assign(overlay.style, {
    position: "fixed",
    left: `${visibleLeft}px`,
    top: `${visibleTop}px`,
    width: `${width}px`,
    height: `${height}px`,
    zIndex: "2147483647",
    pointerEvents: "none",
    boxSizing: "border-box",
    border: `4px solid ${color}`,
    borderRadius: "8px",
    boxShadow: `0 0 0 8px ${shadow}, 0 14px 36px rgba(70, 40, 28, 0.28)`,
    background: tint,
  });

  const badge = document.createElement("div");
  badge.textContent = label;
  Object.assign(badge.style, {
    position: "absolute",
    left: "10px",
    top: "10px",
    padding: "6px 8px",
    borderRadius: "8px",
    background: color,
    color: "#201614",
    font: "700 12px/1 system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
    boxShadow: "0 6px 16px rgba(70, 40, 28, 0.24)",
  });

  overlay.append(badge);

  const faces = Array.isArray(analysis?.faces)
    ? analysis.faces
    : analysis?.face
      ? [analysis.face]
      : [];
  faces.forEach((face, index) => {
    const faceOverlay = buildFaceOverlay(face, width, height, index);
    if (faceOverlay) {
      overlay.append(faceOverlay);
    }
  });

  document.documentElement.append(overlay);

  return {
    ok: true,
    rect: {
      left: Math.round(visibleLeft),
      top: Math.round(visibleTop),
      width: Math.round(width),
      height: Math.round(height),
    },
  };

  function buildFaceOverlay(face, overlayWidth, overlayHeight, index) {
    if (!face?.rect || !face.imageWidth || !face.imageHeight) {
      return null;
    }

    const scaleX = overlayWidth / face.imageWidth;
    const scaleY = overlayHeight / face.imageHeight;
    const faceLeft = clamp(face.rect.left * scaleX, 0, overlayWidth - 1);
    const faceTop = clamp(face.rect.top * scaleY, 0, overlayHeight - 1);
    const faceWidth = clamp(face.rect.width * scaleX, 4, overlayWidth - faceLeft);
    const faceHeight = clamp(face.rect.height * scaleY, 4, overlayHeight - faceTop);

    if (faceWidth < 4 || faceHeight < 4) {
      return null;
    }

    const box = document.createElement("div");
    const boxColor = getFaceBoxColor(face);
    Object.assign(box.style, {
      position: "absolute",
      left: `${faceLeft}px`,
      top: `${faceTop}px`,
      width: `${faceWidth}px`,
      height: `${faceHeight}px`,
      boxSizing: "border-box",
      border: `3px solid ${boxColor}`,
      borderRadius: "6px",
      boxShadow: "0 0 0 2px rgba(23, 27, 33, 0.72), 0 8px 18px rgba(23, 27, 33, 0.26)",
      background: "rgba(255, 240, 190, 0.06)",
    });

    const label = document.createElement("div");
    const scoreText = Number.isFinite(face.score) ? ` · DF ${Math.round(face.score * 100)}%` : "";
    const frameText = face.frameIndex ? ` · f${face.frameIndex + 1}` : "";
    label.textContent = `Face ${index + 1}${frameText}${scoreText}`;
    Object.assign(label.style, {
      position: "absolute",
      left: "0",
      top: "-26px",
      padding: "5px 7px",
      borderRadius: "6px",
      background: boxColor,
      color: "#201614",
      font: "700 11px/1 system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
      boxShadow: "0 4px 12px rgba(23, 27, 33, 0.22)",
      whiteSpace: "nowrap",
    });

    box.append(label);
    return box;
  }

  function getFaceBoxColor(face) {
    const label = String(face.label || "").toLowerCase();
    if (label === "high_suspicion" || Number(face.score) >= 0.95) {
      return "#ff9a86";
    }
    if (label === "uncertain" || Number(face.score) >= 0.75) {
      return "#ffd6a6";
    }
    return "#4bb8fa";
  }

  function clamp(value, min, max) {
    return Math.min(max, Math.max(min, value));
  }

  function deepQueryAll(selector) {
    return getSearchRoots().flatMap((root) => Array.from(root.querySelectorAll(selector)));
  }

  function getSearchRoots() {
    const roots = [document];
    const queue = [document];

    while (queue.length > 0) {
      const root = queue.shift();
      root.querySelectorAll("*").forEach((element) => {
        if (element.shadowRoot) {
          roots.push(element.shadowRoot);
          queue.push(element.shadowRoot);
        }
      });
    }

    return roots;
  }
}

function clearCandidateHighlightInPage(options = {}) {
  const markerAttribute = "data-deepfake-signal-candidate-id";
  document.getElementById("deepfake-signal-highlight-overlay")?.remove();

  if (options.removeMarkers) {
    deepQueryAll(`[${markerAttribute}]`).forEach((element) => {
      element.removeAttribute(markerAttribute);
    });
  }

  return { ok: true };

  function deepQueryAll(selector) {
    return getSearchRoots().flatMap((root) => Array.from(root.querySelectorAll(selector)));
  }

  function getSearchRoots() {
    const roots = [document];
    const queue = [document];

    while (queue.length > 0) {
      const root = queue.shift();
      root.querySelectorAll("*").forEach((element) => {
        if (element.shadowRoot) {
          roots.push(element.shadowRoot);
          queue.push(element.shadowRoot);
        }
      });
    }

    return roots;
  }
}

function renderScanResult(result) {
  const candidates = result.candidates || [];
  const largest = candidates[0];

  candidateCount.textContent = String(result.visibleCandidates ?? candidates.length);
  videoCount.textContent = String(result.counts?.video ?? 0);
  imageCount.textContent = String(result.counts?.image ?? 0);
  largestType.textContent = largest ? formatType(largest.type) : "--";

  if (candidates.length === 0) {
    setStatus("후보 없음", "현재 화면에서 분석할 만한 크기의 영상이나 이미지를 찾지 못했습니다.");
    renderCandidateList([]);
    return;
  }

  const previewPrefix = result.source === "preview" ? "미리보기 데이터입니다. " : "";
  setStatus(
    `${candidates.length}개 후보 발견`,
    `${previewPrefix}후보 위에 마우스를 올리면 모델 API 분석 후 신호 색을 표시합니다.`,
  );
  renderCandidateList(candidates);
}

function renderCandidateList(candidates) {
  candidateList.replaceChildren();

  if (candidates.length === 0) {
    const emptyItem = document.createElement("li");
    emptyItem.className = "empty";
    emptyItem.textContent = "표시할 미디어 후보가 없습니다.";
    candidateList.append(emptyItem);
    return;
  }

  candidates.forEach((candidate, index) => {
    const item = document.createElement("li");
    item.className = "candidate-card";
    item.tabIndex = 0;

    const title = document.createElement("strong");
    title.textContent = `${index + 1}. ${formatType(candidate.type)} · ${candidate.displayWidth}x${candidate.displayHeight}`;

    const source = document.createElement("span");
    source.textContent = buildSourceText(candidate);

    const detail = document.createElement("span");
    detail.textContent = buildCandidateDetail(candidate);

    const analysisResult = document.createElement("span");
    analysisResult.className = "candidate-result";
    analysisResult.textContent = "분석 전";

    item.addEventListener("mouseenter", () => {
      setActiveCandidateCard(item);
      void analyzeAndHighlightCandidate(candidate, item);
    });
    item.addEventListener("focus", () => {
      setActiveCandidateCard(item);
      void analyzeAndHighlightCandidate(candidate, item);
    });

    item.append(title, source, detail, analysisResult);
    candidateList.append(item);
  });
}

async function analyzeAndHighlightCandidate(candidate, item) {
  if (!hasExtensionApi() || !activeTabId || activeWindowId == null || !candidate.candidateId) {
    return;
  }

  const hoverToken = ++activeHoverToken;
  const cached = analysisCache.get(candidate.candidateId);

  if (cached) {
    setStatus(cached.signal.label, buildAnalysisSummary(cached.result));
    setCandidateAnalysisResult(item, cached.signal, cached.result);
    await highlightPageCandidate(candidate, cached.signal, cached.result);
    return;
  }

  try {
    const frameCount = candidate.type === "video" ? VIDEO_SEQUENCE_FRAME_COUNT : 1;
    const frameText = frameCount > 1 ? `${frameCount}개 프레임을 샘플링하고 있습니다.` : "현재 후보 영역을 모델 API로 분석하고 있습니다.";
    setStatus("API 분석 중", frameText);
    await clearPageHighlight({ removeMarkers: false });
    const frames = await captureCandidateFrames(candidate, frameCount, hoverToken);

    if (hoverToken !== activeHoverToken || frames.length === 0) {
      return;
    }

    await highlightPageCandidate(candidate, SIGNALS.analyzing);
    const result = frames.length > 1
      ? await requestSequenceAnalysis(candidate, frames)
      : await requestFrameAnalysis(candidate, frames[0]);

    if (hoverToken !== activeHoverToken) {
      return;
    }

    const signal = signalFromAnalysis(result);
    analysisCache.set(candidate.candidateId, { result, signal });
    setStatus(signal.label, buildAnalysisSummary(result));
    setCandidateAnalysisResult(item, signal, result);
    await highlightPageCandidate(candidate, signal, result);
  } catch (error) {
    if (hoverToken !== activeHoverToken) {
      return;
    }

    setStatus("API 분석 실패", getFriendlyError(error));
    await highlightPageCandidate(candidate, SIGNALS.error);
  }
}

async function highlightPageCandidate(candidate, signal = SIGNALS.deepfake, result = null) {
  if (!hasExtensionApi() || !activeTabId || !candidate.candidateId) {
    return;
  }

  try {
    const [injection] = await chrome.scripting.executeScript({
      target: { tabId: activeTabId },
      func: highlightCandidateInPage,
      args: [candidate.candidateId, signal, buildOverlayAnalysis(result)],
    });

    if (!injection?.result?.ok) {
      setStatus("하이라이트 실패", "후보가 화면에서 사라졌거나 현재 페이지에서 좌표를 다시 찾지 못했습니다.");
    }
  } catch {
    setStatus("하이라이트 실패", "이 페이지에서는 후보 테두리를 표시할 수 없습니다.");
  }
}

function buildOverlayAnalysis(result) {
  const artifact = result?.metadata?.artifact || {};
  const primaryFace = artifact.primary_face || null;
  const rawFaces = Array.isArray(artifact.display_faces) && artifact.display_faces.length > 0
    ? artifact.display_faces
    : Array.isArray(artifact.face_samples)
      ? artifact.face_samples
      : [];
  const faces = rawFaces
    .filter((face) => face?.detected && face.face_rect && face.image_width && face.image_height)
    .slice(0, 8)
    .map((face) => ({
      frameIndex: Number(face.frame_index || 0),
      faceIndex: Number(face.face_index || 0),
      imageWidth: Number(face.image_width),
      imageHeight: Number(face.image_height),
      score: Number(face.score),
      label: String(face.label || ""),
      isPrimary: Boolean(
        primaryFace &&
        Number(primaryFace.frame_index || 0) === Number(face.frame_index || 0) &&
        Number(primaryFace.face_index || 0) === Number(face.face_index || 0)
      ),
      rect: {
        left: Number(face.face_rect.left),
        top: Number(face.face_rect.top),
        width: Number(face.face_rect.width),
        height: Number(face.face_rect.height),
      },
      cropRect: face.crop_rect || null,
    }));

  return { faces };
}

async function requestFrameAnalysis(candidate, image) {
  const response = await fetch(`${DEFAULT_API_URL}/analyze/frame`, {
    method: "POST",
    headers: {
      "content-type": "application/json",
    },
    body: JSON.stringify({
      image,
      media: buildAnalysisMediaPayload(candidate),
      source: "visible_tab_candidate",
    }),
  });

  const body = await response.json().catch(() => ({}));

  if (!response.ok) {
    throw new Error(body.error || `API 요청 실패 (${response.status})`);
  }

  return body;
}

async function requestSequenceAnalysis(candidate, frames) {
  const response = await fetch(`${DEFAULT_API_URL}/analyze/sequence`, {
    method: "POST",
    headers: {
      "content-type": "application/json",
    },
    body: JSON.stringify({
      frames,
      media: buildAnalysisMediaPayload(candidate),
    }),
  });

  const body = await response.json().catch(() => ({}));

  if (!response.ok) {
    throw new Error(body.error || `API 요청 실패 (${response.status})`);
  }

  return body;
}

function buildAnalysisMediaPayload(candidate) {
  return {
    type: candidate.type,
    sourceKind: candidate.sourceKind,
    width: candidate.displayWidth,
    height: candidate.displayHeight,
    visibleArea: candidate.visibleArea,
    pageVisibleArea: candidate.pageVisibleArea,
    sourceHost: candidate.sourceHost,
    thresholds: {
      deepfake: DEEPFAKE_THRESHOLD,
      uncertain: UNCERTAIN_THRESHOLD,
    },
  };
}

async function captureCandidateFrames(candidate, frameCount, hoverToken) {
  const frames = [];

  for (let index = 0; index < frameCount; index += 1) {
    if (hoverToken !== activeHoverToken) {
      return [];
    }

    const image = await chrome.tabs.captureVisibleTab(activeWindowId, {
      format: "jpeg",
      quality: 78,
    });
    frames.push(await cropCandidateImage(image, candidate));

    if (index < frameCount - 1) {
      await sleep(VIDEO_SEQUENCE_FRAME_DELAY_MS);
    }
  }

  return frames;
}

async function cropCandidateImage(imageDataUrl, candidate) {
  const response = await fetch(imageDataUrl);
  const blob = await response.blob();
  const bitmap = await createImageBitmap(blob);
  const viewportWidth = Math.max(1, candidate.viewportWidth || bitmap.width);
  const viewportHeight = Math.max(1, candidate.viewportHeight || bitmap.height);
  const scaleX = bitmap.width / viewportWidth;
  const scaleY = bitmap.height / viewportHeight;
  const rect = candidate.rect || {
    left: 0,
    top: 0,
    right: candidate.displayWidth || viewportWidth,
    bottom: candidate.displayHeight || viewportHeight,
  };
  const sx = clamp(Math.round(rect.left * scaleX), 0, bitmap.width - 1);
  const sy = clamp(Math.round(rect.top * scaleY), 0, bitmap.height - 1);
  const sw = clamp(Math.round((rect.right - rect.left) * scaleX), 1, bitmap.width - sx);
  const sh = clamp(Math.round((rect.bottom - rect.top) * scaleY), 1, bitmap.height - sy);
  const maxSide = 768;
  const resizeRatio = Math.min(1, maxSide / Math.max(sw, sh));
  const canvas = document.createElement("canvas");
  canvas.width = Math.max(1, Math.round(sw * resizeRatio));
  canvas.height = Math.max(1, Math.round(sh * resizeRatio));

  const context = canvas.getContext("2d");
  if (!context) {
    throw new Error("이미지 crop canvas를 만들 수 없습니다.");
  }

  context.drawImage(bitmap, sx, sy, sw, sh, 0, 0, canvas.width, canvas.height);
  bitmap.close?.();
  return canvas.toDataURL("image/jpeg", 0.82);
}

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

function sleep(ms) {
  return new Promise((resolve) => window.setTimeout(resolve, ms));
}

function signalFromAnalysis(result) {
  const score = Number(result?.final_score);
  const label = String(result?.label || "").toLowerCase();
  const safeLabels = new Set(["low_suspicion", "safe", "real", "authentic", "not_deepfake"]);
  const noFaceLabels = new Set(["no_face", "no_face_detected", "unsupported_input"]);
  const uncertainLabels = new Set(["medium_suspicion", "uncertain", "review_needed", "ambiguous"]);
  const suspiciousLabels = new Set([
    "high_suspicion",
    "deepfake",
    "fake",
    "manipulated",
  ]);

  if (noFaceLabels.has(label)) {
    return SIGNALS.noface;
  }

  if (safeLabels.has(label) || (Number.isFinite(score) && score < UNCERTAIN_THRESHOLD)) {
    return SIGNALS.safe;
  }

  if (suspiciousLabels.has(label) || (Number.isFinite(score) && score >= DEEPFAKE_THRESHOLD)) {
    return SIGNALS.deepfake;
  }

  if (uncertainLabels.has(label) || (Number.isFinite(score) && score >= UNCERTAIN_THRESHOLD)) {
    return SIGNALS.uncertain;
  }

  return SIGNALS.error;
}

function buildAnalysisSummary(result) {
  const score = Number(result?.final_score);
  const scoreText = Number.isFinite(score) ? `딥페이크 방향 점수 ${Math.round(score * 100)}%` : "모델 분석 생략";
  const label = result?.label || "unknown";
  const artifact = result?.metadata?.artifact || {};
  const face = artifact.primary_face || (artifact.face_samples || []).find((sample) => sample?.detected);
  const faceText = artifact.face_detected === true
    ? `얼굴 ${artifact.analyzed_faces || 1}개 분석 · ${artifact.analyzed_frames || 1}/${artifact.frame_count || 1}프레임`
    : artifact.face_detected === false
      ? "얼굴 없음"
      : "";
  const bboxText = face?.face_rect
    ? `최고점 bbox ${face.face_rect.left},${face.face_rect.top},${face.face_rect.width}x${face.face_rect.height}`
    : "";
  const framePolicyText = artifact.analyzed_faces > 1 ? "프레임 판단: 최고 얼굴 점수 기준" : "";
  const details = [scoreText, label, faceText, framePolicyText, bboxText].filter(Boolean);
  return details.join(" · ");
}

function setCandidateAnalysisResult(item, signal, result) {
  const resultElement = item?.querySelector(".candidate-result");
  if (!resultElement) {
    return;
  }

  resultElement.textContent = buildAnalysisSummary(result);
  resultElement.style.color = signal.color;
  resultElement.style.fontWeight = "720";
}

async function clearPageHighlight(options = {}) {
  if (!hasExtensionApi() || !activeTabId) {
    return;
  }

  try {
    await chrome.scripting.executeScript({
      target: { tabId: activeTabId },
      func: clearCandidateHighlightInPage,
      args: [options],
    });
  } catch {
    // The popup can outlive page access briefly; ignore cleanup failures.
  }
}

function setActiveCandidateCard(activeCard) {
  candidateList.querySelectorAll(".candidate-card.is-active").forEach((card) => {
    card.classList.remove("is-active");
  });
  activeCard.classList.add("is-active");
}

function buildCandidateDetail(candidate) {
  const visiblePercent = Math.round((candidate.visibleRatio || 0) * 100);
  const popupCoveredPercent = Math.round((candidate.popupCoveredRatio || 0) * 100);
  const popupNote = popupCoveredPercent > 0 ? ` · popup 가림 ${popupCoveredPercent}%` : "";

  if (candidate.type === "video") {
    const playback = candidate.isPlaying ? "재생 중" : "정지 또는 대기";
    return `${playback} · 화면 노출 ${visiblePercent}%${popupNote} · readyState ${candidate.readyState}`;
  }

  const naturalSize = candidate.mediaWidth && candidate.mediaHeight
    ? `${candidate.mediaWidth}x${candidate.mediaHeight}`
    : "크기 정보 없음";
  return `원본 크기 ${naturalSize} · 화면 노출 ${visiblePercent}%${popupNote}`;
}

function buildSourceText(candidate) {
  const kindLabel = {
    media: candidate.type === "video" ? "video" : "img",
    background: "background",
  }[candidate.sourceKind] || "media";

  return `출처 도메인: ${candidate.sourceHost} · 요소: ${kindLabel}`;
}

function renderEmptyState(options = {}) {
  candidateCount.textContent = "--";
  videoCount.textContent = "--";
  imageCount.textContent = "--";
  largestType.textContent = "--";
  renderCandidateList([]);

  if (!options.keepStatus) {
    setStatus(INITIAL_STATE.title, INITIAL_STATE.text);
  }
}

function buildPreviewResult() {
  activeTabId = null;

  return {
    source: "preview",
    totalElements: 3,
    visibleCandidates: 3,
    counts: {
      video: 1,
      image: 2,
    },
    candidates: [
      {
        candidateId: "preview-video-1",
        type: "video",
        viewportWidth: 1440,
        viewportHeight: 900,
        displayWidth: 1280,
        displayHeight: 720,
        mediaWidth: 1920,
        mediaHeight: 1080,
        visibleArea: 921600,
        pageVisibleArea: 921600,
        visibleRatio: 0.94,
        uncoveredRatio: 0.94,
        popupCoveredRatio: 0,
        sourceHost: "example-video.test",
        sourceKind: "media",
        rect: {
          left: 80,
          top: 120,
          right: 1360,
          bottom: 840,
          width: 1280,
          height: 720,
        },
        isPlaying: true,
        readyState: 4,
      },
      {
        candidateId: "preview-image-1",
        type: "image",
        viewportWidth: 1440,
        viewportHeight: 900,
        displayWidth: 640,
        displayHeight: 420,
        mediaWidth: 1280,
        mediaHeight: 840,
        visibleArea: 268800,
        pageVisibleArea: 268800,
        visibleRatio: 1,
        uncoveredRatio: 1,
        popupCoveredRatio: 0,
        sourceHost: "example-image.test",
        sourceKind: "media",
        rect: {
          left: 120,
          top: 160,
          right: 760,
          bottom: 580,
          width: 640,
          height: 420,
        },
        isPlaying: null,
        readyState: null,
      },
      {
        candidateId: "preview-image-2",
        type: "image",
        viewportWidth: 1440,
        viewportHeight: 900,
        displayWidth: 320,
        displayHeight: 240,
        mediaWidth: 640,
        mediaHeight: 480,
        visibleArea: 76800,
        pageVisibleArea: 76800,
        visibleRatio: 0.88,
        uncoveredRatio: 0.88,
        popupCoveredRatio: 0,
        sourceHost: "cdn.example.test",
        sourceKind: "media",
        rect: {
          left: 220,
          top: 180,
          right: 540,
          bottom: 420,
          width: 320,
          height: 240,
        },
        isPlaying: null,
        readyState: null,
      },
    ],
  };
}

function getPopupOcclusionEstimate() {
  const width = Math.ceil(Math.max(
    document.documentElement.clientWidth || 0,
    document.body?.clientWidth || 0,
    window.innerWidth || 0,
  ));
  const height = Math.ceil(Math.max(
    document.documentElement.scrollHeight || 0,
    document.body?.scrollHeight || 0,
    window.innerHeight || 0,
  ));

  return {
    width,
    height,
  };
}

function hasExtensionApi() {
  return Boolean(globalThis.chrome?.tabs?.query && globalThis.chrome?.scripting?.executeScript);
}

function getFriendlyError(error) {
  const message = error?.message || "알 수 없는 오류가 발생했습니다.";

  if (
    message.includes("Cannot access") ||
    message.includes("chrome://") ||
    message.includes("extensions gallery")
  ) {
    return "이 페이지에서는 확장 프로그램이 미디어를 탐색할 수 없습니다.";
  }

  return message;
}

function formatType(type) {
  return type === "video" ? "Video" : "Image";
}

function setStatus(title, text) {
  statusTitle.textContent = title;
  statusText.textContent = text;
}

function setLoading(isLoading) {
  scanBtn.disabled = isLoading;
  scanBtn.textContent = isLoading ? "탐색 중" : "분석 시작";
}
