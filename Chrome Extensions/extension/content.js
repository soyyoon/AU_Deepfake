(() => {
  if (window.__deepfakeSignalLoaded) {
    return;
  }
  window.__deepfakeSignalLoaded = true;

  const OVERLAY_ID = "deepfake-signal-overlay";

  chrome.runtime.onMessage.addListener((message, _sender, sendResponse) => {
    if (message.type === "PING") {
      sendResponse({ ok: true });
      return true;
    }

    if (message.type === "COLLECT_MEDIA") {
      sendResponse(collectMedia());
      return true;
    }

    if (message.type === "SHOW_RESULT") {
      showResult(message.result, message.target);
      sendResponse({ ok: true });
      return true;
    }

    if (message.type === "CLEAR_RESULT") {
      clearOverlay();
      sendResponse({ ok: true });
      return true;
    }

    return false;
  });

  function collectMedia() {
    const candidates = [...document.querySelectorAll("video, img")]
      .map((element) => toCandidate(element))
      .filter(Boolean)
      .sort((a, b) => b.area - a.area);

    return {
      target: candidates[0] || null,
      candidates: candidates.slice(0, 5),
    };
  }

  function toCandidate(element) {
    const rect = element.getBoundingClientRect();
    const width = Math.round(rect.width);
    const height = Math.round(rect.height);
    const left = Math.round(rect.left);
    const top = Math.round(rect.top);

    if (width < 96 || height < 96) {
      return null;
    }

    if (rect.bottom <= 0 || rect.right <= 0 || rect.top >= window.innerHeight || rect.left >= window.innerWidth) {
      return null;
    }

    return {
      tag: element.tagName.toLowerCase(),
      left,
      top,
      width,
      height,
      area: width * height,
      viewportWidth: window.innerWidth,
      viewportHeight: window.innerHeight,
      currentTime: element instanceof HTMLVideoElement ? element.currentTime : null,
      paused: element instanceof HTMLVideoElement ? element.paused : null,
      src: safeSource(element),
    };
  }

  function safeSource(element) {
    const raw = element.currentSrc || element.src || "";
    try {
      const url = new URL(raw, window.location.href);
      return `${url.origin}${url.pathname}`;
    } catch {
      return "";
    }
  }

  function showResult(result, target) {
    clearOverlay();

    const overlay = document.createElement("div");
    overlay.id = OVERLAY_ID;
    overlay.style.position = "fixed";
    overlay.style.left = `${Math.max(8, target.left + 12)}px`;
    overlay.style.top = `${Math.max(8, target.top + 12)}px`;
    overlay.style.zIndex = "2147483647";
    overlay.style.display = "grid";
    overlay.style.gap = "4px";
    overlay.style.maxWidth = "260px";
    overlay.style.padding = "10px 12px";
    overlay.style.border = "1px solid rgba(255,255,255,0.25)";
    overlay.style.borderRadius = "8px";
    overlay.style.background = overlayColor(result.label);
    overlay.style.boxShadow = "0 10px 30px rgba(0,0,0,0.25)";
    overlay.style.color = "#fff";
    overlay.style.font = "13px/1.35 system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif";

    const title = document.createElement("strong");
    title.textContent = labelText(result.label);
    title.style.fontSize = "14px";

    const detail = document.createElement("span");
    detail.textContent = `Score ${Math.round((result.final_score || 0) * 100)} · Uncertainty ${Math.round((result.uncertainty || 0) * 100)}`;

    overlay.append(title, detail);
    document.documentElement.appendChild(overlay);
  }

  function clearOverlay() {
    document.getElementById(OVERLAY_ID)?.remove();
  }

  function labelText(label) {
    return {
      high_suspicion: "조작 의심 높음",
      medium_suspicion: "조작 의심 중간",
      low_suspicion: "조작 의심 낮음",
      uncertain: "판정 불확실",
    }[label] || "분석 결과";
  }

  function overlayColor(label) {
    return {
      high_suspicion: "#bd1e3c",
      medium_suspicion: "#9a5b00",
      low_suspicion: "#11613b",
      uncertain: "#394152",
    }[label] || "#394152";
  }
})();
