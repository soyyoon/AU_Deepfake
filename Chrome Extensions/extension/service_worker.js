const DEFAULT_API_URL = "http://127.0.0.1:8000";

chrome.runtime.onMessage.addListener((message, _sender, sendResponse) => {
  if (message.type === "ANALYZE_ACTIVE_TAB") {
    analyzeActiveTab(message.apiUrl)
      .then((result) => sendResponse({ ok: true, result }))
      .catch((error) => sendResponse({ ok: false, error: error.message }));
    return true;
  }

  if (message.type === "CLEAR_OVERLAY") {
    clearOverlay()
      .then(() => sendResponse({ ok: true }))
      .catch((error) => sendResponse({ ok: false, error: error.message }));
    return true;
  }

  return false;
});

async function analyzeActiveTab(apiUrl = DEFAULT_API_URL) {
  const tab = await getActiveTab();
  if (!tab.id) {
    throw new Error("활성 탭을 찾을 수 없습니다.");
  }

  await ensureContentScript(tab.id);
  const media = await sendToTab(tab.id, { type: "COLLECT_MEDIA" });
  if (!media?.target) {
    throw new Error("현재 화면에서 분석할 영상이나 이미지를 찾지 못했습니다.");
  }

  const image = await chrome.tabs.captureVisibleTab(tab.windowId, {
    format: "jpeg",
    quality: 82,
  });

  const result = await requestAnalysis(apiUrl, {
    image,
    media: media.target,
    candidates: media.candidates,
    source: "visible_tab",
    pageUrl: tab.url,
    capturedAt: new Date().toISOString(),
  });

  await chrome.storage.local.set({ lastResult: result });
  await sendToTab(tab.id, { type: "SHOW_RESULT", result, target: media.target });
  return result;
}

async function clearOverlay() {
  const tab = await getActiveTab();
  if (tab.id) {
    await ensureContentScript(tab.id);
    await sendToTab(tab.id, { type: "CLEAR_RESULT" });
  }
}

async function requestAnalysis(apiUrl, payload) {
  const response = await fetch(`${normalizeApiUrl(apiUrl)}/analyze/frame`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(payload),
  });

  const body = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(body.error || `API 요청 실패 (${response.status})`);
  }
  return body;
}

async function ensureContentScript(tabId) {
  try {
    await sendToTab(tabId, { type: "PING" });
  } catch {
    await chrome.scripting.executeScript({
      target: { tabId },
      files: ["content.js"],
    });
  }
}

function sendToTab(tabId, message) {
  return chrome.tabs.sendMessage(tabId, message);
}

async function getActiveTab() {
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  if (!tab) {
    throw new Error("활성 탭이 없습니다.");
  }
  return tab;
}

function normalizeApiUrl(value) {
  return String(value || DEFAULT_API_URL).replace(/\/+$/, "");
}
