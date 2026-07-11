(() => {
  "use strict";

  const SCRIPT_VERSION = chrome.runtime.getManifest().version;
  const LOAD_FLAG = `__deepfakeSignalYouTubeDemoLoaded_${SCRIPT_VERSION}`;
  if (globalThis[LOAD_FLAG]) {
    return;
  }
  globalThis[LOAD_FLAG] = true;

  const STORAGE_KEY = "youtubeDemoEnabled";
  const DEMO_QUERY_KEY = "deepfake_demo";
  const CARD_ATTRIBUTE = "data-deepfake-signal-demo-card";
  const ORIGINAL_HREF_ATTRIBUTE = "data-deepfake-signal-demo-original-href";
  const PLAYER_ID = "deepfake-signal-demo-player";
  const PLAYER_MOUNT_ATTRIBUTE = "data-deepfake-signal-demo-player-mount";
  const UNDERLAY_ATTRIBUTE = "data-deepfake-signal-demo-underlay";
  const DEMO = Object.freeze({
    title: "Deepfake Signal Demo · Chameleon Sample",
    channel: "Deepfake Signal",
    duration: "0:30",
    videoPath: "demo/233037_medium.mp4",
    thumbnailPath: "demo/233037_thumbnail.jpg",
  });

  const videoUrl = chrome.runtime.getURL(DEMO.videoPath);
  const thumbnailUrl = chrome.runtime.getURL(DEMO.thumbnailPath);

  let demoEnabled = false;
  let applyTimer = null;
  let observer = null;
  let maintenanceTimer = null;

  chrome.storage.onChanged.addListener((changes, areaName) => {
    if (areaName !== "local" || !changes[STORAGE_KEY]) {
      return;
    }

    setDemoEnabled(Boolean(changes[STORAGE_KEY].newValue), {
      reloadOnDisable: true,
    });
  });

  document.addEventListener("yt-navigate-finish", () => scheduleApply(40));
  document.addEventListener("yt-page-data-updated", () => scheduleApply(40));
  window.addEventListener("popstate", () => scheduleApply(40));

  document.addEventListener("click", handleCardActivation, true);
  document.addEventListener("auxclick", handleCardActivation, true);
  document.addEventListener("contextmenu", handleCardActivation, true);
  document.addEventListener("keydown", handleCardActivation, true);
  document.addEventListener("dragstart", handleCardActivation, true);
  document.addEventListener("pointerdown", handleDisabledCardInput, true);
  document.addEventListener("pointerup", handleDisabledCardInput, true);
  document.addEventListener("mousedown", handleDisabledCardInput, true);
  document.addEventListener("mouseup", handleDisabledCardInput, true);
  document.addEventListener("play", handleUnderlyingPlayback, true);

  void initialize();

  async function initialize() {
    let stored = {};
    try {
      stored = await chrome.storage.local.get(STORAGE_KEY);
    } catch {
      stored = {};
    }

    setDemoEnabled(Boolean(stored[STORAGE_KEY]), {
      reloadOnDisable: false,
    });
  }

  function setDemoEnabled(enabled, options = {}) {
    const wasEnabled = demoEnabled;
    demoEnabled = enabled;

    if (document.documentElement) {
      if (enabled) {
        document.documentElement.setAttribute("data-deepfake-signal-youtube-demo", SCRIPT_VERSION);
      } else {
        document.documentElement.removeAttribute("data-deepfake-signal-youtube-demo");
      }
    }

    if (enabled) {
      startMonitoring();
      scheduleApply(0);
      return;
    }

    stopMonitoring();
    removeDemoPlayer();
    clearCardMarkers();

    if (wasEnabled && options.reloadOnDisable) {
      window.location.reload();
    }
  }

  function startMonitoring() {
    if (!observer) {
      observer = new MutationObserver(() => scheduleApply(80));
      const root = document.documentElement || document;
      observer.observe(root, {
        childList: true,
        subtree: true,
        attributes: true,
        attributeFilter: ["hidden"],
      });
    }

    if (!maintenanceTimer) {
      maintenanceTimer = window.setInterval(() => scheduleApply(0), 1200);
    }
  }

  function stopMonitoring() {
    observer?.disconnect();
    observer = null;

    if (maintenanceTimer) {
      window.clearInterval(maintenanceTimer);
      maintenanceTimer = null;
    }

    if (applyTimer) {
      window.clearTimeout(applyTimer);
      applyTimer = null;
    }
  }

  function scheduleApply(delay = 80) {
    if (!demoEnabled || applyTimer) {
      return;
    }

    applyTimer = window.setTimeout(() => {
      applyTimer = null;
      applyDemo();
    }, delay);
  }

  function applyDemo() {
    if (!demoEnabled) {
      return;
    }

    if (document.documentElement) {
      document.documentElement.setAttribute("data-deepfake-signal-youtube-demo", SCRIPT_VERSION);
    }

    const watchRoot = getActiveWatchRoot();
    if (watchRoot && isDemoWatchUrl()) {
      applyWatchDemo(watchRoot);
      return;
    }

    removeDemoPlayer();

    const homeRoot = getActiveHomeRoot();
    if (homeRoot) {
      applyHomeDemo(homeRoot);
    }
  }

  function getActiveHomeRoot() {
    if (window.location.pathname !== "/") {
      return null;
    }

    return document.querySelector(
      'ytd-page-manager > ytd-browse[page-subtype="home"]:not([hidden])',
    ) || document.querySelector('ytd-browse[page-subtype="home"]:not([hidden])');
  }

  function getActiveWatchRoot() {
    if (window.location.pathname !== "/watch") {
      return null;
    }

    return document.querySelector(
      "ytd-page-manager > ytd-watch-flexy:not([hidden])",
    ) || document.querySelector("ytd-watch-flexy:not([hidden])");
  }

  function isDemoWatchUrl() {
    return new URLSearchParams(window.location.search).get(DEMO_QUERY_KEY) === "1";
  }

  function applyHomeDemo(homeRoot) {
    const existingActive = homeRoot.querySelector(`[${CARD_ATTRIBUTE}="active"]`);
    const existingDisabled = Array.from(
      homeRoot.querySelectorAll(`[${CARD_ATTRIBUTE}="disabled"]`),
    ).filter((card) => card.isConnected);
    const candidates = collectHomeCards(homeRoot);
    const activeCard = existingActive?.isConnected
      ? existingActive
      : candidates[0];
    const disabledCards = uniqueElements([
      ...existingDisabled,
      ...candidates,
    ]).filter((card) => card !== activeCard);

    if (!activeCard) {
      return;
    }

    const selectedCards = new Set([activeCard, ...disabledCards]);
    homeRoot.querySelectorAll(`[${CARD_ATTRIBUTE}]`).forEach((card) => {
      if (!selectedCards.has(card)) {
        clearCardMarker(card);
      }
    });

    applyActiveCard(activeCard);
    disabledCards.forEach(applyDisabledCard);
  }

  function uniqueElements(elements) {
    return Array.from(new Set(elements));
  }

  function collectHomeCards(homeRoot) {
    const grid = homeRoot.querySelector("ytd-rich-grid-renderer");
    const contents = grid?.querySelector(":scope > #contents");
    const directCards = Array.from(contents?.children || [])
      .filter((element) => element.matches("ytd-rich-item-renderer"))
      .filter(isVideoCard);

    if (directCards.length > 0) {
      return directCards;
    }

    const fallbackSelectors = [
      "ytd-rich-item-renderer",
      "ytd-rich-grid-media",
      "yt-lockup-view-model",
      "ytd-grid-video-renderer",
    ];

    for (const selector of fallbackSelectors) {
      const candidates = Array.from(homeRoot.querySelectorAll(selector))
        .filter(isVideoCard)
        .filter((card, index, list) => (
          list.findIndex((candidate) => candidate === card || candidate.contains(card)) === index
        ));

      if (candidates.length > 0) {
        return candidates;
      }
    }

    return [];
  }

  function isVideoCard(card) {
    if (!(card instanceof Element)) {
      return false;
    }

    if (card.closest("ytd-reel-shelf-renderer, ytd-ad-slot-renderer") || card.querySelector(
      "ytd-ad-slot-renderer, ytd-in-feed-ad-layout-renderer, ytd-display-ad-renderer",
    )) {
      return false;
    }

    return card.getAttribute(CARD_ATTRIBUTE) === "disabled" || getWatchAnchors(card).length > 0;
  }

  function getWatchAnchors(root) {
    return Array.from(root.querySelectorAll("a[href]"))
      .filter((anchor) => isWatchUrl(anchor.getAttribute("href") || anchor.href));
  }

  function isWatchUrl(value) {
    if (!value) {
      return false;
    }

    try {
      const url = new URL(value, window.location.origin);
      return url.origin === window.location.origin &&
        url.pathname === "/watch" &&
        url.searchParams.has("v");
    } catch {
      return false;
    }
  }

  function applyActiveCard(card) {
    restoreCardInteractivity(card);
    card.setAttribute(CARD_ATTRIBUTE, "active");
    card.removeAttribute("aria-disabled");

    const watchAnchors = getWatchAnchors(card);
    if (!card.dataset.deepfakeSignalDemoTargetUrl && watchAnchors[0]) {
      card.dataset.deepfakeSignalDemoTargetUrl = watchAnchors[0].href;
    }

    const targetUrl = buildDemoWatchUrl(card.dataset.deepfakeSignalDemoTargetUrl);
    if (!targetUrl) {
      return;
    }
    card.dataset.deepfakeSignalDemoUrl = targetUrl;

    updateCardThumbnail(card);
    updateCardTitle(card);
    updateCardChannel(card);
    updateCardDuration(card);

    getWatchAnchors(card).forEach((anchor) => {
      anchor.href = targetUrl;
      anchor.setAttribute("aria-disabled", "false");
      anchor.removeAttribute("tabindex");
      anchor.draggable = false;
    });
  }

  function applyDisabledCard(card) {
    card.setAttribute(CARD_ATTRIBUTE, "disabled");
    card.setAttribute("aria-disabled", "true");
    card.setAttribute("inert", "");
    card.inert = true;

    card.querySelectorAll("a").forEach((anchor) => {
      if (anchor.hasAttribute("href")) {
        anchor.setAttribute(ORIGINAL_HREF_ATTRIBUTE, anchor.getAttribute("href") || "");
      }
      anchor.removeAttribute("href");
      anchor.setAttribute("aria-disabled", "true");
    });
  }

  function buildDemoWatchUrl(value) {
    if (!value) {
      return "";
    }

    try {
      const url = new URL(value, window.location.origin);
      url.searchParams.set(DEMO_QUERY_KEY, "1");
      return url.href;
    } catch {
      return "";
    }
  }

  function updateCardThumbnail(card) {
    const thumbnailAnchor = getWatchAnchors(card)
      .find((anchor) => anchor.querySelector("img"));
    const images = thumbnailAnchor
      ? Array.from(thumbnailAnchor.querySelectorAll("img"))
      : Array.from(card.querySelectorAll("ytd-thumbnail img, yt-thumbnail-view-model img"));

    images.forEach((image) => {
      if (image.src !== thumbnailUrl) {
        image.src = thumbnailUrl;
      }
      image.removeAttribute("srcset");
      image.setAttribute("data-thumb", thumbnailUrl);
      image.setAttribute("alt", DEMO.title);
      image.loading = "eager";
    });
  }

  function updateCardTitle(card) {
    const titleAnchor = card.querySelector([
      'a.ytLockupMetadataViewModelTitle[href*="/watch"]',
      'a#video-title-link[href*="/watch"]',
      'a#video-title[href*="/watch"]',
      'h3 a[href*="/watch"]',
    ].join(", ")) || getWatchAnchors(card)
      .filter((anchor) => anchor.textContent.trim())
      .sort((first, second) => second.textContent.trim().length - first.textContent.trim().length)[0];

    if (!titleAnchor) {
      return;
    }

    const textTarget = titleAnchor.querySelector(
      ".ytAttributedStringHost, .yt-core-attributed-string, yt-formatted-string",
    ) || titleAnchor;
    if (textTarget.textContent.trim() !== DEMO.title) {
      textTarget.textContent = DEMO.title;
    }

    titleAnchor.setAttribute("title", DEMO.title);
    titleAnchor.setAttribute("aria-label", DEMO.title);
    titleAnchor.closest("h3")?.setAttribute("title", DEMO.title);
  }

  function updateCardChannel(card) {
    const channel = card.querySelector([
      "ytd-channel-name #text",
      "ytd-channel-name yt-formatted-string",
      "#channel-name a",
    ].join(", "));

    if (channel && channel.textContent.trim() !== DEMO.channel) {
      channel.textContent = DEMO.channel;
    }
  }

  function updateCardDuration(card) {
    const duration = card.querySelector([
      "ytd-thumbnail-overlay-time-status-renderer #text",
      "yt-thumbnail-badge-view-model .yt-badge-shape__text",
      ".badge-shape-wiz__text",
      ".yt-badge-shape__text",
    ].join(", "));

    if (duration && duration.textContent.trim() !== DEMO.duration) {
      duration.textContent = DEMO.duration;
    }
  }

  function handleCardActivation(event) {
    if (!demoEnabled || !getActiveHomeRoot()) {
      return;
    }

    const card = getMarkedCardFromEvent(event);
    if (!card) {
      return;
    }

    const state = card.getAttribute(CARD_ATTRIBUTE);
    if (state === "disabled") {
      blockEvent(event);
      return;
    }

    if (state !== "active") {
      return;
    }

    if (event.type === "contextmenu" || event.type === "dragstart") {
      return;
    }

    if (event instanceof KeyboardEvent && !["Enter", " "].includes(event.key)) {
      return;
    }

    const targetUrl = card.dataset.deepfakeSignalDemoUrl;
    if (!targetUrl) {
      return;
    }

    blockEvent(event);
    if ((event instanceof MouseEvent && event.button === 1) ||
      (event instanceof MouseEvent && (event.metaKey || event.ctrlKey))) {
      window.open(targetUrl, "_blank", "noopener");
      return;
    }

    window.location.assign(targetUrl);
  }

  function getMarkedCardFromEvent(event) {
    return event.composedPath().find((node) => (
      node instanceof Element && node.hasAttribute(CARD_ATTRIBUTE)
    )) || null;
  }

  function handleDisabledCardInput(event) {
    if (!demoEnabled || !getActiveHomeRoot()) {
      return;
    }

    const card = getMarkedCardFromEvent(event);
    if (card?.getAttribute(CARD_ATTRIBUTE) === "disabled") {
      blockEvent(event);
    }
  }

  function blockEvent(event) {
    event.preventDefault();
    event.stopPropagation();
    event.stopImmediatePropagation();
  }

  function handleUnderlyingPlayback(event) {
    if (!demoEnabled || !isDemoWatchUrl() || !(event.target instanceof HTMLVideoElement)) {
      return;
    }

    if (event.target.closest(`#${PLAYER_ID}`)) {
      return;
    }

    event.target.muted = true;
    event.target.pause();
  }

  function applyWatchDemo(watchRoot) {
    document.documentElement?.setAttribute("data-deepfake-signal-demo-watch", "");
    watchRoot.setAttribute("data-deepfake-signal-demo-watch-root", "");

    const playerHost = watchRoot.querySelector("ytd-player#ytd-player");
    const playerContainer = playerHost?.querySelector(":scope > #container") || playerHost;
    const moviePlayer = playerContainer?.querySelector(":scope > #movie_player") ||
      playerContainer?.querySelector("#movie_player");
    const mount = moviePlayer || playerContainer || playerHost;
    if (!mount) {
      return;
    }
    mount.setAttribute(PLAYER_MOUNT_ATTRIBUTE, "");

    let layer = document.getElementById(PLAYER_ID);
    if (!layer) {
      layer = createDemoPlayer();
    }
    if (layer.parentElement !== mount) {
      mount.appendChild(layer);
    }

    const demoVideo = layer.querySelector("video");
    pauseUnderlyingVideos(watchRoot, demoVideo);
    applyWatchMetadata(watchRoot);

    if (demoVideo?.paused && layer.dataset.userPaused !== "true") {
      void demoVideo.play().catch(() => {
        layer.classList.add("is-paused");
      });
    }
  }

  function createDemoPlayer() {
    const layer = document.createElement("section");
    layer.id = PLAYER_ID;
    layer.setAttribute("aria-label", "Deepfake Signal 데모 영상 플레이어");
    layer.tabIndex = 0;

    const video = document.createElement("video");
    video.src = videoUrl;
    video.poster = thumbnailUrl;
    video.autoplay = true;
    video.loop = true;
    video.muted = true;
    video.playsInline = true;
    video.preload = "auto";
    video.setAttribute("aria-label", DEMO.title);

    const centerPlay = createControlButton("재생", "▶", "deepfake-signal-demo-center-play");
    const controls = document.createElement("div");
    controls.className = "deepfake-signal-demo-controls";

    const progress = document.createElement("div");
    progress.className = "deepfake-signal-demo-progress";
    progress.setAttribute("role", "slider");
    progress.setAttribute("aria-label", "재생 위치");
    progress.tabIndex = 0;
    const progressFill = document.createElement("span");
    progress.append(progressFill);

    const controlRow = document.createElement("div");
    controlRow.className = "deepfake-signal-demo-control-row";
    const playButton = createControlButton("재생", "▶", "deepfake-signal-demo-play");
    const muteButton = createControlButton("음소거 해제", "🔇", "deepfake-signal-demo-mute");
    const time = document.createElement("span");
    time.className = "deepfake-signal-demo-time";
    time.textContent = `0:00 / ${DEMO.duration}`;
    const spacer = document.createElement("span");
    spacer.className = "deepfake-signal-demo-control-spacer";
    const fullscreenButton = createControlButton("전체 화면", "⛶", "deepfake-signal-demo-fullscreen");

    controlRow.append(playButton, muteButton, time, spacer, fullscreenButton);
    controls.append(progress, controlRow);
    layer.append(video, centerPlay, controls);

    const togglePlayback = () => {
      if (video.paused) {
        layer.dataset.userPaused = "false";
        void video.play();
      } else {
        layer.dataset.userPaused = "true";
        video.pause();
      }
    };

    const updatePlaybackUi = () => {
      const duration = Number.isFinite(video.duration) ? video.duration : 0;
      const ratio = duration > 0 ? Math.min(1, video.currentTime / duration) : 0;
      progressFill.style.width = `${ratio * 100}%`;
      time.textContent = `${formatTime(video.currentTime)} / ${duration > 0 ? formatTime(duration) : DEMO.duration}`;
      const paused = video.paused;
      layer.classList.toggle("is-paused", paused);
      playButton.textContent = paused ? "▶" : "❚❚";
      playButton.setAttribute("aria-label", paused ? "재생" : "일시정지");
      centerPlay.setAttribute("aria-label", paused ? "재생" : "일시정지");
      muteButton.textContent = video.muted ? "🔇" : "🔊";
      muteButton.setAttribute("aria-label", video.muted ? "음소거 해제" : "음소거");
    };

    video.addEventListener("click", togglePlayback);
    video.addEventListener("play", updatePlaybackUi);
    video.addEventListener("pause", updatePlaybackUi);
    video.addEventListener("timeupdate", updatePlaybackUi);
    video.addEventListener("loadedmetadata", updatePlaybackUi);
    centerPlay.addEventListener("click", togglePlayback);
    playButton.addEventListener("click", togglePlayback);
    muteButton.addEventListener("click", () => {
      video.muted = !video.muted;
      updatePlaybackUi();
    });
    fullscreenButton.addEventListener("click", () => {
      if (document.fullscreenElement === layer) {
        void document.exitFullscreen();
      } else {
        void layer.requestFullscreen();
      }
    });
    progress.addEventListener("click", (event) => {
      if (!Number.isFinite(video.duration) || video.duration <= 0) {
        return;
      }
      const rect = progress.getBoundingClientRect();
      const ratio = Math.min(1, Math.max(0, (event.clientX - rect.left) / rect.width));
      video.currentTime = ratio * video.duration;
    });
    layer.addEventListener("keydown", (event) => {
      if (event.key === " " || event.key === "k") {
        event.preventDefault();
        togglePlayback();
      }
    });
    layer.addEventListener("dblclick", () => {
      void layer.requestFullscreen();
    });

    updatePlaybackUi();
    void video.play().catch(() => {
      layer.classList.add("is-paused");
    });
    return layer;
  }

  function createControlButton(label, text, className) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = className;
    button.setAttribute("aria-label", label);
    button.textContent = text;
    return button;
  }

  function pauseUnderlyingVideos(watchRoot, demoVideo) {
    watchRoot.querySelectorAll("video").forEach((video) => {
      if (video === demoVideo) {
        return;
      }

      if (!video.hasAttribute(UNDERLAY_ATTRIBUTE)) {
        video.dataset.deepfakeSignalDemoOriginalMuted = String(video.muted);
      }
      video.setAttribute(UNDERLAY_ATTRIBUTE, "");
      video.muted = true;
      video.pause();
    });
  }

  function restoreUnderlyingVideos() {
    document.querySelectorAll(`video[${UNDERLAY_ATTRIBUTE}]`).forEach((video) => {
      video.removeAttribute(UNDERLAY_ATTRIBUTE);
      video.muted = video.dataset.deepfakeSignalDemoOriginalMuted === "true";
      delete video.dataset.deepfakeSignalDemoOriginalMuted;
    });
  }

  function applyWatchMetadata(watchRoot) {
    const title = watchRoot.querySelector([
      "ytd-watch-metadata h1 yt-formatted-string",
      "ytd-watch-metadata h1 .yt-core-attributed-string",
      "#title h1 yt-formatted-string",
    ].join(", "));
    if (title && title.textContent.trim() !== DEMO.title) {
      title.textContent = DEMO.title;
    }

    const channel = watchRoot.querySelector([
      "ytd-channel-name #text",
      "ytd-channel-name yt-formatted-string",
      "#owner #channel-name a",
    ].join(", "));
    if (channel && channel.textContent.trim() !== DEMO.channel) {
      channel.textContent = DEMO.channel;
    }

    const documentTitle = `${DEMO.title} - YouTube`;
    if (document.title !== documentTitle) {
      document.title = documentTitle;
    }
  }

  function removeDemoPlayer() {
    document.getElementById(PLAYER_ID)?.remove();
    document.querySelectorAll(`[${PLAYER_MOUNT_ATTRIBUTE}]`).forEach((mount) => {
      mount.removeAttribute(PLAYER_MOUNT_ATTRIBUTE);
    });
    document.documentElement?.removeAttribute("data-deepfake-signal-demo-watch");
    document.querySelectorAll("[data-deepfake-signal-demo-watch-root]").forEach((root) => {
      root.removeAttribute("data-deepfake-signal-demo-watch-root");
    });
    restoreUnderlyingVideos();
  }

  function clearCardMarkers() {
    document.querySelectorAll(`[${CARD_ATTRIBUTE}]`).forEach((card) => {
      clearCardMarker(card);
    });
  }

  function clearCardMarker(card) {
    restoreCardInteractivity(card);
    card.removeAttribute(CARD_ATTRIBUTE);
    card.removeAttribute("aria-disabled");
  }

  function restoreCardInteractivity(card) {
    card.inert = false;
    card.removeAttribute("inert");

    card.querySelectorAll(`[${ORIGINAL_HREF_ATTRIBUTE}]`).forEach((anchor) => {
      anchor.setAttribute("href", anchor.getAttribute(ORIGINAL_HREF_ATTRIBUTE) || "");
      anchor.removeAttribute(ORIGINAL_HREF_ATTRIBUTE);
      anchor.removeAttribute("aria-disabled");
    });
  }

  function formatTime(value) {
    if (!Number.isFinite(value) || value < 0) {
      return "0:00";
    }

    const totalSeconds = Math.floor(value);
    const minutes = Math.floor(totalSeconds / 60);
    const seconds = String(totalSeconds % 60).padStart(2, "0");
    return `${minutes}:${seconds}`;
  }
})();
