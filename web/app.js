const MAX_FILE_SIZE = 5 * 1024 * 1024;
const ALLOWED_TYPES = new Set(["image/png", "image/jpeg", "image/webp"]);
const ALLOWED_EXTENSIONS = /\.(png|jpe?g|webp)$/i;
const EXAMPLES = [
  { src: "/examples/example1.jpg", label: "Example 1" },
  { src: "/examples/example2.jpg", label: "Example 2" },
  { src: "/examples/example3.jpg", label: "Example 3" },
];

// Keep all DOM lookups together so the UI dependencies are easy to audit.
const elements = {
  uploadState: document.querySelector("#upload-state"),
  uploadArea: document.querySelector("#upload-area"),
  fileInput: document.querySelector("#file-input"),
  uploadError: document.querySelector("#upload-error"),
  exampleList: document.querySelector("#example-list"),
  resultState: document.querySelector("#result-state"),
  previewImage: document.querySelector("#preview-image"),
  loadingState: document.querySelector("#loading-state"),
  loadingText: document.querySelector("#loading-state span:last-child"),
  caption: document.querySelector("#caption"),
  resultError: document.querySelector("#result-error"),
  heatmapControls: document.querySelector("#heatmap-controls"),
  heatmapToggle: document.querySelector("#heatmap-toggle"),
  heatmapFigure: document.querySelector("#heatmap-figure"),
  downloadButton: document.querySelector("#download-button"),
  heatmapGrid: document.querySelector("#heatmap-grid"),
  retryButton: document.querySelector("#retry-button"),
  chooseButton: document.querySelector("#choose-button"),
  repoLink: document.querySelector("#repo-link"),
};

let lastResult = null;
let selectedFile = null;
let previewUrl = null;
let activeController = null;
let activeTimeoutId = null;
let activeSlowTimeoutId = null;
let requestTimedOut = false;
let heatmapMode = false;

function showError(message, target) {
  target.textContent = message;
  target.hidden = false;
}

function clearError(target) {
  target.textContent = "";
  target.hidden = true;
}

function formatCaption(text) {
  const caption = String(text || "").trim();
  if (!caption) {
    return "No caption was generated.";
  }
  const capitalized = caption.charAt(0).toUpperCase() + caption.slice(1);
  return /[.!?]$/.test(capitalized) ? capitalized : `${capitalized}.`;
}

function isAllowedFile(file) {
  return ALLOWED_TYPES.has(file.type) || (
    !file.type && ALLOWED_EXTENSIONS.test(file.name)
  );
}

function setLoading(loading) {
  elements.loadingState.hidden = !loading;
  elements.retryButton.hidden = loading;
  elements.loadingText.textContent = loading
    ? "Generating caption..."
    : "";
  if (loading) {
    elements.caption.textContent = "";
    clearError(elements.resultError);
  }
}

function hasHeatmaps() {
  return Boolean(
    lastResult
    && Array.isArray(lastResult.words)
    && lastResult.words.length > 0
    && Array.isArray(lastResult.heatmaps)
    && lastResult.heatmaps.length === lastResult.words.length,
  );
}

function renderGrid() {
  elements.heatmapGrid.replaceChildren();
  if (!hasHeatmaps()) {
    return;
  }
  lastResult.words.forEach((word, index) => {
    const item = document.createElement("figure");
    const image = document.createElement("img");
    const label = document.createElement("span");
    item.className = "grid-item";
    image.src = lastResult.heatmaps[index];
    image.alt = `Attention heatmap for the word "${word}"`;
    label.textContent = `${index + 1}. ${word}`;
    item.append(image, label);
    elements.heatmapGrid.append(item);
  });
}

function setHeatmapMode(on) {
  heatmapMode = on && hasHeatmaps();
  elements.heatmapToggle.textContent = heatmapMode ? "Hide heatmap" : "Show heatmap";
  elements.heatmapToggle.setAttribute("aria-pressed", String(heatmapMode));
  elements.heatmapFigure.hidden = !heatmapMode;
}

function friendlyRequestError(error) {
  if (error.name === "AbortError" && requestTimedOut) {
    return "This is taking too long. Please try again.";
  }
  if (navigator.onLine === false) {
    return "You are offline.";
  }
  if (error.status === 413) {
    return "That image is too large (max 5 MB).";
  }
  if (error.status === 415) {
    return "Please upload a PNG, JPEG or WEBP image.";
  }
  if (error.status === 400) {
    return "That file could not be read as an image.";
  }
  if (error.status === 429) {
    return "Too many requests, please wait a moment.";
  }
  if (error.status >= 500) {
    return "Something went wrong on the server. Please try again.";
  }
  return "Cannot reach the server. Check your connection.";
}

function renderExamples() {
  EXAMPLES.forEach(({ src, label }) => {
    const button = document.createElement("button");
    const image = document.createElement("img");
    button.type = "button";
    button.className = "example-button";
    button.setAttribute("aria-label", label);
    button.addEventListener("click", () => loadExample(src, label, button));
    image.src = src;
    image.alt = label;
    button.append(image);
    elements.exampleList.append(button);
  });
}

async function loadExample(src, label, button) {
  clearError(elements.uploadError);
  button.disabled = true;
  try {
    const response = await fetch(src);
    if (!response.ok) {
      throw new Error("Example image could not be loaded.");
    }
    const blob = await response.blob();
    const filename = src.split("/").pop() || `${label.toLowerCase().replace(/\s+/g, "-")}.jpg`;
    handleFile(new File([blob], filename, { type: "image/jpeg" }));
  } catch {
    showError("We could not load that example image. Please try again.", elements.uploadError);
  } finally {
    button.disabled = false;
  }
}

function warmUpServer() {
  const apiBase = window.APP_CONFIG?.API_BASE || "";
  fetch(`${apiBase}/api/health`).catch(() => {});
}

function updateRepositoryLink() {
  const repoUrl = window.APP_CONFIG?.REPO_URL || "";
  if (repoUrl) {
    elements.repoLink.href = repoUrl;
    elements.repoLink.hidden = false;
  }
}

function resetHeatmapState() {
  heatmapMode = false;
  elements.heatmapControls.hidden = true;
  elements.heatmapGrid.replaceChildren();
  elements.heatmapFigure.hidden = true;
  elements.heatmapToggle.textContent = "Show heatmap";
  elements.heatmapToggle.setAttribute("aria-pressed", "false");
}

async function downloadHeatmaps() {
  if (!hasHeatmaps()) {
    return;
  }
  try {
    const images = await Promise.all(lastResult.heatmaps.map((src) => new Promise((resolve, reject) => {
      const image = new Image();
      image.onload = () => resolve(image);
      image.onerror = () => reject(new Error("Heatmap image failed to load."));
      image.src = src;
    })));
    const columns = 5;
    const tileWidth = 240;
    const imageHeight = 180;
    const labelHeight = 42;
    const rows = Math.ceil(images.length / columns);
    const canvas = document.createElement("canvas");
    canvas.width = columns * tileWidth;
    canvas.height = rows * (imageHeight + labelHeight);
    const context = canvas.getContext("2d");
    context.fillStyle = "#ffffff";
    context.fillRect(0, 0, canvas.width, canvas.height);
    context.textAlign = "center";
    context.textBaseline = "top";
    context.font = "16px sans-serif";
    images.forEach((image, index) => {
      const x = (index % columns) * tileWidth;
      const y = Math.floor(index / columns) * (imageHeight + labelHeight);
      const scale = Math.min(tileWidth / image.width, imageHeight / image.height);
      const width = image.width * scale;
      const height = image.height * scale;
      context.drawImage(image, x + (tileWidth - width) / 2, y, width, height);
      context.fillStyle = "#111111";
      context.fillText(`${index + 1}. ${lastResult.words[index]}`, x + tileWidth / 2, y + imageHeight + 10);
    });
    const blob = await new Promise((resolve, reject) => {
      canvas.toBlob((result) => {
        if (result) {
          resolve(result);
        } else {
          reject(new Error("Heatmap export failed."));
        }
      }, "image/png");
    });
    if (blob) {
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.download = "heatmaps.png";
      link.click();
      URL.revokeObjectURL(url);
    }
  } catch {
    showError("We could not prepare the heatmap download. Please try again.", elements.resultError);
  }
}

function resetPage() {
  if (activeController) {
    activeController.abort();
    activeController = null;
  }
  if (activeTimeoutId !== null) {
    window.clearTimeout(activeTimeoutId);
    activeTimeoutId = null;
  }
  if (activeSlowTimeoutId !== null) {
    window.clearTimeout(activeSlowTimeoutId);
    activeSlowTimeoutId = null;
  }
  requestTimedOut = false;
  if (previewUrl) {
    URL.revokeObjectURL(previewUrl);
    previewUrl = null;
  }
  selectedFile = null;
  lastResult = null;
  resetHeatmapState();
  elements.fileInput.value = "";
  elements.previewImage.removeAttribute("src");
  elements.uploadState.hidden = false;
  elements.resultState.hidden = true;
  elements.loadingState.hidden = true;
  elements.loadingText.textContent = "";
  elements.caption.textContent = "";
  elements.retryButton.hidden = true;
  clearError(elements.uploadError);
  clearError(elements.resultError);
  elements.uploadArea.focus();
}

function readError(response) {
  const error = new Error(`Request failed with status ${response.status}.`);
  error.status = response.status;
  return error;
}

async function generateCaption(file) {
  if (activeController) {
    activeController.abort();
  }
  if (activeTimeoutId !== null) {
    window.clearTimeout(activeTimeoutId);
  }
  if (activeSlowTimeoutId !== null) {
    window.clearTimeout(activeSlowTimeoutId);
  }
  const controller = new AbortController();
  activeController = controller;
  const timeoutId = window.setTimeout(() => {
    requestTimedOut = true;
    controller.abort();
  }, 60000);
  activeTimeoutId = timeoutId;
  requestTimedOut = false;
  const slowTimeoutId = window.setTimeout(() => {
    if (activeController === controller) {
      elements.loadingText.textContent = "The server is waking up, this can take up to a minute...";
    }
  }, 8000);
  activeSlowTimeoutId = slowTimeoutId;
  const formData = new FormData();
  formData.append("file", file);

  try {
    const apiBase = window.APP_CONFIG?.API_BASE || "";
    const response = await fetch(`${apiBase}/api/caption`, {
      method: "POST",
      body: formData,
      signal: controller.signal,
    });
    if (!response.ok) {
      throw await readError(response);
    }
    const result = await response.json();
    if (activeController !== controller) {
      return;
    }
    lastResult = result;
    elements.caption.textContent = formatCaption(lastResult.caption);
    elements.retryButton.hidden = true;
    resetHeatmapState();
    if (hasHeatmaps()) {
      renderGrid();
      elements.heatmapControls.hidden = false;
    }
  } catch (error) {
    if (activeController !== controller) {
      return;
    }
    showError(friendlyRequestError(error), elements.resultError);
    elements.retryButton.hidden = false;
  } finally {
    window.clearTimeout(timeoutId);
    window.clearTimeout(slowTimeoutId);
    if (activeController === controller) {
      activeController = null;
      activeTimeoutId = null;
      activeSlowTimeoutId = null;
      requestTimedOut = false;
      elements.loadingState.hidden = true;
      elements.loadingText.textContent = "";
    }
  }
}

function handleFile(file) {
  clearError(elements.uploadError);
  if (!file) {
    return;
  }
  if (!isAllowedFile(file)) {
    showError("Please upload a PNG, JPEG or WEBP image.", elements.uploadError);
    return;
  }
  if (file.size > MAX_FILE_SIZE) {
    showError(
      "That image is too large (max 5 MB).",
      elements.uploadError,
    );
    return;
  }

  selectedFile = file;
  resetHeatmapState();
  if (previewUrl) {
    URL.revokeObjectURL(previewUrl);
  }
  previewUrl = URL.createObjectURL(file);
  elements.previewImage.src = previewUrl;
  elements.uploadState.hidden = true;
  elements.resultState.hidden = false;
  elements.retryButton.hidden = true;
  setLoading(true);
  generateCaption(file);
}

elements.fileInput.addEventListener("change", (event) => {
  handleFile(event.target.files[0]);
});

elements.uploadArea.addEventListener("dragenter", (event) => {
  event.preventDefault();
  elements.uploadArea.classList.add("drag-active");
});

elements.uploadArea.addEventListener("dragover", (event) => {
  event.preventDefault();
  elements.uploadArea.classList.add("drag-active");
});

elements.uploadArea.addEventListener("dragleave", (event) => {
  if (!elements.uploadArea.contains(event.relatedTarget)) {
    elements.uploadArea.classList.remove("drag-active");
  }
});

elements.uploadArea.addEventListener("drop", (event) => {
  event.preventDefault();
  elements.uploadArea.classList.remove("drag-active");
  handleFile(event.dataTransfer.files[0]);
});

elements.uploadArea.addEventListener("keydown", (event) => {
  if (event.key === "Enter" || event.key === " ") {
    event.preventDefault();
    elements.fileInput.click();
  }
});

elements.retryButton.addEventListener("click", () => {
  if (selectedFile) {
    clearError(elements.resultError);
    setLoading(true);
    generateCaption(selectedFile);
  }
});

elements.heatmapToggle.addEventListener("click", () => {
  setHeatmapMode(!heatmapMode);
});

elements.downloadButton.addEventListener("click", downloadHeatmaps);

elements.chooseButton.addEventListener("click", () => {
  resetPage();
});

window.addEventListener("dragover", (event) => {
  event.preventDefault();
});

window.addEventListener("drop", (event) => {
  event.preventDefault();
});

document.addEventListener("paste", (event) => {
  if (elements.uploadState.hidden) {
    return;
  }
  const imageItem = Array.from(event.clipboardData?.items || [])
    .find((item) => item.type.startsWith("image/"));
  const file = imageItem?.getAsFile();
  if (file) {
    event.preventDefault();
    handleFile(file);
  }
});

renderExamples();
updateRepositoryLink();
warmUpServer();
