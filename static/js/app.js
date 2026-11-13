document.addEventListener("DOMContentLoaded", () => {
  // Elements
  const textInput = document.getElementById("textInput");
  const detectBtn = document.getElementById("detectBtn");
  const clearBtn = document.getElementById("clearBtn");
  const charCount = document.getElementById("charCount");
  const wordCount = document.getElementById("wordCount");

  const resultPlaceholder = document.getElementById("resultPlaceholder");
  const resultCard = document.getElementById("resultCard");

  const flagEmoji = document.getElementById("flagEmoji");
  const langTitle = document.getElementById("langTitle");
  const nativeName = document.getElementById("nativeName");
  const langCode = document.getElementById("langCode");
  const confidencePercent = document.getElementById("confidencePercent");
  const progressFill = document.getElementById("progressFill");
  const langFamily = document.getElementById("langFamily");
  const langScript = document.getElementById("langScript");
  const cleanedTextPreview = document.getElementById("cleanedTextPreview");
  const distributionList = document.getElementById("distributionList");

  // Sample chips
  const sampleChips = document.querySelectorAll(".chip[data-sample]");

  // Tab navigation
  const tabBtns = document.querySelectorAll(".tab-btn");
  const tabPanels = document.querySelectorAll(".tab-panel");

  tabBtns.forEach((btn) => {
    btn.addEventListener("click", () => {
      tabBtns.forEach((b) => b.classList.remove("active"));
      tabPanels.forEach((p) => p.classList.remove("active"));

      btn.classList.add("active");
      const target = document.getElementById(btn.getAttribute("data-tab"));
      if (target) target.classList.add("active");
    });
  });

  // Text statistics update
  function updateCounters() {
    const text = textInput.value;
    charCount.textContent = `${text.length} chars`;
    const words = text.trim() ? text.trim().split(/\s+/).length : 0;
    wordCount.textContent = `${words} words`;
  }

  textInput.addEventListener("input", () => {
    updateCounters();
    debouncedPredict();
  });

  clearBtn.addEventListener("click", () => {
    textInput.value = "";
    updateCounters();
    resetResult();
    textInput.focus();
  });

  detectBtn.addEventListener("click", () => {
    performPrediction();
  });

  sampleChips.forEach((chip) => {
    chip.addEventListener("click", () => {
      textInput.value = chip.getAttribute("data-sample");
      updateCounters();
      performPrediction();
      textInput.focus();
    });
  });

  function resetResult() {
    resultPlaceholder.style.display = "block";
    resultCard.style.display = "none";
  }

  // Debounce helper
  let debounceTimer = null;
  function debouncedPredict() {
    clearTimeout(debounceTimer);
    const text = textInput.value.trim();
    if (text.length >= 4) {
      debounceTimer = setTimeout(() => {
        performPrediction();
      }, 350);
    } else if (text.length === 0) {
      resetResult();
    }
  }

  // Prediction API call
  async function performPrediction() {
    const text = textInput.value.trim();
    if (!text) {
      resetResult();
      return;
    }

    detectBtn.disabled = true;
    detectBtn.textContent = "Detecting...";

    try {
      const response = await fetch("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: text, top_k: 4 }),
      });

      const data = await response.json();

      if (!response.ok || !data.success) {
        showError(data.error || "Unable to detect language.");
        return;
      }

      displayResult(data);
    } catch (err) {
      showError("Connection error while calling prediction service.");
    } finally {
      detectBtn.disabled = false;
      detectBtn.innerHTML = `<span>Detect Language</span> <svg width="18" height="18" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" d="M14 5l7 7m0 0l-7 7m7-7H3"/></svg>`;
    }
  }

  function displayResult(data) {
    resultPlaceholder.style.display = "none";
    resultCard.style.display = "block";

    flagEmoji.textContent = data.flag || "🌐";
    langTitle.textContent = data.predicted_language;
    nativeName.textContent = data.native || data.predicted_language;
    langCode.textContent = (data.code || "und").toUpperCase();

    confidencePercent.textContent = `${data.confidence}%`;
    progressFill.style.width = `${Math.min(100, data.confidence)}%`;

    langFamily.textContent = data.family || "Unknown";
    langScript.textContent = data.script || "Unknown";
    cleanedTextPreview.textContent = data.cleaned_text || "(empty)";

    // Top distribution list
    distributionList.innerHTML = "";
    if (data.top_predictions && data.top_predictions.length > 0) {
      data.top_predictions.forEach((item) => {
        const itemEl = document.createElement("div");
        itemEl.className = "prob-item";
        itemEl.innerHTML = `
          <div class="prob-header">
            <span>${item.flag} ${item.language} (${item.native})</span>
            <span>${item.confidence_percent}%</span>
          </div>
          <div class="prob-track">
            <div class="prob-fill" style="width: ${item.confidence_percent}%;"></div>
          </div>
        `;
        distributionList.appendChild(itemEl);
      });
    }
  }

  function showError(msg) {
    resultPlaceholder.style.display = "block";
    resultCard.style.display = "none";
    resultPlaceholder.innerHTML = `
      <div class="placeholder-icon" style="color: #ef4444;">⚠️</div>
      <p style="color: #ef4444; font-weight: 600;">${msg}</p>
    `;
  }
});
