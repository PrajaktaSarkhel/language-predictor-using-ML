/**
 * LinguaScan Studio - Frontend Application Engine
 * Implements real-time inference, Unicode script telemetry, theme persistence,
 * audio pronunciation, interactive API playground, and accessible UI interactions.
 */

document.addEventListener("DOMContentLoaded", () => {
  // =========================================================================
  // 1. Theme Management (Light / Dark)
  // =========================================================================
  const themeToggleBtn = document.getElementById("themeToggleBtn");
  const themeIconSun = document.getElementById("themeIconSun");
  const themeIconMoon = document.getElementById("themeIconMoon");
  const htmlRoot = document.documentElement;

  function initTheme() {
    const savedTheme = localStorage.getItem("linguascan-theme");
    const prefersDark = window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches;
    const activeTheme = savedTheme || (prefersDark ? "dark" : "light");
    applyTheme(activeTheme);
  }

  function applyTheme(theme) {
    htmlRoot.setAttribute("data-theme", theme);
    localStorage.setItem("linguascan-theme", theme);
    if (theme === "dark") {
      themeIconSun.style.display = "block";
      themeIconMoon.style.display = "none";
    } else {
      themeIconSun.style.display = "none";
      themeIconMoon.style.display = "block";
    }
  }

  if (themeToggleBtn) {
    themeToggleBtn.addEventListener("click", () => {
      const current = htmlRoot.getAttribute("data-theme") || "light";
      applyTheme(current === "dark" ? "light" : "dark");
    });
  }
  initTheme();

  // =========================================================================
  // 2. Tab Navigation
  // =========================================================================
  const tabBtns = document.querySelectorAll(".segment-btn");
  const tabContents = document.querySelectorAll(".tab-content");

  function switchTab(targetTabId) {
    tabBtns.forEach((btn) => {
      const isTarget = btn.getAttribute("data-tab") === targetTabId;
      btn.classList.toggle("active", isTarget);
      btn.setAttribute("aria-selected", isTarget ? "true" : "false");
    });

    tabContents.forEach((panel) => {
      panel.classList.toggle("active", panel.id === targetTabId);
    });
  }

  tabBtns.forEach((btn) => {
    btn.addEventListener("click", () => {
      const target = btn.getAttribute("data-tab");
      switchTab(target);
    });
  });

  // =========================================================================
  // 3. Elements & State
  // =========================================================================
  const textInput = document.getElementById("textInput");
  const detectBtn = document.getElementById("detectBtn");
  const pasteBtn = document.getElementById("pasteBtn");
  const clearBtn = document.getElementById("clearBtn");
  const charCount = document.getElementById("charCount");
  const wordCount = document.getElementById("wordCount");
  const latencyBadge = document.getElementById("latencyBadge");

  const detectedScriptBadge = document.getElementById("detectedScriptBadge");
  const scriptBadgeText = document.getElementById("scriptBadgeText");
  const scriptBadgeIcon = document.getElementById("scriptBadgeIcon");

  const inspectorEmptyState = document.getElementById("inspectorEmptyState");
  const inspectorResultCard = document.getElementById("inspectorResultCard");

  const resultFlagAvatar = document.getElementById("resultFlagAvatar");
  const resultLanguageTitle = document.getElementById("resultLanguageTitle");
  const resultNativeName = document.getElementById("resultNativeName");
  const resultIsoCode = document.getElementById("resultIsoCode");
  const resultConfidenceScore = document.getElementById("resultConfidenceScore");
  const resultConfidenceFill = document.getElementById("resultConfidenceFill");
  const resultMetaFamily = document.getElementById("resultMetaFamily");
  const resultMetaScript = document.getElementById("resultMetaScript");
  const spectrumList = document.getElementById("spectrumList");

  const nlpDrawerToggle = document.getElementById("nlpDrawerToggle");
  const nlpDrawerBody = document.getElementById("nlpDrawerBody");
  const drawerArrow = document.getElementById("drawerArrow");
  const resultCleanedPreview = document.getElementById("resultCleanedPreview");
  const charRatioStats = document.getElementById("charRatioStats");

  const speakBtn = document.getElementById("speakBtn");
  const copyResultBtn = document.getElementById("copyResultBtn");
  const presetChips = document.querySelectorAll(".preset-chip");

  let latestPredictionData = null;

  // =========================================================================
  // 4. Unicode Script Detector (Heuristic Real-time Telemetry)
  // =========================================================================
  function identifyUnicodeScript(text) {
    if (!text || !text.trim()) {
      return { script: "Waiting for text...", icon: "🔤" };
    }

    const devanagariRegex = /[\u0900-\u097F]/;
    const arabicRegex = /[\u0600-\u06FF\u0750-\u077F]/;
    const cyrillicRegex = /[\u0400-\u04FF]/;
    const hanziRegex = /[\u4E00-\u9FFF]/;
    const kanaRegex = /[\u3040-\u30FF]/;
    const hangulRegex = /[\uAC00-\uD7AF\u1100-\u11FF]/;
    const greekRegex = /[\u0370-\u03FF]/;
    const tamilRegex = /[\u0B80-\u0BFF]/;
    const kannadaRegex = /[\u0C80-\u0CFF]/;
    const malayalamRegex = /[\u0D00-\u0D7F]/;
    const thaiRegex = /[\u0E00-\u0E7F]/;
    const latinRegex = /[A-Za-z\u00C0-\u024F]/;

    if (devanagariRegex.test(text)) return { script: "Devanagari script", icon: "🇮🇳" };
    if (arabicRegex.test(text)) return { script: "Perso-Arabic script", icon: "🇸🇦" };
    if (cyrillicRegex.test(text)) return { script: "Cyrillic script", icon: "🇷🇺" };
    if (kanaRegex.test(text)) return { script: "Japanese (Kana/Kanji)", icon: "🇯🇵" };
    if (hangulRegex.test(text)) return { script: "Korean Hangul script", icon: "🇰🇷" };
    if (hanziRegex.test(text)) return { script: "Hanzi Logographic", icon: "🇨🇳" };
    if (tamilRegex.test(text)) return { script: "Tamil script", icon: "🇮🇳" };
    if (kannadaRegex.test(text)) return { script: "Kannada script", icon: "🇮🇳" };
    if (malayalamRegex.test(text)) return { script: "Malayalam script", icon: "🇮🇳" };
    if (thaiRegex.test(text)) return { script: "Thai script", icon: "🇹🇭" };
    if (greekRegex.test(text)) return { script: "Greek script", icon: "🇬🇷" };
    if (latinRegex.test(text)) return { script: "Latin script", icon: "🌐" };

    return { script: "General text stream", icon: "🔤" };
  }

  function updateCountersAndScript() {
    const val = textInput.value;
    charCount.textContent = `${val.length} chars`;
    const words = val.trim() ? val.trim().split(/\s+/).length : 0;
    wordCount.textContent = `${words} words`;

    const info = identifyUnicodeScript(val);
    scriptBadgeText.textContent = info.script;
    scriptBadgeIcon.textContent = info.icon;
  }

  // =========================================================================
  // 5. Debounced Prediction Logic
  // =========================================================================
  let debounceTimeout = null;

  textInput.addEventListener("input", () => {
    updateCountersAndScript();
    clearTimeout(debounceTimeout);
    const text = textInput.value.trim();
    if (text.length >= 3) {
      debounceTimeout = setTimeout(() => {
        executeInference();
      }, 250);
    } else if (text.length === 0) {
      resetInspector();
    }
  });

  // Keyboard shortcut: Ctrl + Enter to analyze, Esc to clear
  textInput.addEventListener("keydown", (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key === "Enter") {
      e.preventDefault();
      executeInference();
    } else if (e.key === "Escape") {
      clearInput();
    }
  });

  detectBtn.addEventListener("click", () => executeInference());

  if (pasteBtn) {
    pasteBtn.addEventListener("click", async () => {
      try {
        const text = await navigator.clipboard.readText();
        if (text) {
          textInput.value = text;
          updateCountersAndScript();
          executeInference();
          showToast("📋 Text pasted from clipboard");
        }
      } catch (err) {
        textInput.focus();
        showToast("⚠️ Please paste using Ctrl+V");
      }
    });
  }

  clearBtn.addEventListener("click", () => clearInput());

  function clearInput() {
    textInput.value = "";
    updateCountersAndScript();
    resetInspector();
    textInput.focus();
  }

  function resetInspector() {
    inspectorEmptyState.style.display = "flex";
    inspectorResultCard.style.display = "none";
    latencyBadge.style.display = "none";
    latestPredictionData = null;
  }

  // Preset Chips
  presetChips.forEach((chip) => {
    chip.addEventListener("click", () => {
      const sample = chip.getAttribute("data-sample");
      if (sample) {
        textInput.value = sample;
        updateCountersAndScript();
        executeInference();
        textInput.focus();
      }
    });
  });

  // =========================================================================
  // 6. Execute Language Detection API Call
  // =========================================================================
  async function executeInference() {
    const rawText = textInput.value.trim();
    if (!rawText) {
      resetInspector();
      return;
    }

    detectBtn.disabled = true;
    detectBtn.querySelector("span").textContent = "Analyzing...";

    const tStart = performance.now();

    try {
      const response = await fetch("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: rawText, top_k: 4 }),
      });

      const data = await response.json();
      const elapsedMs = Math.round(performance.now() - tStart);

      if (!response.ok || !data.success) {
        showInspectorError(data.error || "Unable to determine language.");
        return;
      }

      latestPredictionData = data;
      renderInspectorResult(data, elapsedMs, rawText);
    } catch (err) {
      showInspectorError("Failed to reach inference server. Please verify connection.");
    } finally {
      detectBtn.disabled = false;
      detectBtn.querySelector("span").textContent = "Detect Language";
    }
  }

  function renderInspectorResult(data, elapsedMs, rawText) {
    inspectorEmptyState.style.display = "none";
    inspectorResultCard.style.display = "block";

    // Latency
    latencyBadge.style.display = "inline-flex";
    latencyBadge.textContent = `⚡ ${elapsedMs} ms`;

    // Identified Entity
    resultFlagAvatar.textContent = data.flag || "🌐";
    resultLanguageTitle.textContent = data.predicted_language;
    resultNativeName.textContent = data.native || data.predicted_language;
    resultIsoCode.textContent = (data.code || "und").toUpperCase();

    // Confidence
    const conf = Number(data.confidence);
    resultConfidenceScore.textContent = `${conf.toFixed(1)}%`;
    resultConfidenceFill.style.width = `${Math.min(100, Math.max(0, conf))}%`;

    // Dynamic confidence palette
    if (conf >= 85) {
      resultConfidenceScore.style.color = "var(--status-high)";
      resultConfidenceFill.style.backgroundColor = "var(--status-high)";
    } else if (conf >= 50) {
      resultConfidenceScore.style.color = "var(--status-med)";
      resultConfidenceFill.style.backgroundColor = "var(--status-med)";
    } else {
      resultConfidenceScore.style.color = "var(--status-low)";
      resultConfidenceFill.style.backgroundColor = "var(--status-low)";
    }

    // Metadata
    resultMetaFamily.textContent = data.family || "Unknown";
    resultMetaScript.textContent = data.script || "Unknown";

    // Candidate Spectrum
    spectrumList.innerHTML = "";
    if (data.top_predictions && data.top_predictions.length > 0) {
      data.top_predictions.forEach((cand, idx) => {
        const row = document.createElement("div");
        row.className = "spectrum-row";
        const candConf = Number(cand.confidence_percent || 0);

        row.innerHTML = `
          <div class="spectrum-info">
            <span class="spectrum-lang">
              <span>${cand.flag || "🌐"}</span>
              <strong>${cand.language}</strong>
              <span style="font-size:0.75rem; color:var(--text-muted);">(${cand.native || ""})</span>
            </span>
            <span class="spectrum-percent">${candConf.toFixed(1)}%</span>
          </div>
          <div class="spectrum-track">
            <div class="spectrum-bar" style="width: ${Math.min(100, candConf)}%; opacity: ${1 - idx * 0.18};"></div>
          </div>
        `;
        spectrumList.appendChild(row);
      });
    }

    // NLP Drawer Telemetry
    resultCleanedPreview.textContent = data.cleaned_text || "(empty)";
    charRatioStats.textContent = `Raw: ${rawText.length} &rarr; Cleaned: ${(data.cleaned_text || "").length}`;
  }

  function showInspectorError(msg) {
    inspectorEmptyState.style.display = "flex";
    inspectorResultCard.style.display = "none";
    latencyBadge.style.display = "none";
    inspectorEmptyState.innerHTML = `
      <div class="empty-state-icon" style="color:var(--status-low);">⚠️</div>
      <p class="empty-state-text" style="color:var(--status-low); font-weight:600;">${msg}</p>
    `;
  }

  // =========================================================================
  // 7. NLP Preprocessing Telemetry Drawer Accordion
  // =========================================================================
  if (nlpDrawerToggle) {
    nlpDrawerToggle.addEventListener("click", () => {
      const isOpen = nlpDrawerBody.style.display === "flex";
      nlpDrawerBody.style.display = isOpen ? "none" : "flex";
      drawerArrow.style.transform = isOpen ? "rotate(0deg)" : "rotate(180deg)";
      nlpDrawerToggle.setAttribute("aria-expanded", !isOpen);
    });
  }

  // =========================================================================
  // 8. Pronunciation (Web Speech API) & Copying
  // =========================================================================
  if (speakBtn) {
    speakBtn.addEventListener("click", () => {
      const textToSpeak = textInput.value.trim();
      if (!textToSpeak) return;

      if ("speechSynthesis" in window) {
        window.speechSynthesis.cancel();
        const utterance = new SpeechSynthesisUtterance(textToSpeak);
        if (latestPredictionData && latestPredictionData.code) {
          utterance.lang = latestPredictionData.code;
        }
        window.speechSynthesis.speak(utterance);
        showToast("🔊 Playing pronunciation");
      } else {
        showToast("⚠️ Speech synthesis not supported in this browser");
      }
    });
  }

  if (copyResultBtn) {
    copyResultBtn.addEventListener("click", () => {
      if (!latestPredictionData) return;
      const summary = `Detected Language: ${latestPredictionData.predicted_language} (${latestPredictionData.native})
ISO Code: ${latestPredictionData.code.toUpperCase()}
Confidence: ${latestPredictionData.confidence}%
Family: ${latestPredictionData.family}
Script: ${latestPredictionData.script}`;

      navigator.clipboard.writeText(summary).then(() => {
        showToast("📋 Result copied to clipboard");
      });
    });
  }

  // =========================================================================
  // 9. Supported Languages Directory: Search & Filter
  // =========================================================================
  const langSearchInput = document.getElementById("langSearchInput");
  const familyFilterPills = document.querySelectorAll("#familyFilterGroup .filter-pill");
  const catalogCards = document.querySelectorAll(".lang-catalog-card");

  // Sample sentences for all 28 languages for quick direct testing
  const LANGUAGE_SAMPLES = {
    "Arabic": "مرحبا بكم في هذا المشروع المتطور للتعرف التلقائي على اللغات.",
    "Chinese": "自然语言处理是计算机科学领域与人工智能领域中的一个重要方向。",
    "Danish": "Kunstig intelligens og maskinlæring revolutionerer softwareudvikling.",
    "Dutch": "Natuurlijke taalverwerking stelt computers in staat om menselijke taal te begrijpen.",
    "English": "Language detection is the backbone of modern multilingual artificial intelligence systems.",
    "Estonian": "Masinõpe ja tehisintellekt loovad uusi võimalusi keeletehnoloogias.",
    "French": "Le traitement automatique du langage naturel permet aux machines de comprendre la langue humaine.",
    "German": "Maschinelles Lernen und künstliche Intelligenz revolutionieren die globale Kommunikation.",
    "Greek": "Η επεξεργασία φυσικής γλώσσας επιτρέπει στους υπολογιστές να κατανοούν την ανθρώπινη γλώσσα.",
    "Hindi": "प्राकृतिक भाषा प्रसंस्करण कंप्यूटर को मानवीय भाषा समझने और उसका विश्लेषण करने में सक्षम बनाता है।",
    "Indonesian": "Pemrosesan bahasa alami memungkinkan komputer memahami dan menganalisis teks secara otomatis.",
    "Italian": "L'elaborazione del linguaggio naturale è una branca fondamentale dell'intelligenza artificiale.",
    "Japanese": "自然言語処理は、コンピュータに人間の言語を理解させる人工知能の技術です。",
    "Kannada": "ನೈಸರ್ಗಿಕ ಭಾಷಾ ಸಂಸ್ಕರಣೆಯು ಕಂಪ್ಯೂಟರ್‌ಗಳಿಗೆ ಮಾನವ ಭಾಷೆಯನ್ನು ಅರ್ಥಮಾಡಿಕೊಳ್ಳಲು ಅನುವು ಮಾಡಿಕೊಡುತ್ತದೆ.",
    "Korean": "자연어 처리는 컴퓨터가 인간의 언어를 이해하고 처리할 수 있도록 돕는 인공지능 기술입니다.",
    "Latin": "Litterae et linguae mundi scientiam ac humanitatem per saecula conservant.",
    "Malayalam": "കമ്പ്യൂട്ടറുകൾക്ക് മനുഷ്യഭാഷ മനസ്സിലാക്കാൻ സഹായിക്കുന്ന ഒരു സാങ്കേതികവിദ്യയാണ് സ്വാഭാവിക ഭാഷാ സംസ്കരണം.",
    "Persian": "پردازش زبان طبیعی به رایانه‌ها امکان می‌دهد زبان انسان را درک و تحلیل کنند.",
    "Portuguese": "O processamento de linguagem natural permite que os computadores compreendam a linguagem humana.",
    "Pushto": "د طبیعي ژبې پروسس کول کمپیوټرونو ته دا وړتیا ورکوي چې انساني ژبه درک کړي.",
    "Romanian": "Procesarea limbajului natural permite computerelor să înțeleagă și să interpreteze limba umană.",
    "Russian": "Обработка естественного языка позволяет компьютерам понимать и анализировать человеческую речь.",
    "Spanish": "El procesamiento del lenguaje natural permite a las máquinas comprender y analizar el texto humano.",
    "Swedish": "Naturlig språkbehandling gör det möjligt för datorer att förstå och analysera mänskligt språk.",
    "Tamil": "இயற்கை மொழி செயலாக்கம் கணினிகளுக்கு மனித மொழியைப் புரிந்துகொள்ளவும் பகுப்பாய்வு செய்யவும் உதவுகிறது.",
    "Thai": "การประมวลผลภาษาธรรมชาติช่วยให้คอมพิวเตอร์สามารถเข้าใจและสื่อสารด้วยภาษาของมนุษย์ได้",
    "Turkish": "Doğal dil işleme, bilgisayarların insan dilini anlamasını ve yorumlamasını sağlayan bir yapay zekâ dalıdır.",
    "Urdu": "قدرتی زبان کی پروسیسنگ کمپیوٹر کو انسانی زبان کو سمجھنے اور تجزیہ کرنے کی صلاحیت فراہم کرتی ہے۔"
  };

  function filterDirectory() {
    const query = (langSearchInput ? langSearchInput.value : "").toLowerCase().trim();
    const activePill = document.querySelector("#familyFilterGroup .filter-pill.active");
    const activeFamily = activePill ? activePill.getAttribute("data-family").toLowerCase() : "all";

    catalogCards.forEach((card) => {
      const langName = card.getAttribute("data-lang") || "";
      const script = card.getAttribute("data-script") || "";
      const family = card.getAttribute("data-family") || "";

      const matchesSearch = !query || langName.includes(query) || script.includes(query) || family.includes(query);
      const matchesFamily = activeFamily === "all" || family.includes(activeFamily);

      card.style.display = matchesSearch && matchesFamily ? "" : "none";
    });
  }

  if (langSearchInput) {
    langSearchInput.addEventListener("input", filterDirectory);
  }

  familyFilterPills.forEach((pill) => {
    pill.addEventListener("click", () => {
      familyFilterPills.forEach((p) => p.classList.remove("active"));
      pill.classList.add("active");
      filterDirectory();
    });
  });

  // "Test in Detector" buttons in catalog
  document.querySelectorAll(".load-sample-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      const lang = btn.getAttribute("data-lang");
      const sample = LANGUAGE_SAMPLES[lang] || `Sample phrase in ${lang}.`;

      switchTab("tab-detect");
      textInput.value = sample;
      updateCountersAndScript();
      executeInference();
      textInput.focus();
      window.scrollTo({ top: 0, behavior: "smooth" });
    });
  });

  // =========================================================================
  // 10. API Playground & Code Copying
  // =========================================================================
  const apiPlaygroundInput = document.getElementById("apiPlaygroundInput");
  const apiSendBtn = document.getElementById("apiSendBtn");
  const apiResponseArea = document.getElementById("apiResponseArea");
  const apiStatusCode = document.getElementById("apiStatusCode");

  if (apiSendBtn && apiPlaygroundInput) {
    apiSendBtn.addEventListener("click", async () => {
      const q = apiPlaygroundInput.value.trim();
      if (!q) return;

      apiSendBtn.disabled = true;
      apiSendBtn.textContent = "Calling...";

      try {
        const res = await fetch("/api/predict", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ text: q, top_k: 3 }),
        });

        apiStatusCode.textContent = res.status;
        const json = await res.json();
        apiResponseArea.textContent = JSON.stringify(json, null, 2);
      } catch (err) {
        apiStatusCode.textContent = "ERR";
        apiResponseArea.textContent = JSON.stringify({ error: err.message }, null, 2);
      } finally {
        apiSendBtn.disabled = false;
        apiSendBtn.textContent = "Send";
      }
    });
  }

  // Copy code buttons
  document.querySelectorAll(".copy-code-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      const targetId = btn.getAttribute("data-target");
      const targetEl = document.getElementById(targetId);
      if (targetEl) {
        const text = targetEl.textContent;
        navigator.clipboard.writeText(text).then(() => {
          showToast("📋 Code copied to clipboard");
        });
      }
    });
  });

  // =========================================================================
  // 11. Confusion Matrix Zoom Lightbox Modal
  // =========================================================================
  const heatmapTrigger = document.getElementById("heatmapTrigger");
  const imageModal = document.getElementById("imageModal");
  const modalCloseBtn = document.getElementById("modalCloseBtn");

  if (heatmapTrigger && imageModal) {
    heatmapTrigger.addEventListener("click", () => {
      imageModal.classList.add("active");
      imageModal.setAttribute("aria-hidden", "false");
    });

    modalCloseBtn.addEventListener("click", () => {
      closeModal();
    });

    imageModal.addEventListener("click", (e) => {
      if (e.target === imageModal) closeModal();
    });

    document.addEventListener("keydown", (e) => {
      if (e.key === "Escape" && imageModal.classList.contains("active")) {
        closeModal();
      }
    });

    function closeModal() {
      imageModal.classList.remove("active");
      imageModal.setAttribute("aria-hidden", "true");
    }
  }

  // =========================================================================
  // 12. Toast Notification Helper
  // =========================================================================
  const toastNotice = document.getElementById("toastNotice");
  const toastMessage = document.getElementById("toastMessage");
  let toastTimer = null;

  function showToast(msg) {
    if (!toastNotice) return;
    toastMessage.textContent = msg;
    toastNotice.classList.add("show");
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => {
      toastNotice.classList.remove("show");
    }, 2400);
  }
});
