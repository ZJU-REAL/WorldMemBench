"use strict";

(() => {
  const results = window.WMB_RESULTS;
  const cases = window.PAPER_CASES;
  const prompts = window.WMB_PROMPTS;
  if (!results || !cases || !prompts) return;

  const $ = id => document.getElementById(id);
  const format = value => Number(value).toFixed(3);

  let metricFilter = "All metrics";
  const filters = ["All metrics", "Video quality", "Static memory", "Dynamic memory", "Camera following"];
  const groupClasses = {
    "Video quality": "group-video",
    "Static memory": "group-static",
    "Dynamic memory": "group-dynamic",
    "Camera following": "group-camera",
  };
  const compactModelNames = {
    "hy-worldplay": "HY\nWorldPlay",
    "matrix-game-3": "Matrix\nGame-3",
    "matrix-game-3.5": "Matrix\nGame-3.5",
    "lingbot-world": "LingBot\nWorld",
    "lingbot-world-v2": "LingBot\nWorld-v2",
    "dreamx-world": "DreamX\nWorld",
  };

  function leaderboardMetrics() {
    return metricFilter === "All metrics"
      ? results.metrics
      : results.metrics.filter(metric => metric.group === metricFilter);
  }

  function renderLeaderboard() {
    const filterBox = $("leaderboard-filters");
    filterBox.replaceChildren();
    for (const name of filters) {
      const button = document.createElement("button");
      button.type = "button";
      button.className = "filter-button";
      button.textContent = name;
      button.setAttribute("aria-pressed", String(metricFilter === name));
      button.addEventListener("click", () => {
        metricFilter = name;
        renderLeaderboard();
      });
      filterBox.append(button);
    }

    const metrics = leaderboardMetrics();
    const table = $("leaderboard-table");
    const head = table.querySelector("thead");
    const body = table.querySelector("tbody");
    head.replaceChildren();
    body.replaceChildren();

    const header = document.createElement("tr");
    const metricHeader = document.createElement("th");
    metricHeader.scope = "col";
    metricHeader.textContent = "Metric";
    header.append(metricHeader);
    for (const model of results.models) {
      const th = document.createElement("th");
      th.scope = "col";
      th.textContent = compactModelNames[model.id] || model.name;
      th.title = model.name;
      header.append(th);
    }
    head.append(header);

    let lastGroup = "";
    for (const metric of metrics) {
      if (metric.group !== lastGroup) {
        const band = document.createElement("tr");
        band.className = `leaderboard-band ${groupClasses[metric.group]}`;
        const cell = document.createElement("th");
        cell.scope = "colgroup";
        cell.colSpan = results.models.length + 1;
        cell.textContent = metric.group;
        band.append(cell);
        body.append(band);
        lastGroup = metric.group;
      }
      const tr = document.createElement("tr");
      tr.className = `leaderboard-metric ${groupClasses[metric.group]}`;
      const name = document.createElement("th");
      name.scope = "row";
      name.textContent = `${metric.label} ${metric.direction === "high" ? "↑" : "↓"}`;
      tr.append(name);
      const ranked = [...new Set(metric.values)].sort((a, b) => metric.direction === "high" ? b - a : a - b);
      for (const value of metric.values) {
        const td = document.createElement("td");
        td.textContent = format(value);
        if (value === ranked[0]) td.classList.add("best-score");
        else if (value === ranked[1]) td.classList.add("second-score");
        tr.append(td);
      }
      body.append(tr);
    }
  }

  let family = "human";
  let anchor = "R1";
  let promptExpanded = false;
  const selectedModels = new Set(["hy-worldplay", "lingbot-world-v2", "sana"]);
  const names = {
    human: { ID: "Identity", App: "Appearance" },
    object: { App: "Appearance", Geo: "Geometry" },
    environment: { Vis: "Visible", Dep: "Depth", Rec: "Reconstruction" },
  };

  function makeFrame(frame, label, model) {
    const figure = document.createElement("figure");
    const image = document.createElement("img");
    image.src = frame.src;
    image.alt = `${model.name}, ${label}, frame ${frame.index}`;
    image.loading = "lazy";
    const caption = document.createElement("figcaption");
    caption.textContent = `${label} · frame ${frame.index}`;
    figure.append(image, caption);
    return figure;
  }

  function makeScoreMap(model) {
    const map = document.createElement("div");
    map.className = "score-map";
    const title = document.createElement("div");
    title.className = "score-map-title";
    title.textContent = "REVISIT SCORES";
    map.append(title);

    const matrix = document.createElement("div");
    matrix.className = "score-matrix";
    const corner = document.createElement("span");
    corner.className = "matrix-corner";
    matrix.append(corner);
    for (const revisit of ["R1", "R2", "R3"]) {
      const button = document.createElement("button");
      button.type = "button";
      button.className = "matrix-return";
      button.textContent = revisit;
      button.setAttribute("aria-pressed", String(anchor === revisit));
      button.addEventListener("click", () => { anchor = revisit; renderComparison(); });
      matrix.append(button);
    }
    const revisits = results.caseScores[family][model.id];
    for (const key of Object.keys(revisits[0])) {
      const label = document.createElement("span");
      label.className = "matrix-metric";
      label.textContent = names[family][key];
      matrix.append(label);
      for (let index = 0; index < 3; index++) {
        const value = revisits[index][key];
        const cell = document.createElement("span");
        cell.className = "score-cell";
        if (anchor === `R${index + 1}`) cell.classList.add("selected-return");
        cell.style.setProperty("--score-ink", (0.06 + value * 0.36).toFixed(3));
        cell.textContent = format(value);
        cell.title = `${names[family][key]}, revisit ${index + 1}: ${format(value)}`;
        matrix.append(cell);
      }
    }
    map.append(matrix);
    return map;
  }

  function renderPrompt() {
    const promptText = $("prompt-text");
    promptText.textContent = prompts[family];
    promptText.classList.toggle("is-clamped", !promptExpanded);
    $("prompt-case").textContent = `${family[0].toUpperCase()}${family.slice(1)} · ${cases[family].case}`;
    const promptToggle = $("prompt-toggle");
    promptToggle.textContent = promptExpanded ? "Show less ↑" : "Read full prompt ↓";
    promptToggle.setAttribute("aria-expanded", String(promptExpanded));
  }

  function renderComparison() {
    $("comparison-grid").querySelectorAll("video").forEach(video => video.pause());
    renderPrompt();

    const caseTabs = $("case-tabs");
    caseTabs.replaceChildren();
    for (const key of ["human", "object", "environment"]) {
      const button = document.createElement("button");
      button.type = "button";
      button.className = "segment-button";
      button.textContent = `${key[0].toUpperCase()}${key.slice(1)} · ${cases[key].case.replace("case_", "")}`;
      button.setAttribute("aria-pressed", String(family === key));
      button.addEventListener("click", () => { family = key; promptExpanded = false; renderComparison(); });
      caseTabs.append(button);
    }

    const returnTabs = $("return-tabs");
    returnTabs.replaceChildren();
    for (const key of ["V", "R1", "R2", "R3"]) {
      const button = document.createElement("button");
      button.type = "button";
      button.className = "segment-button";
      button.textContent = key;
      button.setAttribute("aria-pressed", String(anchor === key));
      button.addEventListener("click", () => { anchor = key; renderComparison(); });
      returnTabs.append(button);
    }

    const options = $("model-options");
    options.replaceChildren();
    for (const model of cases[family].models) {
      const label = document.createElement("label");
      const input = document.createElement("input");
      input.type = "checkbox";
      input.value = model.id;
      input.checked = selectedModels.has(model.id);
      input.addEventListener("change", () => {
        if (input.checked) selectedModels.add(model.id);
        else selectedModels.delete(model.id);
        renderComparison();
      });
      label.append(input, document.createTextNode(model.name));
      options.append(label);
    }

    const grid = $("comparison-grid");
    grid.replaceChildren();
    const selected = cases[family].models.filter(model => selectedModels.has(model.id));
    $("compare-context").textContent = `${cases[family].case} · LRLRLR · ${anchor} · ${selected.length} models`;
    if (!selected.length) {
      const empty = document.createElement("p");
      empty.textContent = "Choose at least one model to compare.";
      grid.append(empty);
      return;
    }
    for (const model of selected) {
      const card = document.createElement("article");
      card.className = "comparison-card";
      const heading = document.createElement("h3");
      heading.textContent = model.name;
      card.append(heading);

      const frames = document.createElement("div");
      frames.className = anchor === "V" ? "frame-pair initial" : "frame-pair";
      frames.append(makeFrame(model.frames.V, "V · initial", model));
      if (anchor !== "V") frames.append(makeFrame(model.frames[anchor], `${anchor} · return`, model));
      card.append(frames);

      card.append(makeScoreMap(model));

      const video = document.createElement("video");
      video.controls = true;
      video.playsInline = true;
      video.preload = "metadata";
      video.poster = model.frames[anchor].src;
      video.src = model.video.src;
      video.setAttribute("aria-label", `${model.name} ${family} case ${cases[family].case} video`);
      const seek = () => {
        const time = model.frames[anchor].index / model.video.fps;
        if (Number.isFinite(time) && time < video.duration) video.currentTime = time;
      };
      video.addEventListener("loadedmetadata", seek, { once: true });
      card.append(video);
      grid.append(card);
    }
  }

  function renderBenchmarkSymbols() {
    const glyphs = {
      C: '<svg viewBox="0 0 24 20" aria-hidden="true"><path d="M2 6h4l2-3h8l2 3h4v12H2z" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linejoin="round"/><circle cx="12" cy="12" r="3.5" fill="none" stroke="currentColor" stroke-width="1.7"/></svg>',
      A: '<svg viewBox="0 0 28 22" aria-hidden="true"><rect x="10" y="1" width="8" height="8" rx="1"/><rect x="1" y="11" width="8" height="8" rx="1"/><rect x="10" y="11" width="8" height="8" rx="1"/><rect x="19" y="11" width="8" height="8" rx="1"/><text x="14" y="7">W</text><text x="5" y="17">A</text><text x="14" y="17">S</text><text x="23" y="17">D</text></svg>',
      L: '<span class="language-letter" aria-hidden="true">T</span>',
    };
    const labels = { C: "camera-driven", A: "action-driven", L: "language-driven" };
    const symbolText = { "✓": ["yes", "✓", "Explicit coverage"], "◐": ["partial", "△", "Partial coverage"], "—": ["none", "×", "Not reported"] };
    for (const row of document.querySelectorAll(".benchmark-table tbody tr")) {
      const input = row.cells[1];
      const active = new Set(input.textContent.split("·"));
      input.replaceChildren();
      input.classList.add("paradigm-cell");
      input.setAttribute("aria-label", `Input: ${[...active].map(code => labels[code]).join(", ")}`);
      for (const code of ["C", "A", "L"]) {
        const glyph = document.createElement("span");
        glyph.className = `paradigm-glyph glyph-${code.toLowerCase()} ${active.has(code) ? "active" : "inactive"}`;
        glyph.innerHTML = glyphs[code];
        glyph.title = `${labels[code]}: ${active.has(code) ? "evaluated" : "not evaluated"}`;
        input.append(glyph);
      }
      for (let index = 3; index < row.cells.length; index++) {
        const cell = row.cells[index];
        const [className, symbol, meaning] = symbolText[cell.textContent.trim()];
        cell.replaceChildren();
        const mark = document.createElement("span");
        mark.className = `coverage-symbol ${className}`;
        mark.textContent = symbol;
        mark.title = meaning;
        cell.setAttribute("aria-label", meaning);
        cell.append(mark);
      }
    }
  }
  renderLeaderboard();
  $("prompt-toggle").addEventListener("click", () => { promptExpanded = !promptExpanded; renderPrompt(); });
  renderComparison();
  renderBenchmarkSymbols();
})();
