const ctx = document.getElementById("chart").getContext("2d");
const MAX_AXES = 3;
const AXIS_POSITIONS = ["left", "right", "left"];

const COLOR_PALETTE = [
  "#fb7185",
  "#38bdf8",
  "#a3e635",
  "#fbbf24",
  "#c084fc",
  "#2dd4bf",
  "#f97316",
  "#818cf8",
  "#e879f9",
  "#4ade80",
];

const PRESET_DURATIONS_MS = {
  "1h": 60 * 60 * 1000,
  "12h": 12 * 60 * 60 * 1000,
  "1d": 24 * 60 * 60 * 1000,
  "1w": 7 * 24 * 60 * 60 * 1000,
};

const NICE_BUCKET_SECONDS = [
  1, 5, 10, 30, 60, 300, 900, 1800, 3600, 21600, 86400,
];
const MAX_POINTS = 2000;

function estimateBucketSeconds(start, end) {
  const totalSeconds = (end - start) / 1000;
  if (totalSeconds <= 0) return null;
  if (totalSeconds <= MAX_POINTS) return null;
  const required = totalSeconds / MAX_POINTS;
  for (const b of NICE_BUCKET_SECONDS) {
    if (b >= required) return b;
  }
  return NICE_BUCKET_SECONDS[NICE_BUCKET_SECONDS.length - 1];
}

function colorForParam(param) {
  const index = window.ALL_PARAMETERS.indexOf(param);
  return COLOR_PALETTE[index % COLOR_PALETTE.length];
}

let chart = new Chart(ctx, {
  type: "line",
  data: { datasets: [] },
  options: {
    responsive: true,
    maintainAspectRatio: false,
    animation: false,
    parsing: false,
    scales: {
      x: {
        type: "time",
        title: { display: true, text: "Time", color: "#858697" },
        ticks: { color: "#858697" },
        grid: { color: "rgba(133, 134, 151, 0.12)" },
      },
    },
    plugins: {
      legend: {
        position: "top",
        labels: {
          color: "#c6c7d0",
          usePointStyle: true,
          boxWidth: 8,
          padding: 18,
        },
      },
    },
  },
});

function setStatus(message) {
  document.getElementById("chart-status").textContent = message;
}

function seriesToPoints(seriesArray) {
  return seriesArray.map((point) => ({ x: new Date(point.t), y: point.v }));
}

// --- Range state ---

let currentRange = { mode: "preset", preset: null, start: null, end: null };

function setPresetRange(preset, { silent } = {}) {
  currentRange = { mode: "preset", preset, start: null, end: null };
  document.querySelectorAll(".preset-btn").forEach((btn) => {
    btn.classList.toggle("active", btn.dataset.preset === preset);
    btn.setAttribute("aria-pressed", String(btn.dataset.preset === preset));
  });
  if (!silent) updateUrl();
}

function setCustomRange(start, end, { silent } = {}) {
  currentRange = { mode: "custom", preset: null, start, end };
  document.querySelectorAll(".preset-btn").forEach((btn) => {
    btn.classList.remove("active");
    btn.setAttribute("aria-pressed", "false");
  });
  document.getElementById("start-input").value = start;
  document.getElementById("end-input").value = end;
  if (!silent) updateUrl();
}

function getRequestWindow() {
  if (currentRange.mode === "preset") {
    const durationMs = PRESET_DURATIONS_MS[currentRange.preset];
    const end = new Date();
    const start = new Date(end.getTime() - durationMs);
    return { start, end };
  }
  return {
    start: new Date(currentRange.start),
    end: new Date(currentRange.end),
  };
}

// --- URL state sync ---

function getSelectedColumns() {
  return Array.from(
    document.querySelectorAll('input[name="columns"]:checked'),
  ).map((el) => el.value);
}

function buildUrlParams(columns) {
  const params = new URLSearchParams();
  columns.forEach((c) => params.append("columns", c));
  if (currentRange.mode === "preset") {
    params.append("preset", currentRange.preset);
  } else {
    params.append("start", currentRange.start);
    params.append("end", currentRange.end);
  }
  return params;
}

function updateUrl() {
  const columns = getSelectedColumns();
  const params = buildUrlParams(columns);
  const newUrl = `${window.location.pathname}?${params.toString()}`;
  history.pushState({}, "", newUrl);
}

// Reads the URL's query string and applies it to DOM + currentRange.
// Returns true if the URL actually specified state; false if it was empty
// (meaning the server-rendered defaults should stand as-is).
function applyStateFromUrl() {
  const params = new URLSearchParams(window.location.search);
  if ([...params.keys()].length === 0) return false;

  const columns = params.getAll("columns");
  if (columns.length > 0) {
    document.querySelectorAll('input[name="columns"]').forEach((cb) => {
      cb.checked = columns.includes(cb.value);
    });
  }

  const preset = params.get("preset");
  const start = params.get("start");
  const end = params.get("end");

  if (preset && PRESET_DURATIONS_MS[preset]) {
    setPresetRange(preset, { silent: true });
  } else if (start && end) {
    setCustomRange(start, end, { silent: true });
  }

  return true;
}

function initRangeFromDom() {
  const activeBtn = document.querySelector(".preset-btn.active");
  currentRange = {
    mode: "preset",
    preset: activeBtn.dataset.preset,
    start: null,
    end: null,
  };
}

// --- Axis assignment ---

function computeAxisAssignments(seriesEntries) {
  const axes = [];
  const assignments = {};

  for (const { param, points } of seriesEntries) {
    const values = points
      .map((p) => p.y)
      .filter((v) => v !== null && v !== undefined);
    const min = values.length ? Math.min(...values) : 0;
    const max = values.length ? Math.max(...values) : 1;

    if (axes.length < MAX_AXES) {
      const id = `y${axes.length}`;
      axes.push({ id, min, max });
      assignments[param] = id;
      continue;
    }

    const mid = (min + max) / 2;
    let bestAxis = axes[0];
    let bestDist = Infinity;
    for (const axis of axes) {
      const dist = Math.abs(mid - (axis.min + axis.max) / 2);
      if (dist < bestDist) {
        bestDist = dist;
        bestAxis = axis;
      }
    }
    assignments[param] = bestAxis.id;
    bestAxis.min = Math.min(bestAxis.min, min);
    bestAxis.max = Math.max(bestAxis.max, max);
  }

  return { assignments, axes };
}

function buildScales(axes) {
  const scales = {
    x: {
      type: "time",
      title: { display: true, text: "Time", color: "#858697" },
      ticks: { color: "#858697" },
      grid: { color: "rgba(133, 134, 151, 0.12)" },
    },
  };
  axes.forEach((axis, i) => {
    scales[axis.id] = {
      type: "linear",
      position: AXIS_POSITIONS[i],
      offset: i === 2,
      title: { display: true, text: axis.id, color: "#858697" },
      ticks: { color: "#858697" },
      grid: { drawOnChartArea: i === 0, color: "rgba(133, 134, 151, 0.12)" },
    };
  });
  return scales;
}

// --- Shared render path ---

function renderSeriesDict(seriesDict, columns, statusLabel) {
  const seriesEntries = columns.map((param) => ({
    param,
    points: seriesToPoints(seriesDict[param] || []),
  }));

  const { assignments, axes } = computeAxisAssignments(seriesEntries);

  chart.data.datasets = seriesEntries.map(({ param, points }) => ({
    label: param,
    data: points,
    yAxisID: assignments[param],
    borderColor: colorForParam(param),
    backgroundColor: colorForParam(param),
    pointRadius: 0,
    borderWidth: 1.5,
    spanGaps: false,
  }));
  const scales = buildScales(axes);
  const range = getRequestWindow();
  scales.x.min = range.start.getTime();
  scales.x.max = range.end.getTime();
  chart.options.scales = scales;
  chart.update();

  const totalPoints = seriesEntries.reduce(
    (sum, s) => sum + s.points.length,
    0,
  );
  setStatus(`${columns.length} series, ${totalPoints} points — ${statusLabel}`);
}

// --- Client-side cache ---

const cache = { start: null, end: null, bucketSeconds: null, series: {} };

function canUseCache(start, end, columns) {
  if (!cache.start || !cache.end) return false;
  if (start < cache.start || end > cache.end) return false;
  return columns.every((col) => col in cache.series);
}

function sliceCacheSeries(start, end, columns) {
  const result = {};
  for (const col of columns) {
    result[col] = (cache.series[col] || []).filter((p) => {
      const t = new Date(p.t);
      return t >= start && t <= end;
    });
  }
  return result;
}

async function prefetchWeekCache() {
  try {
    const qualityResp = await fetch(`/api/data?preset=1w&columns=quality`);
    if (!qualityResp.ok) return;
    const qualityData = await qualityResp.json();
    cache.start = new Date(qualityData.start);
    cache.end = new Date(qualityData.end);
    cache.bucketSeconds = qualityData.bucket_seconds;
    cache.series.quality = qualityData.series.quality;
  } catch (err) {
    console.warn("Background prefetch (quality/1w) failed", err);
    return;
  }

  const others = window.ALL_PARAMETERS.filter((p) => p !== "quality");
  if (others.length === 0) return;

  try {
    const params = new URLSearchParams();
    others.forEach((c) => params.append("columns", c));
    params.append("preset", "1w");
    const othersResp = await fetch(`/api/data?${params.toString()}`);
    if (!othersResp.ok) return;
    const othersData = await othersResp.json();
    Object.assign(cache.series, othersData.series);
  } catch (err) {
    console.warn("Background prefetch (remaining params/1w) failed", err);
  }
}

// --- Fetching / rendering orchestration ---

function buildQueryParams(columns) {
  return buildUrlParams(columns).toString();
}

let fetchSeq = 0;

async function fetchAndRender() {
  const columns = getSelectedColumns();
  const seq = ++fetchSeq;

  if (columns.length === 0) {
    chart.data.datasets = [];
    chart.options.scales = { x: chart.options.scales.x };
    chart.update();
    setStatus("No parameters selected.");
    return;
  }

  const alwaysFresh =
    currentRange.mode === "preset" &&
    (currentRange.preset === "1h" || currentRange.preset === "12h");

  if (!alwaysFresh) {
    const { start, end } = getRequestWindow();
    if (canUseCache(start, end, columns)) {
      const sliced = sliceCacheSeries(start, end, columns);
      const idealBucket = estimateBucketSeconds(start, end);
      const label =
        idealBucket !== null && cache.bucketSeconds > idealBucket
          ? `cached, ${cache.bucketSeconds}s buckets (lower-res — refresh for finer detail)`
          : `cached, ${cache.bucketSeconds}s buckets`;
      renderSeriesDict(sliced, columns, label);
      return;
    }
  }

  setStatus("Loading...");
  try {
    const query = buildQueryParams(columns);
    const response = await fetch(`/api/data?${query}`);
    if (seq !== fetchSeq) return;

    if (!response.ok) {
      const body = await response.json().catch(() => ({}));
      throw new Error(body.detail || `Request failed (${response.status})`);
    }
    const data = await response.json();
    if (seq !== fetchSeq) return;

    const bucketLabel =
      data.bucket_seconds !== null
        ? `bucketed at ${data.bucket_seconds}s`
        : "raw resolution";
    renderSeriesDict(data.series, columns, bucketLabel);
  } catch (err) {
    if (seq !== fetchSeq) return;
    setStatus(`Error: ${err.message}`);
    console.error(err);
  }
}

// --- Event wiring ---

document.querySelectorAll(".preset-btn").forEach((btn) => {
  btn.addEventListener("click", () => {
    setPresetRange(btn.dataset.preset);
    fetchAndRender();
  });
});

let currentPanel = null;
const subnav = document.getElementById("subnav");

function closeSubnav() {
  subnav.classList.remove("open");
  subnav.setAttribute("aria-hidden", "true");
  document.querySelectorAll(".nav-link[aria-expanded]").forEach((link) => {
    link.setAttribute("aria-expanded", "false");
  });
  currentPanel = null;
}

function setActiveNav(link, panelId) {
  document.querySelectorAll(".nav-link").forEach((item) => {
    item.removeAttribute("aria-current");
    if (item.hasAttribute("aria-expanded"))
      item.setAttribute("aria-expanded", "false");
  });
  link.setAttribute("aria-current", "page");

  if (!panelId || panelId === "none") {
    closeSubnav();
    return;
  }

  if (currentPanel === panelId && subnav.classList.contains("open")) {
    closeSubnav();
    return;
  }

  document.querySelectorAll(".subnav-inner").forEach((panel) => {
    panel.hidden = panel.dataset.panel !== panelId;
  });
  subnav.classList.add("open");
  subnav.setAttribute("aria-hidden", "false");
  link.setAttribute("aria-expanded", "true");
  currentPanel = panelId;
}

document.querySelectorAll(".nav-link").forEach((link) => {
  link.addEventListener("click", (event) => {
    if (link.getAttribute("href") === "#") event.preventDefault();
    setActiveNav(link, link.dataset.panel);
  });
});

document.querySelectorAll(".subnav-link").forEach((link) => {
  link.addEventListener("click", (event) => {
    if (link.getAttribute("href") === "#") event.preventDefault();
    document
      .querySelectorAll(".subnav-link")
      .forEach((item) => item.removeAttribute("aria-current"));
    link.setAttribute("aria-current", "page");
  });
});

document
  .getElementById("subnav-close-btn")
  .addEventListener("click", closeSubnav);

document.getElementById("apply-range-btn").addEventListener("click", () => {
  const start = document.getElementById("start-input").value;
  const end = document.getElementById("end-input").value;
  if (!start || !end) {
    setStatus("Please provide both a start and end time.");
    return;
  }
  setCustomRange(start, end);
  fetchAndRender();
});

let checkboxDebounce = null;
document.querySelectorAll('input[name="columns"]').forEach((cb) => {
  cb.addEventListener("change", () => {
    clearTimeout(checkboxDebounce);
    checkboxDebounce = setTimeout(() => {
      updateUrl();
      fetchAndRender();
    }, 150);
  });
});

// Back/forward navigation: re-read the URL and re-render to match.
window.addEventListener("popstate", () => {
  if (!applyStateFromUrl()) {
    initRangeFromDom();
  }
  fetchAndRender();
});

// --- Init ---

document.querySelectorAll('input[name="columns"]').forEach((checkbox) => {
  checkbox.checked = checkbox.value === window.DEFAULT_PARAMETER;
});

const hadUrlState = applyStateFromUrl();
if (!hadUrlState) {
  initRangeFromDom();
}
fetchAndRender().then(() => {
  prefetchWeekCache();
});
