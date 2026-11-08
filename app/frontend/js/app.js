// Logique de la page. Toutes les données viennent de l'API :
//   /api/info     modèle, espèces, étapes, scores
//   /api/samples  images réelles du jeu de test
//   /api/predict/stream  prédiction étape par étape, en temps réel
//   /api/history  dernières analyses (base SQLite)
//   /api/stats    statistiques sur toutes les analyses
const $ = (s) => document.querySelector(s);
const $$ = (s) => [...document.querySelectorAll(s)];

const MAX_SIZE = 10 * 1024 * 1024;   // 10 Mo
const MIN_STEP_VISIBLE = 0.35;       // temps d'affichage minimal d'une étape (s), pour que l'œil suive
const RING_LENGTH = 2 * Math.PI * 52;

let INFO = null;
let SPECIES = {};
let currentFile = null;
let currentUrl = null;
let currentSample = null;            // nom du fichier si l'image vient des exemples
let currentView = $("#view-upload");
let modelInputUrl = null;            // image 224x224 renvoyée par le serveur

const pct = (v, digits = 1) => (v * 100).toLocaleString("fr-FR", { maximumFractionDigits: digits }) + "%";
const num = (v) => Number(v).toLocaleString("fr-FR");

async function api(path, options) {
  const r = await fetch(path, options);
  const data = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(data.detail || `Erreur du serveur (${r.status}).`);
  return data;
}

// ---------- Changement de vue animé ----------
function showView(next) {
  if (next === currentView) return Promise.resolve();
  return new Promise((resolve) => {
    gsap.to(currentView, {
      autoAlpha: 0, y: -8, duration: 0.2, ease: "power2.in",
      onComplete: () => {
        currentView.hidden = true;
        next.hidden = false;
        currentView = next;
        gsap.fromTo(next, { autoAlpha: 0, y: 12 },
          { autoAlpha: 1, y: 0, duration: 0.4, ease: "power3.out", onComplete: resolve });
      },
    });
  });
}

// ---------- Modèle : état, étapes, fiche ----------
async function loadInfo() {
  const status = $("#api-status");
  try {
    INFO = await api("/api/info");
  } catch {
    status.classList.add("off");
    status.querySelector(".status-text").textContent = "Hors ligne";
    return;
  }
  status.classList.add(INFO.pret ? "ok" : "off");
  status.querySelector(".status-text").textContent = INFO.pret ? "Modèle prêt" : "Modèle indisponible";
  status.title = INFO.pret ? INFO.modele : INFO.erreur || "";

  SPECIES = Object.fromEntries(INFO.especes.map((e) => [e.nom, e]));
  $("#steps").innerHTML = INFO.etapes.map((t) => `
    <li>
      <span class="dot"></span>
      <div class="step-body"><span class="step-label">${t}</span><span class="step-detail"></span></div>
      <span class="step-time"></span>
    </li>`).join("");

  // Scores réels, recalculés par scripts/prepare_data.py sur les images de validation et de test.
  const ev = INFO.evaluation;
  const facts = [["Architecture", INFO.modele]];
  if (INFO.caracteristiques) facts.push(["Couche Softmax", `${INFO.caracteristiques} → ${INFO.especes.length} classes`]);
  if (ev) {
    facts.push(["Accuracy (validation)", `${num(ev.validation.accuracy)}%`]);
    facts.push(["Accuracy (test)", `${num(ev.test.accuracy)}%`]);
    facts.push(["F1-score macro (test)", `${num(ev.test.f1_macro)}%`]);
    facts.push(["Jeu de test", `${num(ev.test.images)} images`]);
  } else {
    facts.push(["Évaluation", "non générée"]);
  }
  $("#model-facts").innerHTML = facts.map(([k, v]) => `<div><dt>${k}</dt><dd>${v}</dd></div>`).join("");
}

// ---------- Exemples réels du jeu de test ----------
// Un bouton par espèce (chaque clic passe à la photo suivante) + un bouton « Aléatoire ».
let SAMPLES = [];
const sampleCursor = {};

async function loadSamples() {
  try { SAMPLES = await api("/api/samples"); } catch { return; }
  if (!SAMPLES.length) return;
  const species = [...new Set(SAMPLES.map((s) => s.classe))];
  $("#samples-chips").innerHTML = species.map((name) => {
    const first = SAMPLES.find((s) => s.classe === name);
    return `<button class="chip" type="button" data-classe="${name}"><img src="${first.url}" alt="">${name}</button>`;
  }).join("") + `<button class="chip chip-random" type="button" data-classe="">
      <svg viewBox="0 0 24 24"><path d="M16 3h5v5M4 20L21 3M21 16v5h-5M15 15l6 6M4 4l5 5"/></svg>Aléatoire</button>`;
  $$("#samples-chips .chip").forEach((chip) => chip.addEventListener("click", () => pickSample(chip)));
  $("#samples").hidden = false;
}

function pickSample(chip) {
  const name = chip.dataset.classe;
  let sample;
  if (name) {
    const list = SAMPLES.filter((s) => s.classe === name);
    sampleCursor[name] = ((sampleCursor[name] ?? -1) + 1) % list.length;
    sample = list[sampleCursor[name]];
  } else {
    sample = SAMPLES[Math.floor(Math.random() * SAMPLES.length)];
  }
  selectSample(sample, chip);
}

async function selectSample(sample, chip) {
  try {
    const blob = await (await fetch(sample.url)).blob();
    const file = new File([blob], sample.fichier, { type: blob.type || "image/jpeg" });
    selectFile(file, sample.fichier);
    $$("#samples-chips .chip").forEach((c) => c.classList.toggle("selected", c === chip));
  } catch {
    uploadError("Impossible de charger cet exemple.");
  }
}

// ---------- Sélection de l'image ----------
const dropzone = $("#dropzone");
const fileInput = $("#file-input");

fileInput.addEventListener("change", () => fileInput.files[0] && selectFile(fileInput.files[0]));
["dragenter", "dragover"].forEach((ev) =>
  dropzone.addEventListener(ev, (e) => { e.preventDefault(); dropzone.classList.add("over"); }));
["dragleave", "drop"].forEach((ev) =>
  dropzone.addEventListener(ev, (e) => { e.preventDefault(); dropzone.classList.remove("over"); }));
dropzone.addEventListener("drop", (e) => e.dataTransfer.files[0] && selectFile(e.dataTransfer.files[0]));

function formatSize(bytes) {
  return bytes < 1024 * 1024 ? (bytes / 1024).toFixed(0) + " Ko" : (bytes / 1024 / 1024).toFixed(1) + " Mo";
}

function uploadError(msg) {
  const el = $("#upload-error");
  el.textContent = msg;
  el.hidden = false;
  gsap.fromTo(dropzone, { x: -6 }, { x: 0, duration: 0.4, ease: "elastic.out(1, 0.3)" });
}

function selectFile(file, sampleName = null) {
  $("#upload-error").hidden = true;
  if (!file.type.startsWith("image/")) return uploadError("Ce fichier n'est pas une image.");
  if (file.size > MAX_SIZE) return uploadError("L'image dépasse la taille maximale de 10 Mo.");

  currentFile = file;
  currentSample = sampleName;
  if (!sampleName) $$("#samples-chips .chip").forEach((c) => c.classList.remove("selected"));
  if (currentUrl) URL.revokeObjectURL(currentUrl);
  currentUrl = URL.createObjectURL(file);

  const preview = $("#drop-preview");
  preview.src = currentUrl;
  preview.hidden = false;
  $("#drop-empty").hidden = true;
  dropzone.classList.add("has-image");
  dropzone.style.setProperty("--preview", `url("${currentUrl}")`);

  $("#file-name").textContent = file.name;
  $("#file-size").textContent = formatSize(file.size) + (sampleName ? " · jeu de test DeepWeeds" : "");
  const bar = $("#file-bar");
  if (bar.hidden) {
    bar.hidden = false;
    gsap.fromTo(bar, { autoAlpha: 0, y: 6 }, { autoAlpha: 1, y: 0, duration: 0.3 });
  }
  gsap.fromTo(preview, { autoAlpha: 0 }, { autoAlpha: 1, duration: 0.4 });
}

$("#btn-change").addEventListener("click", () => fileInput.click());
$("#btn-analyze").addEventListener("click", analyze);
$("#btn-restart").addEventListener("click", reset);
$("#btn-retry").addEventListener("click", reset);

// ---------- Analyse en temps réel ----------
// Le serveur renvoie une ligne JSON à la fin de chaque étape (durée et détail mesurés).
async function analyze() {
  if (!currentFile) return;

  $("#scan-img").src = currentUrl;
  $("#model-input").hidden = true;
  $("#input-size").textContent = "en attente du serveur…";
  modelInputUrl = null;
  resetSteps();
  await showView($("#view-analyzing"));

  const scan = gsap.fromTo(".scan-line", { top: "0%" },
    { top: "100%", duration: 1.4, ease: "sine.inOut", repeat: -1, yoyo: true });
  const stopTimer = startTimer();
  setStepActive(0);

  const form = new FormData();
  form.append("file", currentFile);
  if (currentSample) form.append("exemple", currentSample);

  try {
    const result = await streamPrediction(form);
    stopTimer();
    scan.kill();
    showResult(result);
  } catch (err) {
    stopTimer();
    scan.kill();
    $("#error-msg").textContent = err.message === "Failed to fetch"
      ? "Le serveur ne répond pas. Vérifiez que l'API est démarrée."
      : err.message;
    showView($("#view-error"));
  }
}

async function streamPrediction(form) {
  const r = await fetch("/api/predict/stream", { method: "POST", body: form });
  if (!r.ok) {
    const data = await r.json().catch(() => ({}));
    throw new Error(data.detail || `Erreur du serveur (${r.status}).`);
  }
  const reader = r.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let display = Promise.resolve();   // file d'affichage des étapes
  let result = null;

  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let nl;
    while ((nl = buffer.indexOf("\n")) >= 0) {
      const line = buffer.slice(0, nl).trim();
      buffer = buffer.slice(nl + 1);
      if (!line) continue;
      const event = JSON.parse(line);
      if (event.erreur) throw new Error(event.erreur);
      if (event.resultat) result = event.resultat;
      else display = display.then(() => completeStep(event));
    }
  }
  await display;
  if (!result) throw new Error("Réponse incomplète du serveur.");
  return result;
}

function resetSteps() {
  $$("#steps li").forEach((li) => {
    li.classList.remove("active", "done");
    li.querySelector(".step-detail").textContent = "";
    li.querySelector(".step-time").textContent = "";
  });
  gsap.set("#progress-bar", { width: "0%" });
}

function setStepActive(i) {
  const li = $$("#steps li")[i];
  if (li) li.classList.add("active");
}

let lastStepShown = 0;
async function completeStep(event) {
  // Laisse chaque étape visible un court instant : les durées affichées restent les vraies.
  const wait = MIN_STEP_VISIBLE * 1000 - (performance.now() - lastStepShown);
  if (wait > 0) await new Promise((res) => setTimeout(res, wait));
  lastStepShown = performance.now();

  const steps = $$("#steps li");
  const li = steps[event.etape];
  if (!li) return;
  li.classList.remove("active");
  li.classList.add("done");
  li.querySelector(".step-detail").textContent = event.detail || "";
  li.querySelector(".step-time").textContent = `${num(event.duree_ms)} ms`;
  gsap.fromTo(li.querySelector(".step-detail"), { autoAlpha: 0, y: -3 }, { autoAlpha: 1, y: 0, duration: 0.25 });

  if (event.apercu) {
    modelInputUrl = event.apercu;
    const img = $("#model-input");
    img.src = event.apercu;
    img.hidden = false;
    $("#input-size").textContent = "image reçue par le modèle";
    gsap.fromTo(img, { autoAlpha: 0, scale: 0.9 }, { autoAlpha: 1, scale: 1, duration: 0.3 });
  }
  gsap.to("#progress-bar", { width: `${((event.etape + 1) / steps.length) * 100}%`, duration: 0.3, ease: "power1.out" });
  setStepActive(event.etape + 1);
}

// Chronomètre réel, de l'envoi de l'image jusqu'à la réception du résultat.
function startTimer() {
  const t0 = performance.now();
  lastStepShown = t0;
  let running = true;
  const el = $("#elapsed");
  const tick = () => {
    if (!running) return;
    el.textContent = ((performance.now() - t0) / 1000).toLocaleString("fr-FR", { minimumFractionDigits: 2, maximumFractionDigits: 2 }) + " s";
    requestAnimationFrame(tick);
  };
  tick();
  return () => { running = false; };
}

// ---------- Résultat ----------
async function showResult(data) {
  const sp = SPECIES[data.classe] || {};
  const conf = data.confiance;

  $("#pred-class").textContent = data.classe;
  $("#pred-latin").textContent = sp.latin || "";
  $("#pred-desc").textContent = sp.description || "";
  $("#low-conf").hidden = conf >= 0.6;
  $("#meta").textContent = `Temps d'inférence : ${num(data.temps_ms)} ms`;

  const truth = $("#truth");
  truth.hidden = !data.classe_reelle;
  if (data.classe_reelle) {
    const ok = data.classe_reelle === data.classe;
    truth.className = "truth " + (ok ? "ok" : "ko");
    truth.textContent = ok
      ? `Prédiction correcte · classe réelle : ${data.classe_reelle}`
      : `Erreur du modèle · classe réelle : ${data.classe_reelle}`;
  }

  $("#res-img").src = currentUrl;
  $("#res-input").src = modelInputUrl || currentUrl;

  // Barres de probabilités.
  const bars = $("#bars");
  bars.innerHTML = "";
  Object.entries(data.probabilites)
    .sort((a, b) => b[1] - a[1])
    .forEach(([name, p]) => {
      const row = document.createElement("div");
      row.className = "bar-row" + (name === data.classe ? " win" : "");
      row.innerHTML = `
        <div class="bar-label"><span>${name}</span><span class="pct">0%</span></div>
        <div class="bar-track"><div class="bar-fill" data-p="${p}"></div></div>`;
      bars.appendChild(row);
    });


  const ring = $("#ring-fg");
  ring.style.stroke = conf >= 0.75 ? "var(--accent)" : conf >= 0.5 ? "var(--amber)" : "var(--red)";
  gsap.set(ring, { strokeDashoffset: RING_LENGTH });
  $("#conf-value").textContent = "0%";

  await showView($("#view-result"));
  refreshHistory();
  refreshStats();

  const tl = gsap.timeline();
  tl.from(".result-title > :not([hidden])", { y: 10, autoAlpha: 0, stagger: 0.07, duration: 0.35 })
    .to(ring, { strokeDashoffset: RING_LENGTH * (1 - conf), duration: 1.1, ease: "power3.out" }, 0.1)
    .to({ v: 0 }, {
      v: conf, duration: 1.1, ease: "power3.out",
      onUpdate() { $("#conf-value").textContent = pct(this.targets()[0].v); },
    }, 0.1);

  $$(".bar-fill").forEach((bar, i) => {
    const p = parseFloat(bar.dataset.p);
    const label = bar.closest(".bar-row").querySelector(".pct");
    tl.to(bar, { width: (p * 100).toFixed(1) + "%", duration: 0.8, ease: "power2.out" }, 0.3 + i * 0.1);
    tl.to({ v: 0 }, {
      v: p, duration: 0.8, ease: "power2.out",
      onUpdate() { label.textContent = pct(this.targets()[0].v); },
    }, 0.3 + i * 0.1);
  });
}

// ---------- Historique (base SQLite côté serveur) ----------
async function refreshHistory() {
  let items = [];
  try { items = await api("/api/history?limit=5"); } catch { return; }
  $("#history-empty").hidden = items.length > 0;
  $("#btn-clear").hidden = items.length === 0;
  $("#history").innerHTML = items.map((h) => {
    const time = new Date(h.date).toLocaleString("fr-FR", { day: "2-digit", month: "2-digit", hour: "2-digit", minute: "2-digit" });
    const verdict = h.classe_reelle
      ? `<span class="h-tag ${h.classe_reelle === h.classe ? "ok" : "ko"}">${h.classe_reelle === h.classe ? "correct" : "erreur"}</span>`
      : "";
    return `<li>
      ${h.image ? `<img src="${h.image}" alt="">` : `<div class="no-img"></div>`}
      <div class="h-main"><p class="h-name">${h.classe} ${verdict}</p><p class="h-time">${time}</p></div>
      <span class="h-conf">${pct(h.confiance, 0)}</span>
    </li>`;
  }).join("");
}

$("#btn-clear").addEventListener("click", async () => {
  try { await api("/api/history", { method: "DELETE" }); } catch { return; }
  refreshHistory();
  refreshStats();
});

// ---------- Statistiques ----------
async function refreshStats() {
  let s;
  try { s = await api("/api/stats"); } catch { return; }
  $("#st-total").textContent = num(s.total);
  $("#st-conf").textContent = s.total ? pct(s.confiance_moyenne, 0) : "0%";
  $("#st-classes").innerHTML = Object.entries(s.par_classe).map(([name, n]) => {
    const share = s.total ? n / s.total : 0;
    return `<div class="st-class">
      <div class="st-label"><span>${name}</span><span>${n}</span></div>
      <div class="bar-track"><div class="bar-fill" style="width:${(share * 100).toFixed(1)}%"></div></div>
    </div>`;
  }).join("");
  const samples = $("#st-samples");
  samples.hidden = !s.exemples_testes;
  samples.textContent = `Exemples du jeu de test : ${s.exemples_corrects} / ${s.exemples_testes} corrects`;
}

// ---------- Nouvelle analyse ----------
function reset() {
  currentFile = null;
  currentSample = null;
  fileInput.value = "";
  $("#drop-preview").hidden = true;
  $("#drop-empty").hidden = false;
  dropzone.classList.remove("has-image");
  dropzone.style.removeProperty("--preview");
  $("#file-bar").hidden = true;
  $("#upload-error").hidden = true;
  $$("#samples-chips .chip").forEach((c) => c.classList.remove("selected"));
  showView($("#view-upload"));
}

// ---------- Démarrage ----------
loadInfo();
loadSamples();
refreshHistory();
refreshStats();
