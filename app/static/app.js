const ORDER = ["clock", "motor", "eye", "speech"];
const NAMES = { clock: "Clock", motor: "Motor", eye: "Eye", speech: "Speech" };
const TAGS = { flagged: "Flagged", not_flagged: "Not flagged", not_run: "Not completed" };
const $ = id => document.getElementById(id);
const el = (tag, cls, text) => { const e = document.createElement(tag); if (cls) e.className = cls; if (text !== undefined) e.textContent = text; return e; };
const spin = (box, text) => { box.replaceChildren(el("span", "spinner"), text); };
async function post(url, data) {
  const res = await fetch(url, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(data) });
  const j = await res.json().catch(() => ({ error: "bad reply" }));
  if (!res.ok || j.error) throw new Error(j.error || res.statusText);
  return j;
}
const offline = m => m === "Failed to fetch" ? "the app is not running. Start run_app.bat and keep its window open." : m;
const b64 = async blob => { const bytes = new Uint8Array(await blob.arrayBuffer()); let s = ""; for (let i = 0; i < bytes.length; i += 32768) s += String.fromCharCode(...bytes.subarray(i, i + 32768)); return btoa(s); };
const pretty = v => Array.isArray(v) ? (v.join(", ") || "none") : (v && typeof v === "object" ? Object.entries(v).map(e => e.join(": ")).join(", ") : v);
const css = name => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const copy = async (text, btn) => { try { await navigator.clipboard.writeText(text); const o = btn.textContent; btn.textContent = "Copied"; setTimeout(() => { btn.textContent = o; }, 1400); } catch {} };

/* ---------- theme ---------- */
$("theme").onclick = () => {
  const t = document.documentElement.dataset.theme === "dark" ? "light" : "dark"; document.documentElement.dataset.theme = t;
  try { localStorage.setItem("theme", t); } catch {}
  if (typeof redrawCharts === "function") redrawCharts();
};

/* ---------- views ---------- */
let view = "landing";
function show(v) { view = v; ["landing", "patient", "doctor"].forEach(x => { $("view-" + x).hidden = x !== v; }); window.scrollTo(0, 0); updateScene(); }
function updateScene() {   // the moving background belongs to the sign-up / log-in screens only
  const onDetails = view === "patient" && !$("step-details").hidden, onAuth = view === "doctor" && !$("doc-auth").hidden;
  document.body.classList.toggle("has-scene", onDetails || onAuth);
}
$("brand").onclick = () => {
  if (view !== "patient" || !patient.code || !$("step-final").hidden) return location.reload();   // nothing unsaved to lose
  if ($("leave")) return;   // in-page confirmation (browser confirm() popups can be blocked)
  const bar = el("div", "note"); bar.id = "leave"; bar.setAttribute("role", "alert");
  const stay = el("button", "secondary", "Stay"), go = el("button", "", "Leave and lose results");
  stay.onclick = () => bar.remove(); go.onclick = () => location.reload();
  const row = el("div", "row"); row.append(stay, go); bar.append(el("p", "", "Leave now? The tests you have not finished will be lost."), row);
  $("banner").before(bar);
};
$("brand").onkeydown = e => { if (e.key === "Enter") $("brand").onclick(); };
$("go-patient").onclick = () => { show("patient"); setStep("details"); };
$("go-doctor").onclick = () => { show("doctor"); authPanel("login"); };

/* ---------- result rendering (patient steps; the doctor view builds on it) ---------- */
function resultBlock(r) {
  const box = el("div");
  const head = el("div", "row"); head.style.margin = "0"; head.append(el("span", "tag " + r.status, TAGS[r.status]), el("b", "", r.label || ""));
  box.append(head);
  const keys = Object.keys(r.measured || {});
  if (keys.length) { const dl = el("dl"); keys.forEach(k => dl.append(el("dt", "", k), el("dd", "", pretty(r.measured[k])))); box.append(dl); }
  return box;
}

/* ---------- patient state & stepper ---------- */
const STEPS = [["details", "About you"], ["clock", "Clock"], ["motor", "Tapping"], ["eye", "Eyes"], ["speech", "Speech"], ["final", "Results"]];
const patient = { code: null, name: "", age: 70 };
const results = {}, art = {}; let clockJpeg = null;   // art[m] = what the doctor can replay later (drawing view, tap times, gaze path, audio)
function setStep(id) {
  STEPS.forEach(([s]) => { $("step-" + s).hidden = s !== id; });
  const idx = STEPS.findIndex(s => s[0] === id), st = $("stepper"); st.replaceChildren();
  STEPS.forEach(([, label], i) => { const d = el("div", "s" + (i < idx ? " done" : i === idx ? " now" : "")); d.append(el("div", "bar"), el("span", "", label)); st.append(d); });
  window.scrollTo({ top: 0, behavior: "smooth" });
  if (id === "speech") loadPicture();
  if (id === "final") finish();
  updateScene();
}
function setResult(m, r, artifact) {
  const { preview, transcript, ...clean } = r; results[m] = clean; art[m] = artifact || null;
  const box = $("res-" + m); box.replaceChildren(resultBlock(clean));
  if (preview) { const f = el("figure"), img = el("img", "preview"); img.src = "data:image/png;base64," + preview; img.alt = "The drawing as the model saw it"; f.append(img, el("figcaption", "", "What the model saw")); box.append(f); }
  box.hidden = false; $(m + "-next").hidden = false;
}
const next = { clock: "motor", motor: "eye", eye: "speech", speech: "final" };
ORDER.forEach(m => { $(m + "-next").onclick = () => setStep(next[m]); });
$("eye-skip").onclick = () => setStep("speech"); $("speech-skip").onclick = () => setStep("final");
$("pid-copy").onclick = () => copy(patient.code, $("pid-copy"));

/* ---------- registration form (every answer required) ---------- */
const YN = ["No", "Yes", "Not sure"];
const FIELDS = [
  ["Basics", [["name", "Full name", "text"], ["age", "Age", "number", { min: 18, max: 110 }],
    ["gender", "Gender", ["Female", "Male", "Other", "Prefer not to say"]],
    ["ethnicity", "Ethnicity", ["Asian", "Black or African", "Hispanic or Latino", "White", "Mixed or multiple", "Other", "Prefer not to say"]],
    ["education", "Education", ["Less than high school", "High school", "Some college", "Bachelor's degree", "Postgraduate"]]]],
  ["Health", [["systolic_bp", "Blood pressure, top number (systolic)", "number", { min: 70, max: 250, placeholder: "e.g. 130" }],
    ["diastolic_bp", "Blood pressure, bottom number (diastolic)", "number", { min: 40, max: 150, placeholder: "e.g. 80" }],
    ["diabetes", "Diabetes", ["No", "Type 2", "Type 1", "Not sure"]], ["high_blood_pressure", "Told you have high blood pressure?", YN],
    ["high_cholesterol", "High cholesterol?", YN], ["family_history_dementia", "Parent or sibling with dementia?", YN],
    ["head_injury", "Serious head injury?", ["No", "Yes"]], ["hearing_loss", "Hearing loss?", ["No", "Yes"]], ["depression", "Depression or low mood?", ["No", "Yes"]]]],
  ["Lifestyle", [["smoking", "Smoking", ["Never", "Former", "Current"]], ["physical_activity", "Physical activity", ["Rarely", "1–2 days a week", "3 or more days a week"]]]],
];
const qInputs = {};
(() => {
  const host = $("q-fields");
  FIELDS.forEach(([title, items]) => {
    host.append(el("div", "group", title)); const grid = el("div", "grid");
    items.forEach(([key, label, type, opt]) => {
      const wrap = el("div"), lab = el("label"), id = "q-" + key; lab.htmlFor = id; lab.append(label + " ", el("i", "", "*"));
      let inp;
      if (Array.isArray(type)) { inp = el("select"); inp.append(new Option("Select…", "")); type.forEach(o => inp.append(new Option(o, o))); }
      else { inp = el("input"); inp.type = type; Object.assign(inp, opt || {}); if (type === "number") inp.inputMode = "numeric"; }
      inp.id = id; inp.oninput = inp.onchange = () => inp.classList.remove("invalid");
      qInputs[key] = { inp, label }; wrap.append(lab, inp); grid.append(wrap);
    });
    host.append(grid);
  });
})();
$("q-form").onsubmit = async e => {
  e.preventDefault(); const btn = $("q-submit"), msg = $("q-msg"); msg.textContent = "";
  const missing = [], range = [];
  Object.entries(qInputs).forEach(([k, { inp, label }]) => {
    inp.classList.remove("invalid"); const v = inp.value.trim();
    if (!v) { missing.push(label.replace(/ \(.*\)$/, "")); inp.classList.add("invalid"); return; }
    if (inp.type === "number" && (Number(v) < Number(inp.min) || Number(v) > Number(inp.max) || !Number.isFinite(Number(v)))) { range.push(`${label.replace(/ \(.*\)$/, "")} must be ${inp.min}–${inp.max}`); inp.classList.add("invalid"); }
  });
  if (missing.length || range.length) {
    msg.textContent = (missing.length ? "Please fill in: " + missing.join(", ") + ". " : "") + range.join(". ");
    const first = document.querySelector("#q-fields .invalid"); if (first) first.focus(); return;
  }
  btn.disabled = true;
  const q = {}; Object.entries(qInputs).forEach(([k, { inp }]) => { if (k !== "name") q[k] = inp.value.trim(); });
  try {
    const name = qInputs.name.inp.value.trim(), { code } = await post("/api/patient", { name, questionnaire: q });
    Object.assign(patient, { code, name, age: Number(q.age) });
    $("pid-text").textContent = code; $("pid-name").textContent = name; $("idbar").hidden = false; setStep("clock");
  } catch (err) { msg.textContent = "Could not save your details: " + offline(err.message); }
  btn.disabled = false;
};

/* ---------- clock: draw on screen, redraw smoothly, send as JPEG ---------- */
(() => {
  const cv = $("clock-canvas"), ctx = cv.getContext("2d"), W = cv.width; let strokes = [], cur = null;
  const LINE = W * 0.0058;   // ~4.8 px at 828 px: a pen-like line, close to the scale of the scanned training drawings
  function paint(c) {
    c.fillStyle = "#fff"; c.fillRect(0, 0, W, W); c.lineWidth = LINE; c.lineCap = c.lineJoin = "round"; c.strokeStyle = "#000"; c.fillStyle = "#000";
    for (const s of strokes) {
      if (s.length === 1) { c.beginPath(); c.arc(s[0][0], s[0][1], LINE / 2, 0, 7); c.fill(); continue; }
      c.beginPath(); c.moveTo(...s[0]);
      for (let i = 1; i < s.length - 1; i++) c.quadraticCurveTo(...s[i], (s[i][0] + s[i + 1][0]) / 2, (s[i][1] + s[i + 1][1]) / 2);   // smooth through midpoints
      c.lineTo(...s[s.length - 1]); c.stroke();
    }
  }
  const redraw = () => paint(ctx); redraw();
  const pos = e => { const r = cv.getBoundingClientRect(); return [(e.clientX - r.left) * W / r.width, (e.clientY - r.top) * W / r.height]; };
  cv.onpointerdown = e => { cur = [pos(e)]; strokes.push(cur); try { cv.setPointerCapture(e.pointerId); } catch {} redraw(); };
  cv.onpointermove = e => {
    if (!cur) return;
    for (const ev of (e.getCoalescedEvents ? e.getCoalescedEvents() : [e])) { const p = pos(ev), l = cur[cur.length - 1]; if (Math.hypot(p[0] - l[0], p[1] - l[1]) > 1.5) cur.push(p); }
    redraw();
  };
  cv.onpointerup = cv.onpointercancel = () => { cur = null; };
  $("clock-clear").onclick = () => { strokes = []; redraw(); };
  $("clock-undo").onclick = () => { strokes.pop(); redraw(); };
  $("clock-done").onclick = async () => {
    const msg = $("clock-msg"), btn = $("clock-done");
    if (!strokes.length) { msg.textContent = "Please draw the clock first."; return; }
    btn.disabled = true; spin(msg, "Checking the drawing… (the first time takes a few seconds longer)");
    try {
      const blob = await new Promise(r => cv.toBlob(r, "image/jpeg", .95)); clockJpeg = await b64(blob);   // the drawing is saved as a JPEG, then analysed from that file
      const r = await post("/api/clock", { image: clockJpeg });
      msg.textContent = r.status === "not_run" ? "That drawing could not be read. Clear it and draw a little larger and darker." : "";
      setResult("clock", r, { preview: r.preview });
    } catch (e) { msg.textContent = "Could not check the drawing: " + offline(e.message); }
    btn.disabled = false;
  };
})();

/* ---------- motor ---------- */
(() => {
  const timer = $("timer"), tap = $("tap"), start = $("motor-start"), bar = $("motor-bar");
  start.onclick = () => {
    start.disabled = true; let n = 3; timer.textContent = n; bar.style.width = "0";
    const cd = setInterval(() => { n--; if (n > 0) { timer.textContent = n; return; } clearInterval(cd); run(); }, 1000);
  };
  function run() {
    const taps = [], t0 = performance.now(); tap.disabled = false; tap.focus(); tap.classList.remove("secondary");
    const on = () => taps.push((performance.now() - t0) / 1000); tap.addEventListener("pointerdown", on);
    const tick = setInterval(async () => {
      const left = 30 - (performance.now() - t0) / 1000; timer.textContent = Math.max(0, Math.ceil(left)) + " s"; bar.style.width = Math.min(100, (30 - left) / 30 * 100) + "%";
      if (left > 0) return;
      clearInterval(tick); tap.disabled = true; tap.classList.add("secondary"); tap.removeEventListener("pointerdown", on); timer.textContent = "Done";
      const kept = taps.filter(t => t <= 30);
      try { setResult("motor", await post("/api/motor", { taps: kept, age: patient.age }), { taps: kept.map(t => Math.round(t * 1000) / 1000), age: patient.age }); }
      catch (e) { timer.textContent = "Could not score: " + e.message; }
      start.disabled = false; start.textContent = "Repeat";
    }, 200);
  }
})();

/* ---------- eye ---------- */
$("eye-start").onclick = async () => {
  const btn = $("eye-start"), msg = $("eye-msg"); btn.disabled = true;
  try {
    const { startEyeTest } = await import("/static/eye.js");
    await startEyeTest(async (tasks, cal, screen) => { setResult("eye", await post("/api/eye", { tasks, calibration: cal }), { tasks, screen }); }, t => { msg.textContent = t; });
  } catch (e) { msg.textContent = "Could not start the camera test: " + (e.name === "NotAllowedError" ? "camera permission was denied." : e.message); }
  btn.disabled = false;
};

/* ---------- speech ---------- */
function loadPicture() {
  const slot = $("picture-slot"); if (slot.childElementCount) return;
  const img = el("img", "picture"); img.alt = "Picture to describe"; const tries = ["/static/cookie_theft.jpg", "/static/cookie_theft.png"];
  img.onerror = () => { const t = tries.shift(); if (t) img.src = t; else slot.replaceChildren(el("div", "placeholder", "The picture file is missing. Put it at app/static/cookie_theft.jpg and reload.")); };
  img.onerror();
  slot.append(img);
}
$("speech-start").onclick = async () => {
  const btn = $("speech-start"), stop = $("speech-stop"), msg = $("speech-msg"), bar = $("speech-bar"); btn.disabled = true;
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    const rec = new MediaRecorder(stream, { audioBitsPerSecond: 48000 }), chunks = []; rec.ondataavailable = e => chunks.push(e.data);
    const stopped = new Promise(r => { rec.onstop = r; });
    rec.start(); let left = 60; stop.hidden = false; stop.onclick = () => rec.state === "recording" && rec.stop(); msg.textContent = "Recording… 60 s left";
    const tick = setInterval(() => { left--; msg.textContent = "Recording… " + left + " s left"; bar.style.width = (60 - left) / 60 * 100 + "%"; if (left <= 0 && rec.state === "recording") rec.stop(); }, 1000);
    await stopped; clearInterval(tick); stop.hidden = true; stream.getTracks().forEach(t => t.stop());
    const blob = new Blob(chunks, { type: rec.mimeType }), audio = await b64(blob);
    spin(msg, "Analysing your speech… (the first time can take a few minutes)");
    const r = await post("/api/speech", { audio }); msg.textContent = "";
    setResult("speech", r, { audio, mime: rec.mimeType || "audio/webm", transcript: r.transcript });
  } catch (e) { msg.textContent = "Could not run the speech test: " + (e.name === "NotAllowedError" ? "microphone permission was denied." : offline(e.message)); }
  btn.disabled = false; stop.hidden = true;
};

/* ---------- final: summary + save ---------- */
async function finish() {
  const wait = $("final-wait"), body = $("final-body"), msg = $("final-msg"); body.replaceChildren(); body.hidden = true; wait.hidden = false;
  const all = ORDER.map(m => results[m] || { modality: m, status: "not_run", label: NAMES[m] + ": not completed", measured: {}, evidence: "", caveat: "" });
  const t0 = performance.now(), tick = setInterval(() => { msg.textContent = "Writing your summary… " + Math.round((performance.now() - t0) / 1000) + " s (up to 2 minutes)"; }, 1000);
  let data = null, saved = "";
  try { data = await post("/api/summary", { results: all }); } catch (e) { body.append(el("div", "note", "Could not write the summary: " + e.message)); }
  try {
    const withArt = all.map(r => art[r.modality] ? { ...r, artifact: art[r.modality] } : r);
    const s = await post("/api/save", { code: patient.code, results: withArt, combined: data && data.combined, summary: data && data.summary, clock_jpeg: clockJpeg }); saved = "Saved to " + s.where + ".";
  } catch (e) { saved = "Could NOT be saved: " + offline(e.message); }
  clearInterval(tick); wait.hidden = true; body.hidden = false;
  body.append(el("p", "muted small", saved + " Your patient ID is " + patient.code + ". Give it to your doctor."));
  if (data) body.append(el("div", "overall" + (data.combined.follow_up ? " follow" : ""), data.combined.text));
  const tiles = el("div", "tiles");
  all.forEach(r => {
    const t = el("div", "tile"), h = el("h3"); h.append(el("span", "", NAMES[r.modality]), el("span", "tag " + r.status, TAGS[r.status])); t.append(h, el("div", "small", r.label));
    const keys = Object.keys(r.measured || {}).filter(k => k !== "outside the range" || r.measured[k].length);
    if (keys.length) { const dl = el("dl"); keys.forEach(k => dl.append(el("dt", "", k), el("dd", "", pretty(r.measured[k])))); t.append(dl); }
    if (data) t.append(el("p", "says muted", data.summary.modalities[r.modality]));
    tiles.append(t);
  });
  body.append(tiles, el("p", "muted small", "A screening aid, not a diagnosis. Your doctor will review these results with you."));
  const row = el("div", "row"), pr = el("button", "secondary", "Print"), again = el("button", "", "Finish"); pr.onclick = () => window.print(); again.onclick = () => location.reload(); row.append(pr, el("span", "spacer"), again); body.append(row);
}

/* ---------- doctor: log in / sign up / forgot password ---------- */
let token = null, doctorName = "";
function authPanel(name) {
  ["login", "signup", "forgot", "code"].forEach(p => { $("auth-" + p).hidden = p !== name; });
  ["login-msg", "signup-msg", "forgot-msg"].forEach(i => { $(i).textContent = ""; });
  $("doc-auth").hidden = false; $("doc-main").hidden = true; updateScene();
  const f = { login: "l-user", signup: "s-name", forgot: "f-user" }[name]; if (f) $(f).focus();
}
document.querySelectorAll("[data-go]").forEach(b => { b.onclick = () => authPanel(b.dataset.go); });
document.querySelectorAll(".show-pw").forEach(b => { b.onclick = () => { const i = $(b.dataset.for), on = i.type === "password"; i.type = on ? "text" : "password"; b.textContent = on ? "Hide" : "Show"; }; });
$("s-pass").oninput = () => { const p = $("s-pass").value; const score = Math.min(4, (p.length >= 8) + (p.length >= 12) + (/[A-Z]/.test(p) && /[a-z]/.test(p)) + (/\d/.test(p) || /\W/.test(p))); $("s-strength").style.width = score * 25 + "%"; };
function enterDoctor(name) { doctorName = name || doctorName; $("doc-auth").hidden = true; $("doc-main").hidden = false; $("doc-hello").textContent = doctorName ? "Welcome, " + doctorName : "Find a patient"; updateScene(); $("d-q").focus(); }
function showCode(code) { $("recovery-code").textContent = code; authPanel("code"); $("code-done").onclick = () => enterDoctor(); $("code-copy").onclick = () => copy(code, $("code-copy")); }
const authSubmit = (formId, msgId, fn) => { $(formId).onsubmit = async e => { e.preventDefault(); const msg = $(msgId); msg.textContent = ""; const btn = e.submitter; if (btn) btn.disabled = true;
  try { await fn(); } catch (err) { msg.textContent = offline(err.message); } if (btn) btn.disabled = false; }; };
authSubmit("login-form", "login-msg", async () => {
  if (!$("l-user").value.trim() || !$("l-pass").value) throw new Error("Enter your username and password.");
  const r = await post("/api/doctor/login", { user: $("l-user").value, password: $("l-pass").value }); token = r.token; enterDoctor(r.name);
});
authSubmit("signup-form", "signup-msg", async () => {
  const v = id => $(id).value;
  if (!v("s-name").trim() || !v("s-user").trim() || !v("s-pass") || !v("s-pass2")) throw new Error("Please fill in every box.");
  if (v("s-pass") !== v("s-pass2")) throw new Error("The two passwords do not match.");
  const r = await post("/api/doctor/signup", { full_name: v("s-name"), user: v("s-user"), password: v("s-pass"), confirm: v("s-pass2") }); token = r.token; doctorName = r.name; showCode(r.recovery_code);
});
authSubmit("forgot-form", "forgot-msg", async () => {
  const v = id => $(id).value;
  if (!v("f-user").trim() || !v("f-code").trim() || !v("f-pass") || !v("f-pass2")) throw new Error("Please fill in every box.");
  if (v("f-pass") !== v("f-pass2")) throw new Error("The two passwords do not match.");
  const r = await post("/api/doctor/reset", { user: v("f-user"), recovery_code: v("f-code"), password: v("f-pass"), confirm: v("f-pass2") }); token = r.token; doctorName = v("f-user").trim(); showCode(r.recovery_code);
});
$("doc-logout").onclick = () => { token = null; $("record").replaceChildren(); $("hits").replaceChildren(); $("d-q").value = ""; ["l-pass"].forEach(i => { $(i).value = ""; }); authPanel("login"); };
$("doc-recovery").onclick = async () => { try { showCode((await post("/api/doctor/new_recovery", { token })).recovery_code); } catch (e) { $("hits").replaceChildren(el("p", "err", e.message)); } };

/* ---------- doctor: live search ---------- */
let searchSeq = 0, searchTimer = null;
$("d-q").oninput = () => {
  clearTimeout(searchTimer); const q = $("d-q").value.trim(), hits = $("hits");
  if (!q) { hits.replaceChildren(); return; }
  searchTimer = setTimeout(async () => {
    const my = ++searchSeq;
    try {
      const { patients } = await post("/api/doctor/search", { token, q }); if (my !== searchSeq) return;
      hits.replaceChildren(); if (!patients.length) hits.append(el("p", "muted", "No patient found."));
      patients.forEach(p => { const b = el("button", "hit"); b.append(el("span", "", p.name), el("span", "badge", p.code)); b.onclick = () => openRecord(p.code); hits.append(b); });
    } catch (err) { hits.replaceChildren(el("p", "err", err.message)); }
  }, 220);
};
$("d-q").onkeydown = e => { if (e.key === "Enter") { const first = document.querySelector("#hits .hit"); if (first) first.click(); } };

/* ---------- doctor: full record (scores + what the patient drew / said / did) ---------- */
const QLABEL = { age: "Age", gender: "Gender", ethnicity: "Ethnicity", education: "Education", systolic_bp: "Blood pressure, systolic", diastolic_bp: "Blood pressure, diastolic", diabetes: "Diabetes",
  high_blood_pressure: "High blood pressure", high_cholesterol: "High cholesterol", family_history_dementia: "Family history of dementia", smoking: "Smoking", head_injury: "Serious head injury",
  hearing_loss: "Hearing loss", depression: "Depression / low mood", physical_activity: "Physical activity" };
let charts = [];   // [canvas, draw function] so charts repaint when the theme changes
const redrawCharts = () => charts.forEach(([c, f]) => f(c));
function chart(w, h, cls, draw) {
  const c = el("canvas", cls); c.width = w * 2; c.height = h * 2; c.style.aspectRatio = w + "/" + h; charts.push([c, draw]); draw(c); return c;
}
function drawTaps(c, taps) {   // taps per second of the test
  const g = c.getContext("2d"), W = c.width, H = c.height, bins = Array(30).fill(0); taps.forEach(t => { if (t >= 0 && t < 30) bins[Math.floor(t)]++; });
  g.clearRect(0, 0, W, H); const max = Math.max(...bins, 1), bw = (W - 70) / 30; g.font = "22px system-ui"; g.fillStyle = css("--muted"); g.strokeStyle = css("--line");
  g.beginPath(); g.moveTo(50, H - 40); g.lineTo(W - 10, H - 40); g.stroke(); g.fillText(String(max), 8, 36); g.fillText("0", 28, H - 42);
  bins.forEach((n, i) => { const bh = (H - 90) * n / max; g.fillStyle = css("--accent"); g.globalAlpha = .85; g.fillRect(55 + i * bw, H - 40 - bh, bw - 4, bh); g.globalAlpha = 1; });
  g.fillStyle = css("--muted"); [0, 10, 20, 30].forEach(s => g.fillText(s + " s", 50 + s * bw - (s === 30 ? 40 : 0), H - 10)); g.fillText("taps per second", W - 220, 36);
}
function drawGaze(c, tr, screen) {   // gaze path on the screen; bright part = first 300 ms (what the model scored)
  const g = c.getContext("2d"), W = c.width, H = c.height; g.clearRect(0, 0, W, H); g.strokeStyle = css("--line"); g.lineWidth = 2; g.strokeRect(1, 1, W - 2, H - 2);
  if (!tr || !tr.t || !tr.t.length) return; const [sw, sh] = screen || [1280, 720], px = x => Math.max(4, Math.min(W - 4, x / sw * W)), py = y => Math.max(4, Math.min(H - 4, y / sh * H));
  g.lineWidth = 3; g.lineJoin = "round";
  for (let i = 1; i < tr.t.length; i++) { g.strokeStyle = tr.t[i] <= 300 ? css("--accent") : css("--muted"); g.globalAlpha = tr.t[i] <= 300 ? 1 : .55; g.beginPath(); g.moveTo(px(tr.x[i - 1]), py(tr.y[i - 1])); g.lineTo(px(tr.x[i]), py(tr.y[i])); g.stroke(); }
  g.globalAlpha = 1; g.fillStyle = css("--flag"); g.beginPath(); g.arc(px(tr.x[0]), py(tr.y[0]), 7, 0, 7); g.fill();
}
const TASK_NAMES = { antisaccade1: "Look opposite the dot", visual_attention1: "Free viewing", visual_search1: "Find the T" };
function modalityDetails(r, s) {
  const a = r.artifact || {}, box = el("div");
  if (r.modality === "clock") {
    const m = el("div", "media");
    if (s.clock_jpeg) { const f = el("figure"), im = el("img", "drawing"); im.src = "data:image/jpeg;base64," + s.clock_jpeg; im.alt = "The patient's clock drawing"; f.append(im, el("figcaption", "", "Drawing as submitted (JPEG)")); m.append(f); }
    if (a.preview) { const f = el("figure"), im = el("img", "preview"); im.src = "data:image/png;base64," + a.preview; im.alt = "What the model saw"; f.append(im, el("figcaption", "", "Cropped view the model scored")); m.append(f); }
    box.append(m);
  } else if (r.modality === "motor") {
    if (a.taps && a.taps.length) { box.append(chart(440, 200, "chart", c => drawTaps(c, a.taps)), el("p", "muted small", a.taps.length + " taps recorded over 30 s (age used: " + a.age + ").")); }
  } else if (r.modality === "eye") {
    if (a.tasks) { const m = el("div", "media"); Object.entries(a.tasks).forEach(([k, tr]) => { const f = el("figure"); f.append(chart(260, 150, "gaze", c => drawGaze(c, tr, a.screen)), el("figcaption", "", (TASK_NAMES[k] || k) + " · " + tr.t.length + " gaze points")); m.append(f); });
      box.append(m, el("p", "muted small", "Orange dot = where the path starts. Bright line = first 300 ms after the picture appeared (the part the model scored); grey = the rest of the 3 s.")); }
  } else if (r.modality === "speech") {
    if (a.audio) { const au = el("audio"); au.controls = true; au.src = "data:" + (a.mime || "audio/webm") + ";base64," + a.audio; box.append(au); }
    if (a.transcript) box.append(el("p", "muted small", "What the speech-to-text heard:"), el("div", "transcript", a.transcript));
  }
  if (!box.childElementCount && r.status !== "not_run" && !Object.keys(a).length && r.modality !== "clock") box.append(el("p", "muted small", "No recording was stored for this older record."));
  if (r.evidence || r.caveat) { const d = el("details"); d.append(el("summary", "", "About this measure")); if (r.evidence) d.append(el("p", "", "Evidence: " + r.evidence)); if (r.caveat) d.append(el("p", "", "Limits: " + r.caveat)); box.append(d); }
  return box;
}
async function openRecord(code) {
  const box = $("record"); charts = []; box.replaceChildren(el("p", "muted", "Loading…"));
  try {
    const { patient: p, screenings } = await post("/api/doctor/record", { token, code }); box.replaceChildren(); box.scrollIntoView({ behavior: "smooth", block: "start" });
    const info = el("div", "card"), top = el("div", "who"); top.append(el("h2", "", p.name), el("span", "badge", p.code));
    const cp = el("button", "link", "Copy ID"); cp.onclick = () => copy(p.code, cp); top.append(cp);
    info.append(top, el("p", "muted", "Registered " + (p.created_at || "").slice(0, 10)));
    const t = el("table"); Object.entries(p.questionnaire || {}).forEach(([k, v]) => { const tr = el("tr"); tr.append(el("th", "", QLABEL[k] || k), el("td", "", v)); t.append(tr); });
    info.append(el("p", "muted small", "Questionnaire answers are context only; they did not change any result."), t); box.append(info);
    if (!screenings.length) { box.append(el("div", "card muted", "No screening has been completed yet.")); return; }
    screenings.forEach((s, i) => {
      const c = el("div", "card"); c.append(el("h2", "", "Screening " + (screenings.length - i) + " · " + (s.created_at || "").slice(0, 16).replace("T", " ")));
      if (s.combined) c.append(el("div", "overall" + (s.combined.follow_up ? " follow" : ""), s.combined.text));
      const tiles = el("div", "tiles");
      (s.results || []).forEach(r => {
        const tile = el("div", "tile");
        tile.append(el("h3", "", NAMES[r.modality]), resultBlock(r), modalityDetails(r, s));
        if (s.summary && s.summary.modalities) tile.append(el("p", "says muted", s.summary.modalities[r.modality]));
        tiles.append(tile);
      });
      c.append(tiles); box.append(c);
    });
  } catch (err) { box.replaceChildren(el("p", "err", err.message)); }
}

/* ---------- start-up banner ---------- */
fetch("/api/status").then(r => r.json()).then(st => {
  const notes = [];
  for (const [m, files] of Object.entries(st.missing_files)) notes.push(NAMES[m] + " is unavailable: missing " + files.join(", ") + " in models/.");
  if (!st.vendor) notes.push("The eye test is unavailable: the MediaPipe files in app/static/vendor/ are missing.");
  if (!st.ollama.running) notes.push("Ollama is not running, so the summary will use a fixed template.");
  else if (!st.ollama.model) notes.push("Summary model not installed (ollama pull llama3.1:8b); a fixed template will be used.");
  if (st.storage !== "Supabase") notes.push("Supabase is not configured, so records are saved only in a local file (see supabase/SETUP.md).");
  if (!st.cookie_picture) notes.push("The Cookie Theft picture is missing: add app/static/cookie_theft.jpg.");
  if (notes.length) { const b = el("div", "note"); b.setAttribute("role", "status"); notes.forEach(n => b.append(el("p", "small", n))); $("banner").append(b); }
}).catch(() => {});
