// EXPERIMENTAL webcam eye test (see app/eye.py for why the result is not counted). Everything runs in this page; no frame leaves the computer.
// MediaPipe Face Landmarker (vendored in /static/vendor) gives iris + eye-corner landmarks; a 9-point calibration maps them to screen pixels.
// Stimuli are ours: the dataset does not publish its stimuli (FINDINGS 6). Each task is recorded for 300 ms from stimulus onset to match the
// ~0.3 s recordings the model was trained on, which at webcam rate is only ~9 samples.
const REC_MS = 300;

async function loadLandmarker() {
  const { FaceLandmarker, FilesetResolver } = await import("/static/vendor/vision_bundle.mjs");
  const fileset = await FilesetResolver.forVisionTasks("/static/vendor/wasm");
  const make = delegate => FaceLandmarker.createFromOptions(fileset, {
    baseOptions: { modelAssetPath: "/static/vendor/face_landmarker.task", delegate }, runningMode: "VIDEO", numFaces: 1 });
  try { return await make("GPU"); } catch { return await make("CPU"); }
}

// per-frame feature vector: iris position inside the eye (both eyes averaged) + nose position (head movement) + bias
function frameFeatures(lm) {
  const ratio = (iris, a, b, top, bot) => [(lm[iris].x - lm[a].x) / (lm[b].x - lm[a].x), (lm[iris].y - lm[top].y) / (lm[bot].y - lm[top].y)];
  const r = ratio(468, 33, 133, 159, 145), l = ratio(473, 362, 263, 386, 374);
  return [(r[0] + l[0]) / 2, (r[1] + l[1]) / 2, lm[1].x, lm[1].y, 1];
}

function solve(A, b) {                       // Gauss-Jordan on a small square system
  const n = b.length, M = A.map((row, i) => [...row, b[i]]);
  for (let i = 0; i < n; i++) {
    let p = i; for (let k = i + 1; k < n; k++) if (Math.abs(M[k][i]) > Math.abs(M[p][i])) p = k;
    [M[i], M[p]] = [M[p], M[i]];
    for (let k = 0; k < n; k++) if (k !== i) { const f = M[k][i] / M[i][i]; for (let j = i; j <= n; j++) M[k][j] -= f * M[i][j]; }
  }
  return M.map((row, i) => row[n] / row[i]);
}
function fit(F, y) {                          // ridge least squares: (F'F + eps I) w = F'y
  const k = F[0].length, A = Array.from({ length: k }, () => Array(k).fill(0)), b = Array(k).fill(0);
  F.forEach((f, i) => f.forEach((fa, a) => { b[a] += fa * y[i]; f.forEach((fb, c) => { A[a][c] += fa * fb; }); }));
  for (let a = 0; a < k; a++) A[a][a] += 1e-6;
  return solve(A, b);
}

export async function startEyeTest(onDone, onMsg) {
  onMsg("Loading the face model…");
  const lmk = await loadLandmarker();
  const video = document.createElement("video"); video.playsInline = true; video.muted = true;
  video.srcObject = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480, frameRate: 30 } });
  await video.play();
  const stop = () => { alive = false; ov.remove(); video.srcObject.getTracks().forEach(t => t.stop()); };

  const ov = document.createElement("div");
  ov.style.cssText = "position:fixed;inset:0;background:#fff;color:#111;z-index:99;font:20px system-ui;cursor:none";
  document.body.append(ov);
  const W = innerWidth, H = innerHeight;
  const say = t => { ov.innerHTML = ""; const p = document.createElement("p"); p.textContent = t; p.style.cssText = "position:absolute;top:8%;width:100%;text-align:center;margin:0"; ov.append(p); };
  const dot = (x, y, c = "#d00", r = 14) => { const d = document.createElement("div"); d.style.cssText = `position:absolute;left:${x - r}px;top:${y - r}px;width:${2 * r}px;height:${2 * r}px;border-radius:50%;background:${c}`; ov.append(d); };
  const text = (s, x, y, size = 30) => { const d = document.createElement("div"); d.textContent = s; d.style.cssText = `position:absolute;left:${x}px;top:${y}px;font-size:${size}px;transform:translate(-50%,-50%)`; ov.append(d); };
  const wait = ms => new Promise(r => setTimeout(r, ms));

  let alive = true, lastVT = -1, rec = null;
  (function loop() {
    if (!alive) return;
    if (video.currentTime !== lastVT && video.readyState >= 2) {
      lastVT = video.currentTime;
      const res = lmk.detectForVideo(video, performance.now());
      if (rec && res.faceLandmarks && res.faceLandmarks[0]) rec({ t: performance.now(), f: frameFeatures(res.faceLandmarks[0]) });
    }
    requestAnimationFrame(loop);
  })();
  const record = async ms => { const out = []; rec = s => out.push(s); await wait(ms); rec = null; return out; };

  // 1. calibration: 9 points
  say("Keep your head still. Look at each red dot until it moves.");
  await wait(2500);
  const pts = [0.1, 0.5, 0.9].flatMap(fy => [0.1, 0.5, 0.9].map(fx => [fx * W, fy * H]));
  const F = [], X = [], Y = [];
  for (const [px, py] of pts) {
    ov.innerHTML = ""; dot(px, py); await wait(500);
    (await record(900)).forEach(q => { F.push(q.f); X.push(px); Y.push(py); });
  }
  if (F.length < 30) { stop(); onMsg("Your face was not tracked well enough. Face the camera in good light and try again."); return; }
  const wx = fit(F, X), wy = fit(F, Y);
  const gaze = f => [f.reduce((s, v, i) => s + v * wx[i], 0), f.reduce((s, v, i) => s + v * wy[i], 0)];
  const cal = F.reduce((s, f, i) => s + Math.hypot(gaze(f)[0] - X[i], gaze(f)[1] - Y[i]), 0) / F.length;

  const out = {};
  const grab = async name => { const s = await record(REC_MS); const t0 = s.length ? s[0].t : 0; out[name] = { t: s.map(q => q.t - t0), x: s.map(q => gaze(q.f)[0]), y: s.map(q => gaze(q.f)[1]) }; };

  // 2. antisaccade: look to the side OPPOSITE the dot
  say("Look at the cross. When a dot appears, look at the opposite side."); await wait(2500);
  ov.innerHTML = ""; text("+", W / 2, H / 2, 48); await wait(1000);
  dot(Math.random() < 0.5 ? 0.15 * W : 0.85 * W, H / 2); await grab("antisaccade1"); ov.innerHTML = ""; await wait(500);

  // 3. visual attention: free viewing of a scattered scene
  say("Look at the picture freely."); await wait(2000);
  ov.innerHTML = "";
  [[.2, .3], [.7, .25], [.4, .6], [.8, .7], [.25, .8], [.55, .4]].forEach(([a, b], i) => dot(a * W, b * H, ["#d00", "#06c", "#090", "#c80", "#808", "#099"][i], 30));
  await grab("visual_attention1"); ov.innerHTML = ""; await wait(500);

  // 4. visual search: find the one T among L's
  say("Find the letter T among the L's."); await wait(2000);
  ov.innerHTML = ""; const tAt = Math.floor(Math.random() * 20);
  for (let i = 0; i < 20; i++) text(i === tAt ? "T" : "L", (0.12 + 0.19 * (i % 5)) * W, (0.3 + 0.13 * Math.floor(i / 5)) * H, 40);
  await grab("visual_search1");

  stop();
  onMsg(`Done. Mean calibration error ${(cal / W * 100).toFixed(1)}% of screen width.`);
  await onDone(out, cal / W);
}
