// The viewer: the sim stepped at 60 Hz (× a speed), a controller driving it, the three.js scene drawing it, the HUD
// showing the controller's mind, and the Train drawer running a lesson in a worker. ?bot=<spec> picks the controller,
// ?diff=hard ?objects=3 ?seed=7 the situation.
import { createSim, DIFFS } from './sim/world.js';
import { createScene } from './view/scene.js';
import { createOrbit } from './view/orbit.js';
import { createHud, drawChart, BOTS } from './view/hud.js';
import { loadBot, fromJSON } from './bot/bots.js';
import { createScorer } from '../train/core.js';

const q = new URLSearchParams(location.search);
const canvas = document.getElementById('c'), ui = document.getElementById('ui');
const sim = createSim(), view = createScene(canvas), orbit = createOrbit(view.camera, canvas);
let botDef = null, bot = null, scorer = null, speed = 1, acc = 0, seconds = 30, endPause = 0, trained = null, worker = null;
const totals = { n: 0, placed: 0, objects: 0, misses: 0, drops: 0, lost: 0, reward: 0 };
let counted = false;

const hud = createHud(ui, {
  bot: (spec) => (spec === '__file' ? pickFile() : setBot(spec)),
  situation: () => start(+hud.seedIn.value),
  next: () => start(+hud.seedIn.value + 1),
  replay: () => start(+hud.seedIn.value),
  speed: setSpeed,
  step: () => { setSpeed(0); tick(1); },
  train: () => { hud.drawer.hidden = !hud.drawer.hidden; },
  help: () => { hud.help.hidden = !hud.help.hidden; },
  run: runTraining, stop: stopTraining,
  playTrained: () => { if (trained) { botDef = fromJSON(trained, { name: trained.name }); hud.setBot('', `${trained.name}: trained just now — ${trained.trained.algorithm}, score ${trained.trained.score} (rules ${trained.trained.rules}, careful ${trained.trained.careful})`); start(+hud.seedIn.value); hud.drawer.hidden = true; } },
  download: () => { if (!trained) return; const a = document.createElement('a'); a.href = URL.createObjectURL(new Blob([JSON.stringify(trained)], { type: 'application/json' })); a.download = `${trained.name}.json`; a.click(); },
});

function setSpeed(v) { speed = v; hud.setSpeed(v); }
async function setBot(spec) {
  try { botDef = await loadBot(spec); hud.setBot(spec); }
  catch (e) { console.error(e); hud.setBot(spec, `could not load ${spec}: ${e.message} — train it first (see train/README.md)`); botDef = await loadBot('scripted'); }
  start(+hud.seedIn.value);
}
function pickFile() {
  const inp = document.createElement('input'); inp.type = 'file'; inp.accept = '.json,.js,.mjs';
  inp.onchange = async () => { if (inp.files[0]) { try { botDef = await loadBot(inp.files[0]); hud.setBot('', `${botDef.name}: your file`); start(+hud.seedIn.value); } catch (e) { hud.setBot('scripted', `could not load the file: ${e.message}`); } } };
  inp.click();
}
addEventListener('dragover', (e) => e.preventDefault());
addEventListener('drop', async (e) => { e.preventDefault(); const f = e.dataTransfer.files[0]; if (!f) return; try { botDef = await loadBot(f); hud.setBot('', `${botDef.name}: your file`); start(+hud.seedIn.value); } catch (err) { hud.setBot('scripted', `could not load the file: ${err.message}`); } });

/** A fresh situation: the seed, objects and difficulty from the panel; a fresh controller (nothing carried over). */
function start(seed) {
  hud.seedIn.value = seed;
  sim.start({ objects: +hud.objSel.value, diff: hud.diffSel.value, seed });
  bot = botDef.create({ seconds }); scorer = createScorer(sim); counted = false; endPause = 0; acc = 0;
  view.build(sim); hud.clearLog();
  hud.log(sim, 'start', -1, `${botDef.name} · ${DIFFS.find((d) => d.id === sim.diff.id).name.toLowerCase()} · ${sim.objects.length} object${sim.objects.length > 1 ? 's' : ''} · seed ${seed} · camera off by ${(1000 * Math.hypot(sim.bias.x, sim.bias.z)).toFixed(0)} mm`);
}

/** n fixed steps of the sim under the controller, with the reward and the log. */
function tick(n) {
  for (let i = 0; i < n; i++) {
    if (sim.done || sim.frame >= seconds * 60) {
      if (!counted) { counted = true; totals.n++; totals.placed += sim.stats.placed; totals.objects += sim.objects.length; totals.misses += sim.stats.misses; totals.drops += sim.stats.drops; totals.lost += sim.stats.lost; totals.reward += scorer.total; hud.setTotals(totals); if (!sim.done) hud.log(sim, 'timeout', -1, `— ${sim.stats.placed}/${sim.objects.length} placed in ${seconds} s, reward ${scorer.total.toFixed(1)}`); }
      if (++endPause > 90) start(+hud.seedIn.value + 1);        // a pause to look, then the next situation
      continue;
    }
    if (sim.frame % (botDef.every || 4) === 0) scorer.tick();
    const cmd = bot(sim);
    for (const e of scorer.step(cmd)) hud.log(sim, e.name, e.i);
    if (sim.done) hud.log(sim, 'done', -1, `— reward ${scorer.total.toFixed(1)} in ${sim.t.toFixed(1)} s`);
  }
}

let last = performance.now();
function frame(now) {
  requestAnimationFrame(frame);
  const dt = Math.min(0.1, (now - last) / 1000); last = now;
  if (speed > 0) { acc += dt * speed * 60; const n = Math.min(Math.floor(acc), 64); acc -= n; tick(n); }
  const mem = bot && bot.mem;
  view.draw(sim, mem ? { target: mem.target, cam: mem.cam, camOk: mem.camOk } : {});
  view.render();
  if (bot) hud.brain(bot, sim, scorer);
}

// ---- keyboard
addEventListener('keydown', (e) => {
  if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT') return;
  if (e.code === 'Space') { e.preventDefault(); setSpeed(speed ? 0 : 1); }
  else if (e.key === 'n' || e.key === 'N') start(+hud.seedIn.value + 1);
  else if (e.key === 'r' || e.key === 'R') start(+hud.seedIn.value);
  else if (e.key === 's' || e.key === 'S') { setSpeed(0); tick(1); }
  else if (e.key >= '1' && e.key <= '4') setSpeed([0.25, 1, 4, 16][+e.key - 1]);
  else if (e.key === 't' || e.key === 'T') hud.drawer.hidden = !hud.drawer.hidden;
  else if (e.key === '?' || e.key === 'h' || e.key === 'H') hud.help.hidden = !hud.help.hidden;
  else if (e.key === 'Escape') { hud.help.hidden = true; hud.drawer.hidden = true; }
});

// ---- the Train drawer: a worker runs the lesson, the chart and the progress text follow it
const series = { xs: [], ys: [], xs2: [], ys2: [] };
async function runTraining() {
  stopTraining();
  const P = hud.P, algo = hud.algoSel.value, seed = +P.seed.value || 1, diff = hud.diffSel.value;
  let opts, fromSpec = '';
  if (algo === 'clone') opts = { episodes: +P.c_episodes.value, epochs: +P.c_epochs.value, hidden: +P.c_hidden.value, lr: +P.c_lr.value, teach: P.c_teach.value, balance: !!P.c_balance.value, seed, diff };
  else if (algo === 'reinforce') { opts = { episodes: +P.r_episodes.value, batch: +P.r_batch.value, gamma: +P.r_gamma.value, lr: +P.r_lr.value, entropy: +P.r_entropy.value, hidden: +P.r_hidden.value, seed, diff }; fromSpec = hud.fromSel.value; }
  else { opts = { pop: +P.e_pop.value, gens: +P.e_gens.value, elite: +P.e_elite.value, sigma: +P.e_sigma.value, episodes: +P.e_episodes.value, hidden: +P.e_hidden.value, seed, diff }; fromSpec = P.e_from.value; }
  let from = null;
  try {
    if (fromSpec === '__current') from = botDef.json || null;
    else if (fromSpec) from = await (await fetch(fromSpec)).json();
  } catch (e) { hud.progress.textContent = `could not load ${fromSpec}: ${e.message}`; return; }
  if (fromSpec && !from) { hud.progress.textContent = 'the controller now playing is not a network — pick another start'; return; }
  series.xs = []; series.ys = []; series.xs2 = []; series.ys2 = [];
  hud.progress.textContent = `starting ${algo} on ${diff}${from ? ` from ${from.name}` : ''}…\n`;
  hud.playBtn.disabled = hud.dlBtn.disabled = true; hud.runBtn.disabled = true;
  const t0 = performance.now(), T = () => `${((performance.now() - t0) / 1000).toFixed(0)}s`;
  const line = (s) => { hud.progress.textContent += s + '\n'; hud.progress.scrollTop = hud.progress.scrollHeight; };
  const f = (e) => `${e.reward.toFixed(1)} (placed ${e.placed}/${e.objects}, missed ${e.misses}×, dropped ${e.drops}×${e.lost ? `, lost ${e.lost}` : ''}, ${e.seconds.toFixed(1)} s each)`;
  worker = new Worker('train/worker.js', { type: 'module' });
  worker.onmessage = ({ data: p }) => {
    switch (p.stage) {
      case 'collect': hud.progress.textContent = hud.progress.textContent.replace(/collecting.*\n?$/, '') + `collecting ${p.decisions} decisions from ${p.episodes} episodes…\n`; break;
      case 'actions': line(`actions: ${p.count.map((c, i) => `${['approach', 'descend', 'grasp', 'lift', 'carry', 'release', 'n+x', 'n-x', 'n+z', 'n-z', 'wait'][i]} ${(100 * c / p.total).toFixed(1)}%`).join(' · ')}`); break;
      case 'epoch': line(`epoch ${p.epoch}  loss ${p.loss.toFixed(3)}  accuracy ${(100 * p.train).toFixed(1)}% train  ${(100 * p.test).toFixed(1)}% held out`); series.xs.push(p.epoch); series.ys.push(p.loss); series.xs2.push(p.epoch); series.ys2.push(100 * p.test);
        drawChart(hud.chart, [{ xs: series.xs, ys: series.ys, color: '#38e8ff', label: 'loss', digits: 2 }, { xs: series.xs2, ys: series.ys2, color: '#46ff8a', label: 'held-out accuracy %', range: [0, 100] }], { title: 'imitation: the loss falls, the accuracy climbs' }); break;
      case 'gen': line(`gen ${p.gen}  best ${f(p.best)}  mean ${p.mean.toFixed(1)}  rules ${p.rules.toFixed(1)}  ${T()}`); series.xs.push(p.gen); series.ys.push(p.best.reward); series.xs2.push(p.gen); series.ys2.push(p.mean);
        drawChart(hud.chart, [{ xs: series.xs, ys: series.ys, color: '#46ff8a', label: 'best' }, { xs: series.xs2, ys: series.ys2, color: '#38e8ff', label: 'mean' }], { title: 'evolution: reward per generation' }); break;
      case 'batch': line(`episodes ${p.done}  reward ${p.reward.toFixed(1)}  placed ${p.placed}/${p.objects}  nudges ${(100 * p.nudges).toFixed(1)}%  ${T()}`); series.xs.push(p.done); series.ys.push(p.reward); series.xs2.push(p.done); series.ys2.push(100 * p.nudges);
        drawChart(hud.chart, [{ xs: series.xs, ys: series.ys, color: '#46ff8a', label: 'reward per batch' }, { xs: series.xs2, ys: series.ys2, color: '#ff3ea8', label: 'nudges %', range: [0, Math.max(5, ...series.ys2)] }], { title: 'reinforcement: the reward, and the correction being learned' }); break;
      case 'score': line(`\non ${p.diff}, 8 fresh situations:\n  rules    ${f(p.rules)}\n  careful  ${f(p.careful)}\n  trained  ${f(p.trained)}`); break;
      case 'done': trained = p.json; line(`done in ${T()} — "Play it" loads it into the viewer; "Download" saves bots/${p.json.name}.json`); hud.playBtn.disabled = hud.dlBtn.disabled = false; hud.runBtn.disabled = false; worker = null; break;
      case 'error': line(`error: ${p.message}`); hud.runBtn.disabled = false; worker = null; break;
    }
  };
  worker.onerror = (e) => { line(`worker error: ${e.message}`); hud.runBtn.disabled = false; };
  worker.postMessage({ algo, opts, from });
}
function stopTraining() { if (worker) { worker.terminate(); worker = null; hud.progress.textContent += 'stopped\n'; hud.runBtn.disabled = false; } }

// ---- go
hud.diffSel.value = DIFFS.some((d) => d.id === q.get('diff')) ? q.get('diff') : 'hard';
if (['1', '2', '3'].includes(q.get('objects'))) hud.objSel.value = q.get('objects');
if (+q.get('seed') > 0) hud.seedIn.value = q.get('seed');
hud.setTotals(totals); setSpeed(1);
if (q.has('train')) hud.drawer.hidden = false;
if (q.has('help')) hud.help.hidden = false;
await setBot(q.get('bot') || BOTS[0].spec);
requestAnimationFrame(frame);
