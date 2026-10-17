// The panels around the 3D view: the controller and situation controls (left), the controller's brain — what it sees,
// what it decides, what it earns (right) — the event log (bottom), the Train drawer and the help card. Plain DOM.
import { FEATURES, ACTIONS } from '../bot/env.js';
import { DIFFS } from '../sim/world.js';
import { REWARD } from '../../train/core.js';

const h = (tag, attrs = {}, ...kids) => {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) { if (k === 'class') el.className = v; else if (k.startsWith('on')) el.addEventListener(k.slice(2), v); else if (k === 'html') el.innerHTML = v; else el.setAttribute(k, v); }
  for (const k of kids.flat()) if (k != null) el.append(k.nodeType ? k : document.createTextNode(String(k)));
  return el;
};
const fmt1 = (v) => (Math.abs(v) >= 100 ? v.toFixed(0) : v.toFixed(1));

export const BOTS = [
  { spec: 'scripted', label: 'Rules — open loop (the baseline)', note: 'Trusts the camera: goes where it says, descends, closes. Only as good as the calibration.' },
  { spec: 'careful', label: 'Rules — with the correction (by hand)', note: 'The same rules plus one: when the wrist sensor sees the object off-centre, nudge towards it first.' },
  { spec: 'bots/example.js', label: 'bots/example.js — a policy in 15 lines', note: 'The closed-loop pick written on the observation vector alone: copy it and make your own.' },
  { spec: 'bots/clone.json', label: 'Lesson 1 — imitation (clone.json)', note: 'A network fitted to copy the open-loop rules. Copies their blindness too: it never nudges.' },
  { spec: 'bots/reinforce.json', label: 'Lesson 3 — reinforcement (reinforce.json)', note: 'The clone, improved by reward alone. It discovered the nudges on its own — watch for them before the descent.' },
  { spec: 'bots/evolve.json', label: 'Lesson 2 — evolution (evolve.json)', note: 'Random networks, the best survive, mutated copies fill the next generation: no teacher, no gradients. Evolved on normal; it found the whole pick sequence — and its own way of retrying.' },
  { spec: 'bots/clone-careful.json', label: 'Lesson 1b — imitation of the careful rules', note: 'Cloning a teacher that already corrects: the student corrects too. Supervised learning copies whatever it is shown.' },
];

export function createHud(root, cb) {
  // ---- left: controller and situation
  const botSel = h('select', { class: 'sel', onchange: () => cb.bot(botSel.value) }, BOTS.map((b) => h('option', { value: b.spec }, b.label)), h('option', { value: '__file' }, 'Your own file… (or drop a .json / .js on the page)'));
  const botNote = h('div', { class: 'note' });
  const diffSel = h('select', { class: 'sel', onchange: () => cb.situation() }, DIFFS.map((d) => h('option', { value: d.id }, `${d.name} — camera off by up to ${(d.bias * 1000).toFixed(0)} mm${d.drift ? ', drifting' : ''}`)));
  diffSel.value = 'hard';
  const objSel = h('select', { class: 'sel', onchange: () => cb.situation() }, [1, 2, 3].map((n) => h('option', { value: n }, `${n} object${n > 1 ? 's' : ''}`)));
  objSel.value = '3';
  const seedIn = h('input', { class: 'in', type: 'number', value: '7', min: '1', step: '1', onchange: () => cb.situation() });
  const speeds = [[0, '⏸'], [0.25, '¼×'], [1, '1×'], [4, '4×'], [16, '16×']];
  const speedBtns = speeds.map(([v, t]) => h('button', { class: 'btn', 'data-speed': v, onclick: () => cb.speed(v) }, t));
  const totals = h('div', { class: 'totals' });
  const left = h('div', { class: 'panel left' },
    h('h1', {}, 'PICK IT EASY'), h('div', { class: 'sub' }, 'a robotic arm learns to correct its own pick'),
    h('label', {}, 'Controller'), botSel, botNote,
    h('div', { class: 'row' }, h('div', { class: 'col' }, h('label', {}, 'Camera calibration'), diffSel)),
    h('div', { class: 'row' }, h('div', { class: 'col' }, h('label', {}, 'Objects'), objSel), h('div', { class: 'col' }, h('label', {}, 'Seed'), seedIn)),
    h('div', { class: 'row btns' }, h('button', { class: 'btn', onclick: () => cb.next() }, 'New situation'), h('button', { class: 'btn', onclick: () => cb.replay() }, 'Replay')),
    h('label', {}, 'Speed'), h('div', { class: 'row btns' }, ...speedBtns, h('button', { class: 'btn', onclick: () => cb.step() }, 'Step')),
    h('label', {}, 'This session'), totals,
    h('div', { class: 'row btns foot' }, h('button', { class: 'btn acc', onclick: () => cb.train() }, 'Train a controller'), h('button', { class: 'btn', onclick: () => cb.help() }, 'What am I looking at?')),
  );

  // ---- right: the brain
  const actionBig = h('div', { class: 'action' }, '—');
  const actRows = ACTIONS.map((n) => ({ n, bar: h('div', { class: 'bar' }), val: h('span', { class: 'v' }), row: null }));
  for (const r of actRows) r.row = h('div', { class: 'brow' }, h('span', { class: 'k' }, r.n), h('div', { class: 'track' }, r.bar), r.val);
  const featRows = FEATURES.map((n) => ({ n, bar: h('div', { class: 'bar f' }), val: h('span', { class: 'v' }), row: null }));
  for (const r of featRows) r.row = h('div', { class: 'brow' }, h('span', { class: 'k' }, r.n), h('div', { class: 'track mid' }, r.bar), r.val);
  const rewardBig = h('div', { class: 'reward' }, '0.0');
  const rewardParts = h('div', { class: 'parts' });
  const truth = h('div', { class: 'truth' });
  const probsNote = h('div', { class: 'note' }, 'bars: the network\'s probability for each action (a hand-written policy just picks one)');
  const right = h('div', { class: 'panel right' },
    h('h2', {}, 'Decides'), actionBig, h('div', { class: 'bars' }, actRows.map((r) => r.row)), probsNote,
    h('h2', {}, 'Sees ', h('small', {}, `${FEATURES.length} numbers, 15× a second`)), h('div', { class: 'bars' }, featRows.map((r) => r.row)),
    h('h2', {}, 'Earns ', h('small', {}, 'this situation')), rewardBig, rewardParts,
    h('h2', {}, 'Ground truth ', h('small', {}, 'hidden from the controller')), truth,
  );

  // ---- bottom: the log
  const logEl = h('div', { class: 'log' });
  const bottom = h('div', { class: 'panel bottom' }, logEl);

  // ---- the Train drawer
  const algoSel = h('select', { class: 'sel', onchange: () => showAlgo() }, h('option', { value: 'clone' }, 'Lesson 1 — imitation (clone the rules)'), h('option', { value: 'reinforce' }, 'Lesson 3 — reinforcement (REINFORCE)'), h('option', { value: 'evolve' }, 'Lesson 2 — evolution'));
  const fromSel = h('select', { class: 'sel' }, h('option', { value: '' }, 'random weights'), h('option', { value: 'bots/clone.json' }, 'bots/clone.json (imitate first, then improve)'), h('option', { value: 'bots/reinforce.json' }, 'bots/reinforce.json'), h('option', { value: '__current' }, 'the controller now playing (if a network)'));
  fromSel.value = 'bots/clone.json';
  const P = {};   // parameter inputs per algorithm
  const num = (k, v, step = 'any') => (P[k] = h('input', { class: 'in', type: 'number', value: String(v), step }));
  const field = (label, el) => h('div', { class: 'col' }, h('label', {}, label), el);
  const cloneForm = h('div', { class: 'row wrap' }, field('episodes', num('c_episodes', 24)), field('epochs', num('c_epochs', 30)), field('hidden', num('c_hidden', 32)), field('lr', num('c_lr', 0.003)), field('teacher', (P.c_teach = h('select', { class: 'sel' }, h('option', { value: 'naive' }, 'open-loop rules'), h('option', { value: 'careful' }, 'careful rules')))), field('balance', (P.c_balance = h('select', { class: 'sel' }, h('option', { value: '' }, 'off'), h('option', { value: '1' }, 'on')))));
  const reinForm = h('div', { class: 'row wrap' }, field('episodes', num('r_episodes', 24000)), field('batch', num('r_batch', 32)), field('gamma', num('r_gamma', 0.95)), field('lr', num('r_lr', 0.002)), field('entropy', num('r_entropy', 0.02)), field('hidden', num('r_hidden', 32)), field('start from', fromSel));
  const evoForm = h('div', { class: 'row wrap' }, field('population', num('e_pop', 64)), field('generations', num('e_gens', 250)), field('elite', num('e_elite', 8)), field('sigma', num('e_sigma', 0.15)), field('episodes each', num('e_episodes', 4)), field('hidden', num('e_hidden', 32)), field('start from', (P.e_from = h('select', { class: 'sel' }, h('option', { value: '' }, 'random weights'), h('option', { value: 'bots/clone.json' }, 'bots/clone.json')))));
  const seedIn2 = num('seed', 1, '1');
  const runBtn = h('button', { class: 'btn acc', onclick: () => cb.run() }, 'Run'), stopBtn = h('button', { class: 'btn', onclick: () => cb.stop() }, 'Stop');
  const playBtn = h('button', { class: 'btn', onclick: () => cb.playTrained(), disabled: 'true' }, 'Play it'), dlBtn = h('button', { class: 'btn', onclick: () => cb.download(), disabled: 'true' }, 'Download .json');
  const progress = h('pre', { class: 'progress' }), chart = h('canvas', { class: 'chart', width: '640', height: '160' });
  const algoNote = h('div', { class: 'note' });
  const NOTES = {
    clone: 'The rules play; every decision is written down as (what it saw → what it did); a network is fitted to predict the action. Watch the held-out accuracy. The student inherits the teacher\'s blindness: trained on the open-loop rules it never nudges.',
    reinforce: 'The network plays by drawing actions from its own probabilities; every action is then pushed up or down in proportion to the reward that followed it, against the batch\'s average. Start from clone.json: it begins as the rules and, ten thousand episodes later, nudges. Watch the "nudges" column.',
    evolve: 'A population of networks plays the same situations; the best few survive; their mutated copies fill the next generation. No gradients, no teacher. It uses the camera calibration set on the left: on normal the whole pick sequence appears around generation 100 (≈ 2 min); on hard, from random weights, nothing is ever placed — the first grasp is too rare to be selected for. Start from an evolved or cloned network to go on from there.',
  };
  const showAlgo = () => { const a = algoSel.value; cloneForm.hidden = a !== 'clone'; reinForm.hidden = a !== 'reinforce'; evoForm.hidden = a !== 'evolve'; algoNote.textContent = NOTES[a]; };
  showAlgo();
  const drawer = h('div', { class: 'drawer', hidden: 'true' },
    h('div', { class: 'dhead' }, h('h2', {}, 'Train a controller ', h('small', {}, 'in your browser, in a Web Worker — the same code as train/*.mjs')), h('button', { class: 'btn x', onclick: () => cb.train() }, '×')),
    h('div', { class: 'row' }, field('lesson', algoSel), field('seed', seedIn2)), algoNote, cloneForm, reinForm, evoForm,
    h('div', { class: 'row btns' }, runBtn, stopBtn, playBtn, dlBtn), chart, progress);

  // ---- the help card
  const help = h('div', { class: 'help', hidden: 'true' },
    h('h2', {}, 'What am I looking at?'),
    h('p', { html: 'A 4-joint arm picks the objects off the table and drops them in the <b>bin</b> (green rim). It has two senses. A <b>camera over the table</b> tells it where each object is — but the camera\'s calibration is off: every estimate is shifted by the same unknown few millimetres. That estimate is the <b style="color:var(--mag)">pink outline</b> beside each object. A <b>sensor in the wrist</b> (the <b style="color:var(--amber)">amber cone</b>) sees the true offset of an object right under the fingers, but only from close by.' }),
    h('p', { html: 'The <b>open-loop rules</b> trust the camera: on <i>hard</i> they close on air, pinch an edge and drop the object, or knock it off the table. The <b>correction</b> is to use the wrist sensor: nudge over before going down. You can write that rule by hand (the careful rules, <code>bots/example.js</code>) — or <b>train a small network to find it</b>: that is what the three lessons in <code>train/</code> do, the same way the bots in the Hacker Squad game are trained. Imitation copies the rules, blindness included; reinforcement, started from the imitation, discovers the nudges from the reward alone.' }),
    h('p', { html: 'Right panel: what the controller <b>sees</b> (21 numbers), <b>decides</b> (one of 11 actions, every 4 frames) and <b>earns</b>. Drag to orbit, wheel to zoom, right-drag to pan. <kbd>Space</kbd> pause · <kbd>N</kbd> new situation · <kbd>1</kbd>–<kbd>4</kbd> speed · <kbd>S</kbd> step · <kbd>T</kbd> train · <kbd>?</kbd> this card.' }),
    h('div', { class: 'row btns' }, h('button', { class: 'btn', onclick: () => cb.help() }, 'Close')));

  root.append(left, right, bottom, drawer, help);

  // ---- updates
  const hud = { botSel, diffSel, objSel, seedIn, P, algoSel, fromSel, runBtn, stopBtn, playBtn, dlBtn, drawer, help, chart, progress };
  hud.setBot = (spec, note) => { if ([...botSel.options].some((o) => o.value === spec)) botSel.value = spec; botNote.textContent = note || (BOTS.find((b) => b.spec === spec) || {}).note || ''; };
  hud.setSpeed = (v) => speedBtns.forEach((b) => b.classList.toggle('on', +b.dataset.speed === v));
  hud.setTotals = (t) => { totals.innerHTML = `<span>${t.n} situation${t.n === 1 ? '' : 's'}</span><span>placed <b>${t.placed}/${t.objects}</b></span><span>missed <b>${t.misses}×</b></span><span>dropped <b>${t.drops}×</b></span><span>lost <b>${t.lost}</b></span><span>mean reward <b>${t.n ? fmt1(t.reward / t.n) : '—'}</b></span>`; };
  hud.brain = (bot, sim, scorer) => {
    const a = bot.a ?? -1, probs = bot.policy && bot.policy.probs;
    actionBig.textContent = a >= 0 ? ACTIONS[a] : '—';
    actionBig.classList.toggle('nudge', a >= 6 && a <= 9);
    actRows.forEach((r, i) => { const p = probs ? probs[i] : i === a ? 1 : 0; r.bar.style.width = `${(100 * p).toFixed(1)}%`; r.val.textContent = probs ? `${(100 * p).toFixed(0)}%` : ''; r.row.classList.toggle('on', i === a); });
    probsNote.hidden = !!probs;
    const obs = bot.obs;
    if (obs) featRows.forEach((r, i) => { const v = obs[i]; r.bar.style.left = v < 0 ? `${50 + 50 * v}%` : '50%'; r.bar.style.width = `${Math.abs(50 * v)}%`; r.val.textContent = v.toFixed(2); r.row.classList.toggle('hot', Math.abs(v) > 0.001 && (r.n.startsWith('wrist') || r.n === 'slip')); });
    rewardBig.textContent = fmt1(scorer.total); rewardBig.classList.toggle('neg', scorer.total < 0);
    rewardParts.innerHTML = Object.entries(scorer.parts).filter(([, v]) => v).map(([k, v]) => `<span class="${v < 0 ? 'neg' : ''}">${k} <b>${v > 0 ? '+' : ''}${fmt1(v)}</b></span>`).join('');
    const b = sim.bias, held = sim.holding >= 0 ? sim.objects[sim.holding] : null, t = bot.mem && bot.mem.target >= 0 ? sim.objects[bot.mem.target] : null;
    let off = '';
    if (t && t.state === 'table') { const tip = sim.arm.tip; off = `fingers → target <b>${(1000 * (t.x - tip.x)).toFixed(0)}, ${(1000 * (t.z - tip.z)).toFixed(0)} mm</b> (clean within ${(1000 * t.w * 0.5).toFixed(0)} mm)`; }
    truth.innerHTML = `camera error <b>${(1000 * b.x).toFixed(0)}, ${(1000 * b.z).toFixed(0)} mm</b> (${(1000 * Math.hypot(b.x, b.z)).toFixed(0)} mm)<br>${off}${held ? `<br>holding object ${held.i + 1}${held.slip ? ' <b class="neg">by an edge — slipping</b>' : ' — centred'}` : ''}<br>t = ${sim.t.toFixed(1)} s · frame ${sim.frame}`;
  };
  const LOGC = { timeout: 'warn', grasp: 'ok', lift: 'ok', place: 'ok', slipGrasp: 'warn', bump: 'warn', setdown: '', miss: 'bad', drop: 'bad', lost: 'bad', done: 'ok', start: 'info' };
  const TEXT = { start: 'new situation:', timeout: 'time is up', grasp: 'clean grasp', slipGrasp: 'grasped by an edge — slipping', lift: 'lifted', place: 'placed in the bin', miss: 'closed on air', drop: 'dropped on the table', lost: 'off the table — lost', bump: 'a finger shoved it', setdown: 'set down', done: 'all objects in the bin' };
  hud.log = (sim, name, i, extra = '') => {
    const line = h('div', { class: `l ${LOGC[name] || ''}` }, h('span', { class: 't' }, `${sim.t.toFixed(1)}s`), `${TEXT[name] || name}${i >= 0 ? ` · object ${i + 1}` : ''}${extra ? ' ' + extra : ''}`);
    logEl.prepend(line); while (logEl.children.length > 60) logEl.lastChild.remove();
  };
  hud.clearLog = () => { logEl.innerHTML = ''; };
  return hud;
}

/** A small line chart: series = [{ xs, ys, color, label }], drawn on a canvas. */
export function drawChart(canvas, series, { title = '' } = {}) {
  const c = canvas.getContext('2d'), W = canvas.width, H = canvas.height, m = { l: 44, r: 12, t: 18, b: 22 };
  c.clearRect(0, 0, W, H);
  c.fillStyle = 'rgba(255,255,255,.03)'; c.fillRect(m.l, m.t, W - m.l - m.r, H - m.t - m.b);
  const all = series.filter((s) => s.ys.length);
  if (!all.length) return;
  const xmax = Math.max(...all.map((s) => s.xs.at(-1))), xmin = Math.min(...all.map((s) => s.xs[0]));
  c.font = '11px ui-monospace, Consolas, monospace'; c.fillStyle = '#8fb4c4'; c.textAlign = 'left'; c.fillText(title, m.l, 12);
  all.forEach((s, si) => {
    let ymin = Math.min(...s.ys), ymax = Math.max(...s.ys); if (ymax - ymin < 1e-6) { ymax += 1; ymin -= 1; }
    if (s.range) { ymin = s.range[0]; ymax = s.range[1]; }
    const X = (x) => m.l + (W - m.l - m.r) * (xmax > xmin ? (x - xmin) / (xmax - xmin) : 0), Y = (y) => m.t + (H - m.t - m.b) * (1 - (y - ymin) / (ymax - ymin));
    c.strokeStyle = s.color; c.lineWidth = 1.5; c.beginPath();
    s.ys.forEach((y, i) => (i ? c.lineTo(X(s.xs[i]), Y(y)) : c.moveTo(X(s.xs[i]), Y(y)))); c.stroke();
    c.fillStyle = s.color; c.textAlign = si ? 'right' : 'left';
    c.fillText(`${s.label} ${s.ys.at(-1).toFixed(s.digits ?? 1)}`, si ? W - m.r : m.l + 4, H - 6);
    c.textAlign = si ? 'left' : 'right';
    c.fillText(ymax.toFixed(s.digits ?? 0), si ? W - m.r + 2 : m.l - 4, m.t + 10); c.fillText(ymin.toFixed(s.digits ?? 0), si ? W - m.r + 2 : m.l - 4, H - m.b);
  });
}
