// What every trainer shares, in plain JavaScript that runs in Node (train/*.mjs) and in the browser (train/worker.js,
// the Train panel): one *episode* (a situation played for `seconds`, scored with a *reward*), a set of varied
// *situations*, scoring, and the three lessons themselves — imitation, evolution, reinforcement — each a function that
// takes the sim and the options and reports its progress through a callback.
//
//   const ep = episode(sim, { policy, ...situation })      policy(obs, s, mem) → action plays one situation
//   ep.reward, ep.parts, ep.samples (with record: true)    the score, what it is made of, and every decision made
//   situations(r, n)                                       n random situations (objects, seed) at a difficulty
//   evaluate(sim, policy, sits)                            the mean reward over those situations
//   trainClone / trainEvolve / trainReinforce(sim, opts, report) → { net, meta }
import { sense, observe, act, createMemory, FEATURES, ACTIONS, A } from '../src/bot/env.js';
import { scriptedPolicy } from '../src/bot/scripted.js';
import { netPolicy } from '../src/bot/bots.js';
import { createNet, forward, backward, createAdam, createGrads, mutate, argmax, rng } from '../src/bot/nn.js';
import { H, BIN } from '../src/sim/world.js';

export const teacher = scriptedPolicy();                       // the open-loop rules: the baseline and lesson 1's teacher
export const careful = scriptedPolicy({ style: 'careful' });   // the rules with the hand-written correction: the bar to reach

// ---- the reward: what "picking well" means, in numbers. Change these and the controllers learn something else.
export const REWARD = {
  grasp: 10,       // a clean grasp (the object centred between the fingers) — once per object, or grasp-release-grasp pays
  slipGrasp: 2,    // a grasp by an edge: it will slip out
  lift: 3,         // the held object lifted clear of the table — once per object
  place: 20,       // released into the bin
  miss: -2,        // the fingers closed on air
  drop: -5,        // the object fell back on the table (slipped, or released high)
  lost: -15,       // off the table
  bump: -1,        // a finger shoved the object
  closer: 0.1,     // per centimetre the fingers got closer (over the table) to an object than ever before (shaping: or
                   //   a learner that starts from nothing never finds the first grasp)
  align: 0.5,      // per centimetre closer than ever before while low, within the wrist sensor's reach (shaping: the correction)
  toBin: 0.1,      // per centimetre closer to the bin than ever before with an object in hand (shaping)
  reach: 2,        // the pre-grasp pose reached: fingertips low and within 1.5 cm of an object's centre (once per object)
  tick: -0.02,     // per decision: time is money
  done: 30,        // every object in the bin
};

/** The reward, step by step, for a sim that someone else drives (the viewer, or episode() below): make one per
 *  situation, call tick() at every decision and step(cmd) instead of sim.step(cmd); read .total and .parts. */
export function createScorer(sim) {
  const parts = Object.fromEntries(Object.keys(REWARD).map((k) => [k, 0]));
  const bestD = sim.objects.map(() => Infinity), bestA = sim.objects.map(() => Infinity), reached = sim.objects.map(() => false), once = { grasp: [], lift: [] };
  let bestB = Infinity;
  const sc = { parts, total: 0, since: 0, last: [] };          // since: the reward since the last decision; last: this step's (name, value) pairs
  const earn = (k, n = 1) => { const v = n * REWARD[k]; parts[k] += v; sc.total += v; sc.since += v; sc.last.push([k, v]); };
  sc.tick = () => { sc.since = 0; earn('tick'); };
  sc.step = (cmd) => {
    sc.last = [];
    // shaping: every centimetre of new progress pays a little — towards an object while empty-handed (more while low and
    // close: that is the correction), towards the bin while holding one; the pre-grasp pose once per object
    const tip = sim.arm.tip;
    if (sim.holding < 0) {
      for (const o of sim.objects) {
        if (o.state !== 'table') continue;
        const d = Math.hypot(o.x - tip.x, o.z - tip.z);
        if (d < bestD[o.i] - 0.0005) { if (bestD[o.i] < Infinity) earn('closer', (bestD[o.i] - d) * 100); bestD[o.i] = d; }
        if (tip.y < H.SENSOR && d < H.SENSOR_R && d < bestA[o.i] - 0.0005) { if (bestA[o.i] < Infinity) earn('align', (bestA[o.i] - d) * 100); bestA[o.i] = d; }
        if (!reached[o.i] && tip.y < 0.03 && d < 0.015) { reached[o.i] = true; earn('reach'); }
      }
      bestB = Infinity;
    } else {
      const d = Math.hypot(BIN.x - tip.x, BIN.z - tip.z);
      if (d < bestB - 0.0005) { if (bestB < Infinity) earn('toBin', (bestB - d) * 100); bestB = d; }
    }
    const events = sim.step(cmd);
    for (const e of events) {
      if (e.name === 'grasp' || e.name === 'lift') { if (!once[e.name][e.i]) { once[e.name][e.i] = true; earn(e.name); } }
      else if (e.name in REWARD) earn(e.name);
      if (e.name === 'bump' || e.name === 'drop' || e.name === 'setdown') bestD[e.i] = Infinity;   // it moved: the approach starts over
    }
    if (sim.done && !sc.finished) { sc.finished = true; earn('done'); }
    return events;
  };
  return sc;
}

/** One episode: a situation played with `policy(obs, s, mem) → action` (deciding every `every` frames; with
 *  record: true each decision is kept as { obs, a, r }: observation, action, the reward earned until the next decision)
 *  or a raw `bot(sim) → cmd` (`create: (opts) → bot` makes a fresh one, so nothing is carried over between episodes). */
export function episode(sim, { policy, bot, create, objects = 3, diff = 'hard', seconds = 30, seed = 1, every = 4, record = false } = {}) {
  sim.start({ objects, diff, seed });
  const play = bot || (create && create({ seconds }));
  const mem = createMemory(); mem.seconds = seconds;
  const obs = new Float32Array(FEATURES.length), samples = [], tEnd = Math.round(seconds * 60), sc = createScorer(sim);
  let a = A.wait, steps = 0;
  while (!sim.done && sim.frame < tEnd) {
    let cmd;
    if (policy) {
      const s = sense(sim, mem);
      if (sim.frame % every === 0) {
        if (record && samples.length) samples[samples.length - 1].r = sc.since;
        sc.tick(); mem.decF = sim.frame;
        a = policy(observe(s, mem, obs), s, mem);
        if (record) samples.push({ obs: Float32Array.from(obs), a, r: 0 });
        steps++;
      }
      cmd = act(sim, a, mem);
    } else { cmd = play(sim); if (sim.frame % every === 0) sc.tick(); }
    sc.step(cmd);
  }
  if (record && samples.length) samples[samples.length - 1].r = sc.since;
  const st = sim.stats;
  return { reward: sc.total, parts: sc.parts, steps, frames: sim.frame, placed: st.placed, misses: st.misses, drops: st.drops, lost: st.lost, objects: sim.objects.length, done: sim.done, samples: record ? samples : null };
}

/** n situations to play: 1–3 objects at random spots, a random camera bias (r: a seeded random source). Hard by
 *  default: on easy the open-loop rules already pick everything, and there is nothing to learn. */
export const situations = (r, n, { diff = 'hard', seconds = 30, objectsMax = 3 } = {}) =>
  Array.from({ length: n }, () => ({ seed: 1 + r.pick(1e9), objects: 1 + r.pick(objectsMax), diff, seconds }));

/** The mean reward of a policy (or a raw controller: { create }) over a list of situations, with the mean breakdown. */
export function evaluate(sim, who, sits, every = 4) {
  const parts = {}; let reward = 0, placed = 0, objects = 0, misses = 0, drops = 0, lost = 0, frames = 0;
  for (const sit of sits) {
    const ep = episode(sim, { ...sit, ...(typeof who === 'function' ? { policy: who } : who), every });
    reward += ep.reward; placed += ep.placed; objects += ep.objects; misses += ep.misses; drops += ep.drops; lost += ep.lost; frames += ep.frames;
    for (const k in ep.parts) parts[k] = (parts[k] || 0) + ep.parts[k] / sits.length;
  }
  return { reward: reward / sits.length, placed, objects, misses, drops, lost, seconds: frames / 60 / sits.length, parts };
}
export const fmt = (e) => `${e.reward.toFixed(1).padStart(7)}  (placed ${String(e.placed).padStart(2)}/${e.objects}, missed ${String(e.misses).padStart(3)}×, dropped ${e.drops}×${e.lost ? `, lost ${e.lost}` : ''}, ${e.seconds.toFixed(1)} s each)`;

const sizes = (hidden) => [FEATURES.length, ...(Array.isArray(hidden) ? hidden : [hidden]), ACTIONS.length];
const shuffle = (arr, r) => { for (let i = arr.length - 1; i > 0; i--) { const j = r.pick(i + 1); [arr[i], arr[j]] = [arr[j], arr[i]]; } return arr; };

/** The final score every lesson ends with: the trained controller against the rules, on eight fresh situations. */
export function score(sim, net, every, seed, seconds, report, diff = 'hard') {
  const sits = situations(rng(seed + 1000), 8, { seconds, diff });
  const e = evaluate(sim, netPolicy(net), sits, every), ref = evaluate(sim, teacher, sits, every), best = evaluate(sim, careful, sits, every);
  report({ stage: 'score', trained: e, rules: ref, careful: best, diff });
  return { score: +e.reward.toFixed(1), rules: +ref.reward.toFixed(1), careful: +best.reward.toFixed(1) };
}

// ---- Lesson 1 — imitation learning (behaviour cloning): watch the rules play, write down what they saw and what
// they did, and fit a network to copy it. Supervised learning, nothing more: inputs (observations) → labels (actions).
export function trainClone(sim, { episodes = 24, seconds = 30, epochs = 30, hidden = 32, lr = 0.003, balance = false, seed = 1, teach = 'naive', diff = 'hard', every = 4 } = {}, report = () => {}) {
  const r = rng(seed), who = teach === 'careful' ? careful : teacher;
  // 1. collect: the teacher plays, every decision is kept as (observation, action)
  const data = [];
  const sits = situations(r, episodes, { seconds, diff });
  sits.forEach((sit, i) => { data.push(...episode(sim, { policy: who, ...sit, every, record: true }).samples); report({ stage: 'collect', episode: i + 1, episodes, decisions: data.length }); });
  const count = ACTIONS.map((_, a) => data.filter((d) => d.a === a).length);
  report({ stage: 'actions', count, total: data.length });
  // class weights: rare actions count more (the square root keeps it mild)
  const w = count.map((c) => (balance && c ? Math.sqrt(data.length / ACTIONS.length / c) : 1));
  // 2. fit: minibatch gradient descent on the cross-entropy  −log p(teacher's action | observation)
  shuffle(data, r);
  const nTest = Math.floor(data.length / 10), test = data.slice(0, nTest), train = data.slice(nTest);
  const net = createNet(sizes(hidden), { seed }), adam = createAdam(net), grads = createGrads(net);
  const accuracy = (set) => set.filter((d) => argmax(forward(net, d.obs).at(-1)) === d.a).length / set.length;
  for (let e = 1; e <= epochs; e++) {
    let loss = 0;
    shuffle(train, r);
    for (let i = 0; i < train.length; i += 64) {
      for (const d of train.slice(i, i + 64)) {
        const acts = forward(net, d.obs), z = acts.at(-1);
        let m = -Infinity; for (const v of z) if (v > m) m = v;
        let s = 0; for (const v of z) s += Math.exp(v - m);
        loss -= w[d.a] * (z[d.a] - m - Math.log(s));
        backward(net, acts, d.a, w[d.a], grads);
      }
      adam.step(grads, lr);
    }
    report({ stage: 'epoch', epoch: e, epochs, loss: loss / train.length, train: accuracy(train), test: accuracy(test) });
  }
  // 3. score: fresh situations, the student against the teacher
  const sc = score(sim, net, every, seed, seconds, report, diff);
  return { net, meta: { every, algorithm: 'clone', teacher: teach, diff, episodes, decisions: data.length, epochs, hidden, balance, ...sc } };
}

// ---- Lesson 2 — evolution (a genetic algorithm, no gradients): a population of networks plays, the best few
// survive, their mutated copies fill the next generation. Nothing but "try, score, keep the best". On `normal` by
// default: there the open-loop grasp pays often enough for a random network to stumble on it and be selected; on
// `hard` the first grasp is too rare to find by chance (try it — and then try --from bots/evolve.json --diff hard).
export function trainEvolve(sim, { pop = 64, gens = 250, elite = 8, sigma = 0.15, episodes = 4, seconds = 30, hidden = 32, from = null, seed = 1, diff = 'normal', every = 4 } = {}, report = () => {}) {
  const r = rng(seed);
  let gen = Array.from({ length: pop }, (_, i) => (from ? (i ? mutate(from, sigma, r) : from) : createNet(sizes(hidden), { seed: seed * 100 + i })));
  let best = null;
  for (let g = 1; g <= gens; g++) {
    const sits = situations(r, episodes, { seconds, diff });                 // the same test for everyone this generation
    const scored = gen.map((net) => ({ net, e: evaluate(sim, netPolicy(net), sits, every) })).sort((a, b) => b.e.reward - a.e.reward);
    const ref = evaluate(sim, teacher, sits, every), mean = scored.reduce((s, x) => s + x.e.reward, 0) / pop;
    report({ stage: 'gen', gen: g, gens, best: scored[0].e, mean, rules: ref.reward });
    best = scored[0].net;
    gen = scored.slice(0, elite).map((x) => x.net);                          // the elites stay; the rest are mutated copies of a random elite
    while (gen.length < pop) gen.push(mutate(scored[r.pick(elite)].net, sigma, r));
  }
  const sc = score(sim, best, every, seed, seconds, report, diff);
  return { net: best, meta: { every, algorithm: 'evolve', diff, pop, gens, elite, sigma, episodes, from: from ? 'yes' : undefined, ...sc } };
}

// ---- Lesson 3 — reinforcement learning (REINFORCE, the simplest policy gradient): the controller plays by drawing
// actions from its own probabilities, then every action is made more likely in proportion to how much reward followed
// it (and less likely when less than usual followed). No teacher: only the reward says what is good.
export function trainReinforce(sim, { episodes = 24000, batch = 32, seconds = 30, gamma = 0.95, lr = 0.002, entropy = 0.02, epsilon = 0, hidden = 32, from = null, seed = 1, diff = 'hard', every = 4 } = {}, report = () => {}) {
  const r = rng(seed);
  const net = from || createNet(sizes(hidden), { seed });
  const adam = createAdam(net), grads = createGrads(net), explore = netPolicy(net, { sample: true, rng: r, epsilon });
  for (let done = 0; done < episodes; done += batch) {
    // play a batch of episodes, drawing actions from the policy's probabilities (exploration)
    const eps = situations(r, batch, { seconds, diff }).map((sit) => episode(sim, { policy: explore, ...sit, every, record: true }));
    // the return of each decision: the discounted sum of the rewards that followed it
    const all = [];
    for (const ep of eps) { let G = 0; for (let t = ep.samples.length - 1; t >= 0; t--) { G = ep.samples[t].r + gamma * G; all.push({ ...ep.samples[t], G }); } }
    // the advantage: the return against the batch's average, in units of its spread — "better than usual" is positive
    const mean = all.reduce((s, x) => s + x.G, 0) / all.length, sd = Math.sqrt(all.reduce((s, x) => s + (x.G - mean) ** 2, 0) / all.length) || 1;
    // the policy gradient: push up the log-probability of each action taken, weighted by its advantage
    for (const x of all) backward(net, forward(net, x.obs), x.a, (x.G - mean) / sd, grads, entropy);
    adam.step(grads, lr);
    const reward = eps.reduce((s, e) => s + e.reward, 0) / batch, placed = eps.reduce((s, e) => s + e.placed, 0), objects = eps.reduce((s, e) => s + e.objects, 0);
    const nudges = all.filter((x) => x.a >= A['nudge+x'] && x.a <= A['nudge-z']).length / all.length;
    report({ stage: 'batch', done: done + batch, episodes, reward, placed, objects, nudges });
  }
  const sc = score(sim, net, every, seed, seconds, report, diff);
  return { net, meta: { every, algorithm: 'reinforce', diff, episodes, batch, seconds, gamma, lr, entropy, epsilon, from: from ? 'yes' : undefined, ...sc } };
}
