// Which controller drives the arm. The viewer (?bot=…, or a file dropped on the page), bench/run.mjs --bot and
// train/play.mjs all take the same spec:
//   'scripted'            the open-loop rules (scripted.js) — the baseline and the teacher
//   'careful'             the same rules with the hand-written wrist-sensor correction
//   'bots/<name>.json'    a trained network (nn.js weights, written by train/*.mjs); a bare name means bots/<name>.json
//   'bots/<name>.js'      your own module:  export function policy(obs, s, mem) { return action }   (+ export const every = 4)
//                         or, raw:          export function createBot() { return (sim) => cmd }
// A loaded controller is { name, create() → (sim) → cmd }: make a fresh one per episode.
import { ACTIONS, FEATURES, createEnvBot } from './env.js';
import { scriptedPolicy } from './scripted.js';
import { forward, fromJSON as netFromJSON, argmax, softmax, sample as draw } from './nn.js';

export const SCRIPTED = { name: 'scripted', every: 4, create: (o) => createEnvBot(scriptedPolicy(), { every: 4, ...o }) };
export const CAREFUL = { name: 'careful', every: 4, create: (o) => createEnvBot(scriptedPolicy({ style: 'careful' }), { every: 4, ...o }) };

/** A network as a policy: obs → the most likely action (greedy), or a draw from its probabilities (`sample`: exploration;
 *  with `epsilon`, that often a uniformly random action instead — so an action the network never tries still gets tried). */
export function netPolicy(net, { sample = false, rng = Math.random, epsilon = 0 } = {}) {
  const p = (obs) => {
    const logits = forward(net, obs).at(-1);
    p.probs = softmax(logits);
    if (!sample) return argmax(logits);
    return epsilon && rng() < epsilon ? Math.floor(rng() * logits.length) : draw(p.probs, rng);
  };
  p.net = net;
  return p;
}

/** A controller from a saved network (bots/*.json). Refuses one trained on another observation or action set. */
export function fromJSON(j, { name = j.name || 'network' } = {}) {
  const net = netFromJSON(j);
  if (net.sizes[0] !== FEATURES.length || net.sizes.at(-1) !== ACTIONS.length)
    throw new Error(`${name}: trained on ${net.sizes[0]} features × ${net.sizes.at(-1)} actions; the env has ${FEATURES.length} × ${ACTIONS.length} — retrain it`);
  if (j.features && j.features.some((f, i) => f !== FEATURES[i])) console.warn(`${name}: the feature names differ from env.js FEATURES — the controller may read the wrong numbers`);
  const every = j.every || 4;
  return { name, every, json: j, net, create: (o) => createEnvBot(netPolicy(net), { every, ...o }) };
}

/** A controller from a module (bots/*.js): `policy` on the env, or `createBot` for raw commands. */
export function fromModule(mod, { name = 'module' } = {}) {
  if (typeof mod.createBot === 'function') return { name, create: () => mod.createBot() };
  if (typeof mod.policy === 'function') return { name, every: mod.every || 4, create: (o) => createEnvBot(mod.policy, { every: mod.every || 4, ...o }) };
  throw new Error(`${name}: a bot module exports policy(obs, s, mem) or createBot()`);
}

export const specUrl = (spec) => (/[./]/.test(spec) ? spec : `bots/${spec}.json`);

/** Browser loader: a spec string (see the top), or a File dropped on the page. */
export async function loadBot(spec) {
  if (!spec || spec === 'scripted') return SCRIPTED;
  if (spec === 'careful') return CAREFUL;
  if (typeof File !== 'undefined' && spec instanceof File) {
    const name = spec.name.replace(/\.(json|m?js)$/, '');
    if (/\.json$/i.test(spec.name)) return fromJSON(JSON.parse(await spec.text()), { name });
    return fromModule(await import(URL.createObjectURL(spec)), { name });
  }
  const url = specUrl(spec), name = url.split('/').pop().replace(/\.(json|m?js)$/, '');
  if (/\.m?js$/i.test(url)) return fromModule(await import(new URL(url, location.href).href), { name });
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${url}: ${res.status}`);
  return fromJSON(await res.json(), { name });
}
