// The Train panel's worker: runs one lesson (train/core.js, the same code as train/*.mjs) off the page's thread and
// posts its progress; the result is the network as bots/*.json would hold it.
//   postMessage({ algo: 'clone' | 'reinforce' | 'evolve', opts, from: <bot json> | null })
//   ← { stage, ... } progress (see core.js report()), then { stage: 'done', json, meta }
import { createSim } from '../src/sim/world.js';
import { FEATURES, ACTIONS } from '../src/bot/env.js';
import { toJSON, fromJSON } from '../src/bot/nn.js';
import { trainClone, trainEvolve, trainReinforce } from './core.js';

self.onmessage = ({ data }) => {
  const { algo, opts, from } = data, sim = createSim(), report = (p) => postMessage(p);
  try {
    const o = { ...opts, from: from ? fromJSON(from) : null, every: from && from.every || 4 };
    const { net, meta } = algo === 'clone' ? trainClone(sim, o, report) : algo === 'evolve' ? trainEvolve(sim, o, report) : trainReinforce(sim, o, report);
    const name = `${algo}-${new Date().toISOString().slice(0, 10)}`;
    postMessage({ stage: 'done', meta, json: toJSON(net, { name, every: meta.every, features: FEATURES, actions: ACTIONS, trained: { date: new Date().toISOString().slice(0, 10), ...meta, from: from ? from.name : undefined } }) });
  } catch (e) { postMessage({ stage: 'error', message: String(e && e.message || e) }); }
};
