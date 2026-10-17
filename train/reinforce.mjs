// Lesson 3 — reinforcement learning (REINFORCE, the simplest policy gradient): the controller plays by drawing
// actions from its own probabilities, then every action is made more likely in proportion to how much reward followed
// it. No teacher: only the reward (core.js REWARD) says what is good. Started from the imitation controller
// (--from bots/clone.json) it begins as the open-loop rules and *discovers the correction*: the nudges, never seen in
// the teacher's data, turn a miss into a clean grasp, and the reward says so.
//
//   node train/reinforce.mjs [--episodes 24000] [--batch 32] [--seconds 30] [--gamma 0.95] [--lr 0.002] [--entropy 0.02]
//                            [--epsilon 0] [--hidden 32] [--from bots/clone.json] [--out bots/reinforce.json] [--seed 1] [--diff hard]
//
// What to watch: the mean reward per batch (noisy; it can get worse before it gets better) and the share of nudges
// among the decisions — the correction being learned. --gamma is the discount: how far ahead a reward still counts
// for an action (0.95 at 15 decisions a second ≈ 1.3 s: a nudge is paid by the grasp that follows it, not by the whole shift). --entropy keeps some randomness in the policy; --lr the step.
import { makeSim, saveBot, loadNet, arg, printer } from './lib.mjs';
import { trainReinforce } from './core.js';

const from = arg('from', '');
const start = from ? await loadNet(from) : null;
const opts = { episodes: +arg('episodes', 24000), batch: +arg('batch', 32), seconds: +arg('seconds', 30), gamma: +arg('gamma', 0.95), lr: +arg('lr', 0.002), entropy: +arg('entropy', 0.02), epsilon: +arg('epsilon', 0), hidden: +arg('hidden', 32), seed: +arg('seed', 1), diff: arg('diff', 'hard'), from: start ? start.net : null, every: start ? start.json.every || 4 : 4 };
const { net, meta } = trainReinforce(makeSim(), opts, printer());
await saveBot(arg('out', 'bots/reinforce.json'), net, { ...meta, from: from || undefined });
