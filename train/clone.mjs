// Lesson 1 — imitation learning (behaviour cloning): the open-loop rules play, every decision is written down as
// (observation, action), and a network is fitted to copy it. It copies the teacher's habits, mistakes included: it
// trusts the camera just as blindly, so on `hard` it misses just as often. It can never be better than the teacher —
// that is what lessons 2 and 3 are for.
//
//   node train/clone.mjs [--episodes 24] [--seconds 30] [--epochs 30] [--hidden 32] [--lr 0.003] [--balance]
//                        [--teach careful] [--out bots/clone.json] [--seed 1] [--diff hard]
//
// What to watch: the loss falls and the accuracy rises epoch by epoch — on the held-out 10 % too, or it is memorising.
// --teach careful copies the rules *with* the hand-written correction instead: then the student nudges too (and the
// action balance it prints shows how rare a nudge is — try --balance).
import { makeSim, saveBot, arg, flag, printer } from './lib.mjs';
import { trainClone } from './core.js';

const opts = { episodes: +arg('episodes', 24), seconds: +arg('seconds', 30), epochs: +arg('epochs', 30), hidden: +arg('hidden', 32), lr: +arg('lr', 0.003), balance: flag('balance'), teach: arg('teach', 'naive'), seed: +arg('seed', 1), diff: arg('diff', 'hard') };
const { net, meta } = trainClone(makeSim(), opts, printer());
await saveBot(arg('out', 'bots/clone.json'), net, meta);
