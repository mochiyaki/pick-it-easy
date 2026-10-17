// Lesson 2 — evolution (a genetic algorithm, no gradients): a population of random networks plays, the best few
// survive, their mutated copies fill the next generation. Nothing but "try, score, keep the best" — which is why it
// works on any controller, even one that is not a network, and why it is slow.
//
//   node train/evolve.mjs [--pop 64] [--gens 250] [--elite 8] [--sigma 0.15] [--episodes 4] [--seconds 30] [--hidden 32]
//                         [--from bots/clone.json] [--out bots/evolve.json] [--seed 1] [--diff normal]
//
// What to watch: the generation's best and mean reward against the rules' score on the same situations. Everyone in
// a generation plays the *same* situations (common random numbers), or luck would decide who survives. --from starts
// from a trained controller instead of random weights (evolution as fine-tuning). --sigma is the mutation size.
// It trains on `normal` by default — the one lesson that does: from random weights nothing is ever placed on `hard`
// (the first grasp pays too rarely to be selected for), while on normal the whole pick sequence appears around
// generation 100. `--from bots/evolve.json --diff hard --sigma 0.05` carries it on to hard (a curriculum) — whether it
// improves there is the exercise: in our run it did not, in 100 generations.
import { makeSim, saveBot, loadNet, arg, printer } from './lib.mjs';
import { trainEvolve } from './core.js';

const from = arg('from', '');
const opts = { pop: +arg('pop', 64), gens: +arg('gens', 250), elite: +arg('elite', 8), sigma: +arg('sigma', 0.15), episodes: +arg('episodes', 4), seconds: +arg('seconds', 30), hidden: +arg('hidden', 32), seed: +arg('seed', 1), diff: arg('diff', 'normal'), from: from ? (await loadNet(from)).net : null };
const { net, meta } = trainEvolve(makeSim(), opts, printer());
await saveBot(arg('out', 'bots/evolve.json'), net, { ...meta, from: from || undefined });
