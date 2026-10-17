// Score a controller: it plays a handful of situations and the mean reward is printed against another controller's
// on the very same situations — the open-loop rules by default.
//
//   node train/play.mjs --bot bots/clone.json [--vs scripted|careful] [--episodes 8] [--seconds 30] [--diff hard] [--seed 7] [--show]
import { loadBotFile } from '../bench/bot.mjs';
import { makeSim, situations, episode, evaluate, arg, flag, fmt } from './lib.mjs';
import { rng } from '../src/bot/nn.js';

const sim = makeSim(), r = rng(+arg('seed', 7));
const sits = situations(r, +arg('episodes', 8), { seconds: +arg('seconds', 30), diff: arg('diff', 'hard') });
const who = await loadBotFile(arg('bot', 'scripted')), vs = await loadBotFile(arg('vs', 'scripted'));
const as = (b) => ({ create: b.create });
if (flag('show')) for (const sit of sits) {
  const e = episode(sim, { ...sit, ...as(who) });
  console.log(`  ${String(sit.objects)} object${sit.objects > 1 ? 's' : ' '} seed ${String(sit.seed).padStart(10)}: ${e.reward.toFixed(1).padStart(7)}  placed ${e.placed}/${e.objects}  missed ${String(e.misses).padStart(2)}×  dropped ${e.drops}×${e.lost ? `  lost ${e.lost}` : ''}  ${(e.frames / 60).toFixed(1)} s`);
}
const a = evaluate(sim, as(who), sits), b = evaluate(sim, as(vs), sits);
console.log(`  ${who.name.padEnd(10)} ${fmt(a)}\n  ${vs.name.padEnd(10)} ${fmt(b)}`);
