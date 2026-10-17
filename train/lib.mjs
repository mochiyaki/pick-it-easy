// The Node side of training: the sim, saving / loading bots/*.json, the command line, and printing a lesson's progress.
import { writeFile, readFile } from 'node:fs/promises';
import { createSim } from '../src/sim/world.js';
import { FEATURES, ACTIONS } from '../src/bot/env.js';
import { toJSON, fromJSON } from '../src/bot/nn.js';
export { episode, situations, evaluate, fmt, teacher, careful, REWARD } from './core.js';

export const arg = (k, d) => { const i = process.argv.indexOf('--' + k); return i > 0 ? process.argv[i + 1] : d; };
export const flag = (k) => process.argv.includes('--' + k);
export const makeSim = () => createSim();
export const mmss = (ms) => `${Math.floor(ms / 60000)}:${String(Math.floor(ms / 1000) % 60).padStart(2, '0')}`;

export async function saveBot(file, net, meta) {
  const j = toJSON(net, { name: file.split('/').pop().replace(/\.json$/, ''), every: meta.every, features: FEATURES, actions: ACTIONS, trained: { date: new Date().toISOString().slice(0, 10), ...meta } });
  await writeFile(file, JSON.stringify(j));
  console.log(`saved ${file}  (${(JSON.stringify(j).length / 1024).toFixed(0)} KB)`);
}
export async function loadNet(file) { const j = JSON.parse(await readFile(file, 'utf8')); return { net: fromJSON(j), json: j }; }

/** A reporter that prints each lesson's progress the way the game's trainers do. */
export function printer(t0 = performance.now()) {
  const { fmt } = printerDeps;
  return (p) => {
    const T = mmss(performance.now() - t0);
    switch (p.stage) {
      case 'collect': process.stdout.write(`\r  collecting ${p.decisions} decisions from ${p.episodes} episodes…`); break;
      case 'actions': console.log(`\n  actions: ${ACTIONS.map((n, a) => `${n} ${(100 * p.count[a] / p.total).toFixed(1)}%`).join(' · ')}`); break;
      case 'epoch': console.log(`  epoch ${String(p.epoch).padStart(2)}  loss ${p.loss.toFixed(3)}  accuracy ${(100 * p.train).toFixed(1)}% (train)  ${(100 * p.test).toFixed(1)}% (held out)`); break;
      case 'gen': console.log(`  gen ${String(p.gen).padStart(2)}  best ${fmt(p.best)}  mean ${p.mean.toFixed(1).padStart(7)}  rules ${p.rules.toFixed(1).padStart(7)}  ${T}`); break;
      case 'batch': console.log(`  episodes ${String(p.done).padStart(4)}  reward ${p.reward.toFixed(1).padStart(7)}  placed ${String(p.placed).padStart(2)}/${p.objects}  nudges ${(100 * p.nudges).toFixed(1)}%  ${T}`); break;
      case 'score': console.log(`\n  on ${p.diff}, 8 fresh situations:\n  rules    ${fmt(p.rules)}\n  careful  ${fmt(p.careful)}\n  trained  ${fmt(p.trained)}\n  ${T} wall`); break;
    }
  };
}
import * as printerDeps from './core.js';
