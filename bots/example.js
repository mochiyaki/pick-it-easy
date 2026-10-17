// The smallest controller you can plug in: a policy on the env (src/bot/env.js) — the numbers in, an action out.
// It is the closed-loop pick written by hand: the same open-loop sequence as the rules, plus "if the wrist sensor
// sees the object off-centre, nudge towards it" — and "if the grasp is slipping while still low, let go and try
// again". Try it:  ?bot=bots/example.js  in the viewer, or  node train/play.mjs --bot bots/example.js
// obs: Float32Array, one entry per FEATURES name; s: the raw scan (see env.js sense()); mem: the scratch state.
// Return an index into ACTIONS: use A['<name>'].
import { A, FEATURES } from '../src/bot/env.js';
const F = Object.fromEntries(FEATURES.map((n, i) => [n, i]));

export const every = 4;                                       // decide every 4 frames (15 times a second)
export function policy(obs) {
  if (obs[F.holding]) {
    if (obs[F.slip] && obs[F.height] < 0.1) return A.release;   // pinched by an edge and still low: set it down, regrasp
    if (obs[F.overBin] && obs[F.height] > 0.7) return A.release;
    if (obs[F.height] < 0.7 && !obs[F.overBin]) return A.lift;
    return A.carry;
  }
  if (obs[F.closing]) return obs[F.aperture] === 0 ? A.release : A.wait;   // closing: wait; closed on air: open again
  if (obs[F.aperture] < 0.9) return A.wait;
  if (obs[F.wristSees] && Math.max(Math.abs(obs[F.wristDx]), Math.abs(obs[F.wristDz])) > 0.133) {   // > 4 mm off (two thirds of a nudge): nudge, the larger axis first
    if (Math.abs(obs[F.wristDx]) >= Math.abs(obs[F.wristDz])) return obs[F.wristDx] > 0 ? A['nudge+x'] : A['nudge-x'];
    return obs[F.wristDz] > 0 ? A['nudge+z'] : A['nudge-z'];
  }
  if (!obs[F.wristSees] && obs[F.camDist] > 0.05) return A.approach;   // not there yet: where the camera says
  if (obs[F.height] > 0.06) return A.descend;                          // above the grasp height: down
  return A.grasp;
}
