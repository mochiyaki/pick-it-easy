// The hand-written controller, as a policy: a few rules that read the scan and name an action (env.js ACTIONS). It is
// the classic open-loop pick: the camera says where the object is, the arm goes there, descends, grasps, lifts,
// carries, releases. It trusts the camera completely — so it is only as good as the calibration, and on `hard` it
// closes on air most of the time, reopens, and tries the very same spot again. That is the baseline every trained
// controller is measured against and the teacher train/clone.mjs copies, flaws included.
//   policy(obs, s, mem) → action     s: sense() — the rules read the scan directly; the numbers are for networks
// style 'careful' adds one rule — when the wrist sensor sees the object off-centre, nudge towards it before going
// down — which is the whole correction, written by hand. The lessons in train/ are about a model finding it alone.
import { A } from './env.js';
import { H } from '../sim/world.js';

export function scriptedPolicy({ style = 'naive' } = {}) {
  const careful = style === 'careful';
  return (obs, s) => {
    const arm = s.arm;
    // 1. an object in hand: up to travel height, over to the bin, let go
    if (s.holding) {
      if (s.overBin && s.tip.y > H.TRAVEL - 0.02) return A.release;
      if (s.tip.y < H.TRAVEL - 0.02 && !s.overBin) return A.lift;
      return A.carry;
    }
    // 2. the fingers are closing: wait for them; closed on air: open them again (then rule 4 goes back to the camera's spot)
    if (arm.apCmd === 0) return arm.ap === 0 ? A.release : A.wait;
    if (arm.ap < 0.9) return A.wait;
    if (!s.o) return A.approach;                                    // nothing left: approach() goes home
    // 3. (careful only) the wrist sensor sees the object off-centre: nudge towards it, largest offset first
    if (careful && s.sees && Math.max(Math.abs(s.wDx), Math.abs(s.wDz)) > 0.004) {   // more than two thirds of a nudge off
      if (Math.abs(s.wDx) >= Math.abs(s.wDz)) return s.wDx > 0 ? A['nudge+x'] : A['nudge-x'];
      return s.wDz > 0 ? A['nudge+z'] : A['nudge-z'];
    }
    // 4. not above the camera's estimate yet: go there (at hover height)
    if (s.camDist > 0.008 && !(careful && s.sees)) return A.approach;
    // 5. above it: straight down, then close
    if (s.tip.y > H.GRASP + 0.003) return A.descend;
    return A.grasp;
  };
}
