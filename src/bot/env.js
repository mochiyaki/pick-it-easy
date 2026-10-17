// The controller's environment: what a policy sees and what it can do, in the shape machine learning likes — numbers
// in, a number out. Every controller of the arm (the hand-written rules in scripted.js, a neural network in nn.js,
// your own) sits between these two functions:
//
//   const s = sense(sim, mem)        the scan: which object we are after, where the camera says it is, what the wrist
//                                    sensor says, the gripper, the bin (per frame)
//   const obs = observe(s)           → Float32Array, one number per FEATURES entry, each about -1…1 (what it sees)
//   const a = policy(obs, s, mem)    → an index into ACTIONS (what it decides) — this is the part you train
//   const cmd = act(sim, a, mem)     → { goal, grip }: where the fingertips go and whether they open or close
//
// createEnvBot(policy) wires the four into a (sim) → cmd controller that the viewer and the trainer can drive.
// An action is a "where to", not a joint angle: "approach" means "above where the camera says the object is", and the
// arm's inverse kinematics does the rest — a policy never has to learn trigonometry, it learns *when* to do what, and,
// the point of this project, *when the camera is wrong* and a nudge is due. (Same design as the game's bot env: there
// "walk" is a path-finder and the bot learns when to walk, fight and roll.)
import { BIN, H, inBin } from '../sim/world.js';
import { ARM } from '../sim/arm.js';

// ---- the action set: what one decision can be
export const ACTIONS = [
  'approach',  // 0 go to the hover height above where the camera says the target is
  'descend',   // 1 straight down to the grasp height, where we are
  'grasp',     // 2 close the fingers
  'lift',      // 3 straight up to the travel height
  'carry',     // 4 to the travel height above the bin
  'release',   // 5 open the fingers
  'nudge+x',   // 6 six millimetres along +x, at this height       ← the corrections: the camera is wrong, the wrist
  'nudge-x',   // 7 six millimetres along -x                           sensor is right, move over
  'nudge+z',   // 8 six millimetres along +z
  'nudge-z',   // 9 six millimetres along -z
  'wait',      // 10 nothing new: keep going where we were going
];
export const A = Object.fromEntries(ACTIONS.map((n, i) => [n, i]));
export const NUDGE = 0.006;   // one nudge: 6 mm. A policy nudges while the sensor's offset is over 4 mm, so the residual is under 4 mm per axis

// ---- the observation: one number per line, each about -1…1. Change this list and every controller sees differently.
export const FEATURES = [
  'camDx',       // the camera's estimate of the target minus the fingertips, x / 15 cm (0: no target)
  'camDz',       // same, z
  'camDist',     // its length / 15 cm — "how far off the estimate are we"
  'height',      // the fingertips' height / 25 cm
  'wristSees',   // 1: the wrist sensor sees the target (we are low and within 6 cm)
  'wristDx',     // the sensor's offset to the target, x / 3 cm (the truth — this is the correction signal)
  'wristDz',     // same, z
  'wristDist',   // its length / 3 cm
  'aperture',    // the fingers, 0 closed … 1 open
  'closing',     // 1: the fingers are commanded closed
  'holding',     // 1: an object is between the fingers
  'slip',        // 1: and it is pinched by an edge (a force sensor would feel it) — it will fall
  'binDx',       // the bin minus the fingertips, x / 50 cm
  'binDz',       // same, z
  'overBin',     // 1: the fingertips are above the bin
  'moving',      // the tip's speed / its top speed
  'atGoal',      // 1: the tip has arrived where it was sent
  'size',        // the target's half-width / 3 cm (the camera reads sizes well enough)
  'left',        // objects still on the table / 3
  'misses',      // grasps that closed on air so far / 3
  'timeLeft',    // of the episode, 1 → 0
];
const F = Object.fromEntries(FEATURES.map((n, i) => [n, i]));

/** A controller's scratch state (per episode): the target, the smoothed camera estimate, the last decision. */
export const createMemory = () => ({ target: -1, cam: { x: 0, z: 0 }, camOk: false, decF: -1, nudgeF: -1, seconds: 30, s: null });

/** The scan: everything a decision could be based on, in world terms. Call once per frame; the result is kept on mem.s. */
export function sense(sim, mem) {
  const arm = sim.arm, tip = arm.tip, objs = sim.objects;
  // the target: the object we hold, else the nearest one still on the table (kept until it is binned or lost)
  let t = mem.target;
  if (sim.holding >= 0) t = sim.holding;
  else if (t < 0 || objs[t].state !== 'table') {
    t = -1; let bd = 1e9;
    for (const o of objs) if (o.state === 'table') { const d = Math.hypot(o.x - tip.x, o.z - tip.z); if (d < bd) { bd = d; t = o.i; } }
    mem.camOk = false;
  }
  mem.target = t;
  const o = t >= 0 ? objs[t] : null;
  // the camera's estimate, smoothed (a tracker): the per-frame noise is filtered, the calibration bias is not
  if (o && o.state === 'table') {
    if (!mem.camOk) { mem.cam.x = o.camX; mem.cam.z = o.camZ; mem.camOk = true; }
    else { mem.cam.x += 0.35 * (o.camX - mem.cam.x); mem.cam.z += 0.35 * (o.camZ - mem.cam.z); }
  }
  const w = sim.sensor, sees = !!(o && w.sees && w.i === t);
  const s = {
    sim, arm, tip, f: sim.frame, o, t,
    camDx: o ? mem.cam.x - tip.x : 0, camDz: o ? mem.cam.z - tip.z : 0,
    sees, wDx: sees ? w.dx : 0, wDz: sees ? w.dz : 0,
    holding: sim.holding >= 0, slip: sim.holding >= 0 && objs[sim.holding].slip,
    binDx: BIN.x - tip.x, binDz: BIN.z - tip.z, overBin: inBin(tip.x, tip.z),
    atGoal: Math.hypot(arm.goal.x - tip.x, arm.goal.y - tip.y, arm.goal.z - tip.z) < 0.003,
    left: objs.filter((q) => q.state === 'table').length,
    timeLeft: Math.max(0, 1 - sim.t / mem.seconds),
  };
  s.camDist = Math.hypot(s.camDx, s.camDz); s.wDist = Math.hypot(s.wDx, s.wDz);
  mem.s = s;
  return s;
}

/** The scan as numbers: the observation vector a learning policy is trained on. */
export function observe(s, mem, out = new Float32Array(FEATURES.length)) {
  const arm = s.arm, c = (v, m) => Math.max(-1, Math.min(1, v / m));
  out[F.camDx] = c(s.camDx, 0.15); out[F.camDz] = c(s.camDz, 0.15); out[F.camDist] = c(s.camDist, 0.15);
  out[F.height] = c(s.tip.y, 0.25);
  out[F.wristSees] = s.sees ? 1 : 0; out[F.wristDx] = c(s.wDx, 0.03); out[F.wristDz] = c(s.wDz, 0.03); out[F.wristDist] = c(s.wDist, 0.03);
  out[F.aperture] = arm.ap; out[F.closing] = arm.apCmd === 0 ? 1 : 0;
  out[F.holding] = s.holding ? 1 : 0; out[F.slip] = s.slip ? 1 : 0;
  out[F.binDx] = c(s.binDx, 0.5); out[F.binDz] = c(s.binDz, 0.5); out[F.overBin] = s.overBin ? 1 : 0;
  out[F.moving] = c(arm.vel, ARM.SPEED); out[F.atGoal] = s.atGoal ? 1 : 0;
  out[F.size] = s.o ? c(s.o.w, 0.03) : 0;
  out[F.left] = c(s.left, 3); out[F.misses] = c(s.sim.stats.misses, 3); out[F.timeLeft] = s.timeLeft;
  return out;
}

/** The decision as a command for this frame. Uses the scan on mem.s. A nudge is applied once per decision (mem.decF
 *  is the frame the current action was decided on; the loop that calls policy() sets it). */
export function act(sim, a, mem) {
  const s = mem.s, tip = sim.arm.tip;
  if (!s) return null;
  const cmd = { goal: null, grip: null };
  const go = (x, y, z) => { cmd.goal = { x, y, z }; };
  switch (a) {
    case A.approach: if (s.o && s.o.state === 'table') go(mem.cam.x, H.HOVER, mem.cam.z); else go(ARM.HOME.x, ARM.HOME.y, ARM.HOME.z); break;
    case A.descend: go(tip.x, H.GRASP, tip.z); break;
    case A.grasp: cmd.grip = 'close'; break;
    case A.lift: go(tip.x, H.TRAVEL, tip.z); break;
    case A.carry: go(BIN.x, H.TRAVEL, BIN.z); break;
    case A.release: cmd.grip = 'open'; break;
    case A['nudge+x']: case A['nudge-x']: case A['nudge+z']: case A['nudge-z']:
      if (mem.nudgeF !== mem.decF) {
        mem.nudgeF = mem.decF;
        const dx = a === A['nudge+x'] ? NUDGE : a === A['nudge-x'] ? -NUDGE : 0, dz = a === A['nudge+z'] ? NUDGE : a === A['nudge-z'] ? -NUDGE : 0;
        go(tip.x + dx, tip.y, tip.z + dz);
      }
      break;
    default: break;   // wait: the arm keeps going where it was going
  }
  return cmd;
}

/** A policy ((obs, s, mem) → ACTIONS index) as a controller ((sim) → cmd). `every`: frames between decisions (the
 *  action is held in between; 4 = fifteen decisions a second). `seconds`: the episode length the policy is told. */
export function createEnvBot(policy, { every = 4, seconds = 30 } = {}) {
  const mem = createMemory(); mem.seconds = seconds;
  let a = A.wait, last = -1e9;
  const obs = new Float32Array(FEATURES.length);
  const bot = (sim) => {
    const s = sense(sim, mem);
    if (sim.frame - last >= every) { last = sim.frame; mem.decF = sim.frame; a = policy(observe(s, mem, obs), s, mem); }
    bot.a = a; bot.obs = obs;
    return act(sim, a, mem);
  };
  bot.mem = mem; bot.a = a; bot.obs = obs;
  return bot;
}
