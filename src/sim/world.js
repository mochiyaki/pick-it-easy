// The cell: a table with a few objects on it, a bin beside it, the arm, and two senses — a camera over the table whose
// calibration is off (every estimate is shifted by the same unknown few millimetres, plus a little noise each frame)
// and a short-range sensor in the wrist that sees the true offset of an object right under the fingers. The whole
// simulation is pure JavaScript and deterministic from a seed: no renderer, no clock, so a 30-second episode takes a
// millisecond in Node and the trainer can play thousands. The browser only draws it (src/view).
//
//   const sim = createSim()
//   sim.start({ objects: 3, diff: 'hard', seed: 7 })  a fresh situation: objects placed, the camera's bias drawn
//   sim.step({ goal: {x,y,z} | null, grip: 'open' | 'close' | null })   one 60 Hz step; sim.events: what just happened
//   sim.arm, sim.objects, sim.sensor, sim.holding, sim.stats, sim.done
//
// Physics, as little as will teach the lesson: a grasp is clean when the fingers close within half the object's width
// of its centre (the object centres itself between them); a grasp further off than that pinches an edge and the object
// slips out a second later; further still, the fingers close on air (a miss). A finger coming down on top of an object
// shoves it aside. Released over the bin, the object is placed; over the table it is dropped; anywhere else it is lost.
import { createArm, setGoal, stepArm, fk, ARM } from './arm.js';
const PAD = 0.008;   // half the width of a finger pad
import { rng as makeRng } from '../bot/nn.js';

export const TABLE = { x0: -0.30, x1: 0.30, z0: 0.14, z1: 0.46, thick: 0.03 };
export const SPAWN = { x0: -0.24, x1: 0.24, z0: 0.18, z1: 0.41, rMin: 0.17, rMax: 0.44, gap: 0.09 };
export const BIN = { x: -0.33, z: 0.10, r: 0.075, h: 0.06 };
export const H = { GRASP: 0.012, HOVER: 0.11, TRAVEL: 0.20, SENSOR: 0.13, SENSOR_R: 0.06, LIFTED: 0.08 };
export const KINDS = ['cube', 'cylinder', 'sphere'];
export const COLORS = ['#ff7a3a', '#46ff8a', '#38e8ff', '#ff3ea8', '#ffd75e', '#b08cff'];

// The difficulty is how far off the camera is. On easy the open-loop rules pick every object; on hard they miss most.
export const DIFFS = [
  { id: 'easy', name: 'Easy', bias: 0.006, noise: 0.001, slipFrames: 120, drift: 0 },
  { id: 'normal', name: 'Normal', bias: 0.016, noise: 0.002, slipFrames: 90, drift: 0 },
  { id: 'hard', name: 'Hard', bias: 0.030, noise: 0.0025, slipFrames: 60, drift: 0 },
  { id: 'extreme', name: 'Extremely hard', bias: 0.045, noise: 0.004, slipFrames: 40, drift: 0.6 },   // the bias wanders too
];
export const diffById = (id) => (typeof id === 'object' && id ? id : DIFFS.find((d) => d.id === id) || DIFFS[2]);

export const inBin = (x, z) => Math.hypot(x - BIN.x, z - BIN.z) < BIN.r;
export const onTable = (x, z) => x > TABLE.x0 && x < TABLE.x1 && z > TABLE.z0 && z < TABLE.z1;

export function createSim() {
  const sim = { frame: 0, t: 0, dt: 1 / 60, arm: createArm(), objects: [], diff: DIFFS[2], bias: { x: 0, z: 0 }, biasR: 0, biasA: 0, rng: makeRng(1), seed: 1,
    sensor: { i: -1, dx: 0, dz: 0, sees: false }, holding: -1, stats: null, events: [], log: [], done: false, params: null };

  sim.start = ({ objects = 3, diff = 'hard', seed = 1 } = {}) => {
    const r = sim.rng = makeRng(seed), D = sim.diff = diffById(diff);
    sim.seed = seed; sim.params = { objects, diff: D.id, seed };
    sim.frame = 0; sim.t = 0; sim.arm = createArm(); sim.holding = -1; sim.done = false; sim.events = []; sim.log = [];
    sim.stats = { placed: 0, grasps: 0, slips: 0, misses: 0, drops: 0, lost: 0, bumps: 0, frames: 0 };
    // the objects: a kind, a size, a free spot on the table
    sim.objects = [];
    for (let i = 0; i < objects; i++) {
      const kind = KINDS[r.pick(KINDS.length)], w = 0.015 + r() * 0.01, h = kind === 'cylinder' ? 2.4 * w : 2 * w;
      let x = 0, z = 0.3;
      for (let tries = 0; tries < 200; tries++) {
        x = SPAWN.x0 + r() * (SPAWN.x1 - SPAWN.x0); z = SPAWN.z0 + r() * (SPAWN.z1 - SPAWN.z0);
        const rr = Math.hypot(x, z);
        if (rr < SPAWN.rMin || rr > SPAWN.rMax) continue;
        if (sim.objects.every((o) => Math.hypot(o.x - x, o.z - z) >= SPAWN.gap)) break;
      }
      sim.objects.push({ i, kind, w, h, x, z, y: 0, color: COLORS[i % COLORS.length], state: 'table', camX: x, camZ: z, slip: false, off: { x: 0, z: 0 }, hang: 0, slipT: 0, lifted: false });
    }
    // the camera's calibration error: one shift for the whole episode, drawn uniformly from a disc of radius diff.bias
    sim.biasA = r() * 2 * Math.PI; sim.biasR = D.bias * Math.sqrt(r());
    sim.bias.x = sim.biasR * Math.cos(sim.biasA); sim.bias.z = sim.biasR * Math.sin(sim.biasA);
    perceive(sim);
    return sim;
  };

  sim.step = (cmd) => {
    const arm = sim.arm, dt = sim.dt, D = sim.diff;
    sim.events = [];
    if (cmd) {
      if (cmd.goal) setGoal(arm, cmd.goal.x, cmd.goal.y, cmd.goal.z);
      if (cmd.grip === 'close') arm.apCmd = 0; else if (cmd.grip === 'open') arm.apCmd = 1;
    }
    const ap0 = arm.ap;
    stepArm(arm, dt);
    const tip = arm.tip, held = sim.holding >= 0 ? sim.objects[sim.holding] : null;
    const emit = (name, i = -1) => { const e = { f: sim.frame, name, i }; sim.events.push(e); if (sim.log.length < 2000) sim.log.push(e); };

    if (held) {
      // the held object rides with the fingers
      held.x = tip.x + held.off.x; held.z = tip.z + held.off.z; held.y = tip.y - held.hang;
      if (!held.lifted && tip.y > H.LIFTED) { held.lifted = true; emit('lift', held.i); }
      if (arm.apCmd === 1) release(sim, held, emit, 'release');                       // let go
      else if (held.slip && tip.y > 0.03 && ++held.slipT > D.slipFrames) release(sim, held, emit, 'slip');   // it slips out
    } else if (arm.apCmd === 0 && ap0 > 0) {
      // closing: do the fingers catch anything? u: the offset along the fingers' axis (forgiven: the pads push the
      // object to the middle), v: across it (not forgiven: the pads grip an edge, or pass the object altogether)
      const halfGap = arm.ap * ARM.GRIP_W / 2, { axis, radial } = fk(arm);
      for (const o of sim.objects) {
        if (o.state !== 'table' || tip.y >= o.h - 0.003) continue;                 // the fingertips must be below the top
        const dx = o.x - tip.x, dz = o.z - tip.z, u = Math.abs(dx * axis.x + dz * axis.z), v = Math.abs(dx * radial.x + dz * radial.z);
        if (v > o.w + 0.004 || u > halfGap - 0.002) continue;                      // the pads pass by it / it is outside them
        if (halfGap > u + o.w) continue;                                           // not touching yet
        if (v < 0.5 * o.w) {                                                       // clean: the object centres itself
          o.state = 'held'; o.off.x = 0; o.off.z = 0; o.x = tip.x; o.z = tip.z; o.slip = false;
        } else {                                                                   // pinched by an edge: it will slip
          o.state = 'held'; o.off.x = radial.x * (dx * radial.x + dz * radial.z); o.off.z = radial.z * (dx * radial.x + dz * radial.z);
          o.x = tip.x + o.off.x; o.z = tip.z + o.off.z; o.slip = true;
        }
        o.hang = tip.y - o.y; o.slipT = 0; o.lifted = false; sim.holding = o.i;
        sim.stats.grasps++; if (o.slip) sim.stats.slips++;
        emit(o.slip ? 'slipGrasp' : 'grasp', o.i);
        break;
      }
      if (sim.holding < 0 && arm.ap === 0 && ap0 > 0) { sim.stats.misses++; emit('miss'); }
    }
    if (sim.holding >= 0) arm.ap = Math.max(arm.ap, 2 * sim.objects[sim.holding].w / ARM.GRIP_W);   // the fingers rest on the object
    // an open finger coming down on an object (or sweeping into it) shoves it out of the way, along the fingers' axis
    if (sim.holding < 0 && arm.apCmd === 1 && arm.ap > 0.5) {
      const halfGap = arm.ap * ARM.GRIP_W / 2, { axis, radial } = fk(arm);
      for (const o of sim.objects) {
        if (o.state !== 'table' || tip.y >= o.h) continue;
        const dx = o.x - tip.x, dz = o.z - tip.z, u = dx * axis.x + dz * axis.z, v = dx * radial.x + dz * radial.z;
        if (Math.abs(Math.abs(u) - halfGap) >= o.w + 0.004 || Math.abs(v) >= o.w + PAD) continue;
        const u2 = Math.sign(u || 1) * (halfGap + o.w + 0.008);
        o.x = tip.x + axis.x * u2 + radial.x * v; o.z = tip.z + axis.z * u2 + radial.z * v; sim.stats.bumps++; emit('bump', o.i);
        if (!onTable(o.x, o.z)) { o.state = 'lost'; o.y = -0.3; sim.stats.lost++; emit('lost', o.i); }
      }
    }
    // the camera drifts on the hardest setting: the bias turns slowly around the truth
    if (D.drift) { sim.biasA += D.drift * dt; sim.bias.x = sim.biasR * Math.cos(sim.biasA); sim.bias.z = sim.biasR * Math.sin(sim.biasA); }
    sim.frame++; sim.t = sim.frame * dt; sim.stats.frames = sim.frame;
    perceive(sim);
    sim.done = sim.objects.every((o) => o.state === 'bin' || o.state === 'lost');
    return sim.events;
  };
  return sim;
}

/** The object leaves the fingers: into the bin, onto the table, or off the edge. `how`: 'release' | 'slip'. */
function release(sim, o, emit, how) {
  sim.holding = -1; o.state = 'table'; o.slip = false; o.off.x = 0; o.off.z = 0;
  const fromHeight = o.y;
  o.y = 0;
  if (inBin(o.x, o.z)) { o.state = 'bin'; sim.stats.placed++; emit('place', o.i); return; }
  if (!onTable(o.x, o.z)) { o.state = 'lost'; o.y = -0.3; sim.stats.lost++; emit('lost', o.i); return; }
  if (how === 'slip' || fromHeight > 0.02) { sim.stats.drops++; emit('drop', o.i); } else emit('setdown', o.i);
}

/** The two senses, once per step: the camera's estimate of every object (bias + noise), the wrist sensor's truth. */
function perceive(sim) {
  const r = sim.rng, D = sim.diff, tip = sim.arm.tip;
  for (const o of sim.objects) { o.camX = o.x + sim.bias.x + r.gauss() * D.noise; o.camZ = o.z + sim.bias.z + r.gauss() * D.noise; }
  const s = sim.sensor; s.i = -1; s.sees = false; s.dx = 0; s.dz = 0;
  if (tip.y < H.SENSOR) {
    let best = H.SENSOR_R;
    for (const o of sim.objects) {
      if (o.state !== 'table') continue;
      const d = Math.hypot(o.x - tip.x, o.z - tip.z);
      if (d < best) { best = d; s.i = o.i; }
    }
    if (s.i >= 0) { const o = sim.objects[s.i]; s.sees = true; s.dx = o.x - tip.x + r.gauss() * 0.0003; s.dz = o.z - tip.z + r.gauss() * 0.0003; }
  }
}
