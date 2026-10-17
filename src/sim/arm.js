// The arm: a 4-joint robot (base yaw, shoulder, elbow, wrist) with a two-finger gripper that always points down, in
// metres and radians. Pure maths, no three.js: the same file runs the headless trainer in Node and the 3D view.
//
//   const arm = createArm()                    at the home pose, gripper open
//   setGoal(arm, x, y, z)                      where the fingertips should go (clamped to what the arm can reach)
//   stepArm(arm, dt)                           the tip moves towards the goal at ≤ SPEED, the joints follow (inverse
//                                              kinematics), the fingers open / close towards arm.apCmd
//   fk(arm)                                    the joints' positions in the world: { S, E, W, T } (for drawing)
//
// Kinematics: the base turns (yaw) so the arm's plane contains the target; in that plane the upper arm (L1) and the
// forearm (L2) are a two-link chain solved in closed form (elbow up); the wrist (L3, vertical) keeps the gripper
// pointing at the table. "Where to" is a point; the maths is the arm's job, never the policy's — the same split as
// the game, where a bot says "walk to the objective" and the env runs the path-finding.
export const ARM = {
  SH: 0.10,        // shoulder height above the table
  L1: 0.28,        // upper arm
  L2: 0.24,        // forearm
  L3: 0.10,        // wrist joint → fingertips (always vertical)
  SPEED: 0.45,     // m/s, the tip's top speed
  ACCEL: 3.0,      // m/s², how fast it gets there (a real arm ramps)
  FINE: 0.15,      // m/s, the creep speed for the last 2 cm of a move (a 6 mm nudge is done within 3 frames: inside one decision)
  FINE_D: 0.02,
  GRIP_W: 0.08,    // the fingers fully open, tip to tip
  GRIP_SPEED: 5,   // aperture units per second: 0.2 s to close
  MIN_R: 0.10,     // the tip can't come closer than this to the base axis
  MAX_D: 0.515,    // shoulder → wrist, the chain almost straight (L1 + L2 = 0.52)
  HOME: { x: 0, y: 0.22, z: 0.22 },
};

export function createArm() {
  const arm = { yaw: 0, q1: 0, q2: 0, q3: 0, tip: { x: 0, y: 0, z: 0 }, goal: { x: 0, y: 0, z: 0 }, vel: 0, ap: 1, apCmd: 1, clamped: false };
  setGoal(arm, ARM.HOME.x, ARM.HOME.y, ARM.HOME.z);
  arm.tip.x = arm.goal.x; arm.tip.y = arm.goal.y; arm.tip.z = arm.goal.z;
  solve(arm);
  return arm;
}

/** Clamp a tip position into the reachable set: above the table, not too close to the base, within the chain's reach. */
export function clampReach(p) {
  let { x, y, z } = p, clamped = false;
  if (y < 0.004) { y = 0.004; clamped = true; }
  let r = Math.hypot(x, z);
  if (r < ARM.MIN_R) { if (r < 1e-6) { x = 0; z = ARM.MIN_R; } else { x *= ARM.MIN_R / r; z *= ARM.MIN_R / r; } r = ARM.MIN_R; clamped = true; }
  const h = y + ARM.L3 - ARM.SH, D = Math.hypot(r, h);
  if (D > ARM.MAX_D) {                       // too far: pull the wrist back along the shoulder → wrist line
    const k = ARM.MAX_D / D, r2 = r * k, h2 = h * k;
    x *= r2 / r; z *= r2 / r; y = Math.max(0.004, h2 + ARM.SH - ARM.L3); clamped = true;
  }
  return { x, y, z, clamped };
}

export function setGoal(arm, x, y, z) {
  const c = clampReach({ x, y, z });
  arm.goal.x = c.x; arm.goal.y = c.y; arm.goal.z = c.z; arm.clamped = c.clamped;
  return c;
}

/** Inverse kinematics for the current tip: yaw, shoulder (q1, from horizontal), elbow (q2, relative), wrist (q3). */
export function solve(arm) {
  const { x, y, z } = arm.tip, r = Math.hypot(x, z), h = y + ARM.L3 - ARM.SH, D = Math.min(Math.hypot(r, h), ARM.MAX_D);
  arm.yaw = Math.atan2(x, z);
  const phi = Math.atan2(h, r);
  const cA = Math.max(-1, Math.min(1, (D * D + ARM.L1 * ARM.L1 - ARM.L2 * ARM.L2) / (2 * D * ARM.L1)));   // at the shoulder
  const cB = Math.max(-1, Math.min(1, (D * D + ARM.L2 * ARM.L2 - ARM.L1 * ARM.L1) / (2 * D * ARM.L2)));   // at the wrist
  const q1 = phi + Math.acos(cA), a2 = phi - Math.acos(cB);                                              // elbow up
  arm.q1 = q1; arm.q2 = a2 - q1; arm.q3 = -Math.PI / 2 - a2;
}

/** Forward kinematics: shoulder S, elbow E, wrist W, fingertips T, plus the finger axis (unit, in the table plane). */
export function fk(arm) {
  const sy = Math.sin(arm.yaw), cy = Math.cos(arm.yaw), a2 = arm.q1 + arm.q2;
  const S = { x: 0, y: ARM.SH, z: 0 };
  const E = { x: S.x + sy * ARM.L1 * Math.cos(arm.q1), y: S.y + ARM.L1 * Math.sin(arm.q1), z: S.z + cy * ARM.L1 * Math.cos(arm.q1) };
  const W = { x: E.x + sy * ARM.L2 * Math.cos(a2), y: E.y + ARM.L2 * Math.sin(a2), z: E.z + cy * ARM.L2 * Math.cos(a2) };
  const T = { x: W.x, y: W.y - ARM.L3, z: W.z };
  return { S, E, W, T, axis: { x: cy, z: -sy }, radial: { x: sy, z: cy } };   // the fingers close across the arm's plane (axis); radial: along it
}

/** One fixed step: the tip towards the goal (trapezoid speed profile), the fingers towards the command. */
export function stepArm(arm, dt) {
  const dx = arm.goal.x - arm.tip.x, dy = arm.goal.y - arm.tip.y, dz = arm.goal.z - arm.tip.z, d = Math.hypot(dx, dy, dz);
  if (d < 1e-5) { arm.tip.x = arm.goal.x; arm.tip.y = arm.goal.y; arm.tip.z = arm.goal.z; arm.vel = 0; }
  else {
    const vStop = Math.sqrt(2 * ARM.ACCEL * d);                                   // the speed from which it can still stop at the goal
    arm.vel = Math.min(ARM.SPEED, arm.vel + ARM.ACCEL * dt, vStop);
    if (d <= ARM.FINE_D) arm.vel = Math.max(arm.vel, ARM.FINE);                  // fine positioning: a steady creep, so a 1 cm nudge takes 4 frames
    const s = Math.min(d, arm.vel * dt), k = s / d;
    arm.tip.x += dx * k; arm.tip.y += dy * k; arm.tip.z += dz * k;
    if (s >= d - 1e-6) arm.vel = 0;
  }
  solve(arm);
  const da = arm.apCmd - arm.ap, m = ARM.GRIP_SPEED * dt;
  arm.ap = Math.abs(da) <= m ? arm.apCmd : arm.ap + Math.sign(da) * m;
}
