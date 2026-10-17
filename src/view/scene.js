// The 3D view of the cell: the arm drawn from its joint positions, the objects, the bin, and the two senses made
// visible — the camera's estimate as a ghost outline beside each object (the calibration error you can see) and the
// wrist sensor as a cone of light under the fingers when it sees something. Reads the sim; never writes it.
import * as THREE from 'three';
import { fk, ARM } from '../sim/arm.js';
import { TABLE, BIN, H } from '../sim/world.js';

const col = (c) => new THREE.Color(c);
const mat = (c, o = {}) => new THREE.MeshStandardMaterial({ color: col(c), roughness: 0.55, metalness: 0.15, ...o });

export function createScene(canvas) {
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, powerPreference: 'high-performance' });
  renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
  renderer.shadowMap.enabled = true; renderer.shadowMap.type = THREE.PCFShadowMap;
  renderer.toneMapping = THREE.ACESFilmicToneMapping; renderer.toneMappingExposure = 1.05;
  const scene = new THREE.Scene();
  scene.background = col('#07090f'); scene.fog = new THREE.Fog(col('#07090f'), 2.2, 5);
  const camera = new THREE.PerspectiveCamera(42, 1, 0.02, 20);

  // light: a cool key from above-front, a warm fill, a soft ambient
  scene.add(new THREE.HemisphereLight(col('#8fb4c4'), col('#101418'), 0.55));
  const key = new THREE.DirectionalLight(col('#e8fbff'), 2.0); key.position.set(0.6, 1.6, 0.9); key.castShadow = true;
  key.shadow.mapSize.set(2048, 2048); key.shadow.camera.near = 0.5; key.shadow.camera.far = 4;
  key.shadow.camera.left = key.shadow.camera.bottom = -0.8; key.shadow.camera.right = key.shadow.camera.top = 0.8; key.shadow.bias = -0.0005;
  scene.add(key);
  const fill = new THREE.DirectionalLight(col('#ffb02e'), 0.5); fill.position.set(-1, 0.6, -0.5); scene.add(fill);

  // the floor and the bench
  const floor = new THREE.Mesh(new THREE.PlaneGeometry(8, 8), mat('#0b0e15', { roughness: 0.9 })); floor.rotation.x = -Math.PI / 2; floor.position.y = -0.4; floor.receiveShadow = true; scene.add(floor);
  const grid = new THREE.GridHelper(8, 80, col('#1a2230'), col('#121823')); grid.position.y = -0.399; scene.add(grid);
  const tw = TABLE.x1 - TABLE.x0, td = TABLE.z1 - TABLE.z0;
  const table = new THREE.Mesh(new THREE.BoxGeometry(tw, TABLE.thick, td), mat('#2a3340', { roughness: 0.8 }));
  table.position.set((TABLE.x0 + TABLE.x1) / 2, -TABLE.thick / 2, (TABLE.z0 + TABLE.z1) / 2); table.receiveShadow = true; table.castShadow = true; scene.add(table);
  const top = new THREE.Mesh(new THREE.PlaneGeometry(tw, td), new THREE.MeshStandardMaterial({ color: col('#38506a'), roughness: 0.95, transparent: true, opacity: 0.35 }));
  top.rotation.x = -Math.PI / 2; top.position.set(table.position.x, 0.0005, table.position.z); scene.add(top);
  const tgrid = new THREE.GridHelper(0.6, 12, col('#4a6a8a'), col('#2e4458')); tgrid.scale.z = td / tw; tgrid.position.set(table.position.x, 0.001, table.position.z); scene.add(tgrid);
  for (const [x, z] of [[TABLE.x0 + 0.03, TABLE.z0 + 0.03], [TABLE.x1 - 0.03, TABLE.z0 + 0.03], [TABLE.x0 + 0.03, TABLE.z1 - 0.03], [TABLE.x1 - 0.03, TABLE.z1 - 0.03]]) {
    const leg = new THREE.Mesh(new THREE.BoxGeometry(0.03, 0.37, 0.03), mat('#1c2430')); leg.position.set(x, -0.215, z); leg.castShadow = true; scene.add(leg);
  }
  // the base plinth
  const plinth = new THREE.Mesh(new THREE.CylinderGeometry(0.11, 0.13, 0.4, 32), mat('#1c2430')); plinth.position.y = -0.2; plinth.castShadow = true; plinth.receiveShadow = true; scene.add(plinth);
  // the bin: a ring of wall segments and a floor, with a glowing rim
  const bin = new THREE.Group(); bin.position.set(BIN.x, 0, BIN.z); scene.add(bin);
  const wall = new THREE.Mesh(new THREE.CylinderGeometry(BIN.r + 0.004, BIN.r - 0.004, BIN.h, 40, 1, true), new THREE.MeshStandardMaterial({ color: col('#243244'), roughness: 0.6, side: THREE.DoubleSide }));
  wall.position.y = BIN.h / 2 - 0.1; wall.castShadow = true; bin.add(wall);
  const bfloor = new THREE.Mesh(new THREE.CylinderGeometry(BIN.r, BIN.r, 0.01, 40), mat('#1a2432')); bfloor.position.y = -0.105; bin.add(bfloor);
  const stand = new THREE.Mesh(new THREE.CylinderGeometry(0.02, 0.03, 0.3, 16), mat('#1c2430')); stand.position.y = -0.26; bin.add(stand);
  const rim = new THREE.Mesh(new THREE.TorusGeometry(BIN.r, 0.004, 8, 48), new THREE.MeshBasicMaterial({ color: col('#46ff8a') })); rim.rotation.x = Math.PI / 2; rim.position.y = BIN.h - 0.1; bin.add(rim);

  // the arm: links drawn as capsules between the joints, joints as spheres, a palm and two fingers
  const armMat = mat('#d8e4ee', { roughness: 0.35, metalness: 0.5 }), jointMat = mat('#2a3a4c', { roughness: 0.3, metalness: 0.7 }), accent = mat('#38e8ff', { emissive: col('#38e8ff'), emissiveIntensity: 0.4 });
  const link = (r) => { const m = new THREE.Mesh(new THREE.CylinderGeometry(r, r, 1, 20), armMat); m.castShadow = true; scene.add(m); return m; };
  const ball = (r, mm = jointMat) => { const m = new THREE.Mesh(new THREE.SphereGeometry(r, 24, 16), mm); m.castShadow = true; scene.add(m); return m; };
  const base = new THREE.Mesh(new THREE.CylinderGeometry(0.075, 0.09, 0.07, 32), jointMat); base.position.y = 0.035; base.castShadow = true; scene.add(base);
  const turret = new THREE.Mesh(new THREE.BoxGeometry(0.09, 0.07, 0.09), armMat); turret.position.y = 0.075; turret.castShadow = true; scene.add(turret);
  const upper = link(0.026), fore = link(0.021), wristL = link(0.014);
  const jS = ball(0.036), jE = ball(0.03), jW = ball(0.024);
  const palm = new THREE.Mesh(new THREE.BoxGeometry(0.1, 0.022, 0.03), jointMat); palm.castShadow = true; scene.add(palm);
  const fingerGeo = new THREE.BoxGeometry(0.008, 0.05, 0.016);
  const f1 = new THREE.Mesh(fingerGeo, accent), f2 = new THREE.Mesh(fingerGeo, accent); f1.castShadow = f2.castShadow = true; scene.add(f1, f2);
  const ring = new THREE.Mesh(new THREE.TorusGeometry(0.03, 0.003, 8, 32), new THREE.MeshBasicMaterial({ color: col('#38e8ff') })); scene.add(ring);   // the status ring on the wrist

  // the wrist sensor: a cone of light under the fingers while it sees, and a marker on what it sees
  const cone = new THREE.Mesh(new THREE.ConeGeometry(H.SENSOR_R, 1, 32, 1, true), new THREE.MeshBasicMaterial({ color: col('#ffb02e'), transparent: true, opacity: 0.12, side: THREE.DoubleSide, depthWrite: false }));
  cone.visible = false; scene.add(cone);
  const seen = new THREE.Mesh(new THREE.RingGeometry(0.006, 0.009, 24), new THREE.MeshBasicMaterial({ color: col('#ffb02e'), side: THREE.DoubleSide, depthWrite: false }));
  seen.rotation.x = -Math.PI / 2; seen.visible = false; scene.add(seen);
  // the goal: where the fingertips were sent
  const goalMark = new THREE.Mesh(new THREE.OctahedronGeometry(0.007), new THREE.MeshBasicMaterial({ color: col('#46ff8a'), wireframe: true })); scene.add(goalMark);
  // the shadow line from the fingertips down to the table
  const plumbGeo = new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(), new THREE.Vector3()]);
  const plumb = new THREE.Line(plumbGeo, new THREE.LineDashedMaterial({ color: col('#38e8ff'), dashSize: 0.01, gapSize: 0.008, transparent: true, opacity: 0.5 })); scene.add(plumb);
  const plumbDot = new THREE.Mesh(new THREE.RingGeometry(0.004, 0.006, 20), new THREE.MeshBasicMaterial({ color: col('#38e8ff'), side: THREE.DoubleSide, depthWrite: false })); plumbDot.rotation.x = -Math.PI / 2; scene.add(plumbDot);

  // the objects and their camera ghosts (built per situation)
  let objMeshes = [], ghosts = [], binCount = 0;
  const geoFor = (o) => o.kind === 'cube' ? new THREE.BoxGeometry(2 * o.w, o.h, 2 * o.w) : o.kind === 'cylinder' ? new THREE.CylinderGeometry(o.w, o.w, o.h, 28) : new THREE.SphereGeometry(o.w, 28, 20);
  const build = (sim) => {
    for (const m of [...objMeshes, ...ghosts]) { scene.remove(m); m.geometry.dispose(); }
    objMeshes = []; ghosts = []; binCount = 0;
    for (const o of sim.objects) {
      const m = new THREE.Mesh(geoFor(o), mat(o.color, { roughness: 0.4 })); m.castShadow = true; m.receiveShadow = true; scene.add(m); objMeshes.push(m);
      const g = new THREE.LineSegments(new THREE.EdgesGeometry(geoFor(o)), new THREE.LineBasicMaterial({ color: col('#ff3ea8'), transparent: true, opacity: 0.9 }));
      g.add(new THREE.Mesh(geoFor(o), new THREE.MeshBasicMaterial({ color: col('#ff3ea8'), transparent: true, opacity: 0.18, depthWrite: false })));   // a faint fill, so the ghost reads from any angle
      scene.add(g); ghosts.push(g);
    }
  };

  const v = new THREE.Vector3(), up = new THREE.Vector3(0, 1, 0), q = new THREE.Quaternion();
  const between = (m, a, b) => { v.set(b.x - a.x, b.y - a.y, b.z - a.z); const l = v.length(); m.position.set((a.x + b.x) / 2, (a.y + b.y) / 2, (a.z + b.z) / 2); m.scale.set(1, l, 1); m.quaternion.setFromUnitVectors(up, v.normalize()); };

  /** Draw the sim's current state. ui: { target, cam: {x,z}, camOk, action } from the controller's memory (ghost colours). */
  const draw = (sim, ui = {}) => {
    if (objMeshes.length !== sim.objects.length) build(sim);
    const arm = sim.arm, k = fk(arm), yaw = arm.yaw;
    turret.rotation.y = yaw; base.rotation.y = yaw;
    between(upper, k.S, k.E); between(fore, k.E, k.W); between(wristL, k.W, { x: k.T.x, y: k.T.y + 0.04, z: k.T.z });
    jS.position.copy(k.S); jE.position.copy(k.E); jW.position.copy(k.W);
    palm.position.set(k.T.x, k.T.y + 0.05, k.T.z); palm.rotation.y = yaw;
    const half = arm.ap * ARM.GRIP_W / 2 + 0.004;
    f1.position.set(k.T.x + k.axis.x * half, k.T.y + 0.025, k.T.z + k.axis.z * half); f1.rotation.y = yaw;
    f2.position.set(k.T.x - k.axis.x * half, k.T.y + 0.025, k.T.z - k.axis.z * half); f2.rotation.y = yaw;
    ring.position.set(k.T.x, k.T.y + 0.065, k.T.z); ring.rotation.x = Math.PI / 2;
    const held = sim.holding >= 0, slip = held && sim.objects[sim.holding].slip;
    ring.material.color.set(slip ? '#ff4a3a' : held ? '#46ff8a' : arm.apCmd === 0 ? '#ffb02e' : '#38e8ff');
    f1.material.emissive.set(ring.material.color); f1.material.color.set(ring.material.color);
    goalMark.position.set(arm.goal.x, arm.goal.y, arm.goal.z); goalMark.rotation.y += 0.05;
    goalMark.visible = Math.hypot(arm.goal.x - k.T.x, arm.goal.y - k.T.y, arm.goal.z - k.T.z) > 0.004;
    plumbGeo.attributes.position.setXYZ(0, k.T.x, k.T.y, k.T.z); plumbGeo.attributes.position.setXYZ(1, k.T.x, 0.001, k.T.z); plumbGeo.attributes.position.needsUpdate = true; plumb.computeLineDistances();
    plumbDot.position.set(k.T.x, 0.0015, k.T.z);
    // the wrist sensor
    const s = sim.sensor;
    cone.visible = s.sees; seen.visible = s.sees;
    if (s.sees) { cone.position.set(k.T.x, k.T.y / 2, k.T.z); cone.scale.set(1, k.T.y, 1); cone.rotation.x = Math.PI; seen.position.set(k.T.x + s.dx, 0.002, k.T.z + s.dz); }
    // the objects, the ghosts, the bin stack
    binCount = 0;
    sim.objects.forEach((o, i) => {
      const m = objMeshes[i], g = ghosts[i];
      m.visible = o.state !== 'lost';
      if (o.state === 'bin') { m.position.set(BIN.x + (i % 2 ? 0.02 : -0.02), -0.1 + o.h / 2 + 0.005 + 0.0 * binCount, BIN.z + (i > 1 ? 0.02 : -0.02)); binCount++; }
      else m.position.set(o.x, o.y + o.h / 2, o.z);
      m.material.emissive.set(ui.target === i && o.state === 'table' ? o.color : '#000000'); m.material.emissiveIntensity = 0.25;
      g.visible = o.state === 'table';
      if (g.visible) {
        const useMem = ui.target === i && ui.camOk;
        g.position.set(useMem ? ui.cam.x : o.camX, o.h / 2, useMem ? ui.cam.z : o.camZ);
        g.material.color.set(ui.target === i ? '#ff3ea8' : '#9a3a7a'); g.children[0].material.opacity = ui.target === i ? 0.2 : 0.08;
      }
    });
  };

  const resize = () => {
    const w = canvas.clientWidth, h = canvas.clientHeight;
    if (canvas.width !== Math.floor(w * renderer.getPixelRatio()) || canvas.height !== Math.floor(h * renderer.getPixelRatio())) renderer.setSize(w, h, false);
    camera.aspect = w / h; camera.updateProjectionMatrix();
  };
  return { renderer, scene, camera, draw, build, resize, render: () => { resize(); renderer.render(scene, camera); } };
}
