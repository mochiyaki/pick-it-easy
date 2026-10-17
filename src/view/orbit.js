// A small orbit camera: drag to turn, wheel to zoom, right-drag / shift-drag to pan, touch too. No addons needed.
import * as THREE from 'three';

export function createOrbit(camera, dom, { target = new THREE.Vector3(-0.02, 0.04, 0.24), yaw = 0.62, pitch = 0.5, dist = 1.05 } = {}) {
  const o = { target, yaw, pitch, dist, minDist: 0.3, maxDist: 3 };
  let drag = null, pinch = 0;
  const apply = () => {
    const cp = Math.cos(o.pitch);
    camera.position.set(o.target.x + Math.sin(o.yaw) * cp * o.dist, o.target.y + Math.sin(o.pitch) * o.dist, o.target.z + Math.cos(o.yaw) * cp * o.dist);
    camera.lookAt(o.target);
  };
  const turn = (dx, dy) => { o.yaw -= dx * 0.006; o.pitch = Math.max(0.05, Math.min(1.5, o.pitch + dy * 0.006)); };
  const pan = (dx, dy) => {
    const right = new THREE.Vector3().setFromMatrixColumn(camera.matrix, 0), up = new THREE.Vector3().setFromMatrixColumn(camera.matrix, 1);
    o.target.addScaledVector(right, -dx * 0.0012 * o.dist).addScaledVector(up, dy * 0.0012 * o.dist);
  };
  dom.addEventListener('pointerdown', (e) => { drag = { x: e.clientX, y: e.clientY, b: e.button, shift: e.shiftKey }; dom.setPointerCapture(e.pointerId); });
  dom.addEventListener('pointermove', (e) => {
    if (!drag || pinch) return;
    const dx = e.clientX - drag.x, dy = e.clientY - drag.y; drag.x = e.clientX; drag.y = e.clientY;
    if (drag.b === 2 || drag.shift) pan(dx, dy); else turn(dx, dy);
    apply();
  });
  dom.addEventListener('pointerup', () => { drag = null; });
  dom.addEventListener('contextmenu', (e) => e.preventDefault());
  dom.addEventListener('wheel', (e) => { e.preventDefault(); o.dist = Math.max(o.minDist, Math.min(o.maxDist, o.dist * Math.exp(e.deltaY * 0.001))); apply(); }, { passive: false });
  dom.addEventListener('touchstart', (e) => { if (e.touches.length === 2) pinch = Math.hypot(e.touches[0].clientX - e.touches[1].clientX, e.touches[0].clientY - e.touches[1].clientY); }, { passive: true });
  dom.addEventListener('touchmove', (e) => {
    if (e.touches.length !== 2 || !pinch) return;
    const d = Math.hypot(e.touches[0].clientX - e.touches[1].clientX, e.touches[0].clientY - e.touches[1].clientY);
    o.dist = Math.max(o.minDist, Math.min(o.maxDist, o.dist * pinch / d)); pinch = d; apply();
  }, { passive: true });
  dom.addEventListener('touchend', () => { pinch = 0; });
  o.apply = apply; o.set = (v) => { Object.assign(o, v); apply(); };
  apply();
  return o;
}
