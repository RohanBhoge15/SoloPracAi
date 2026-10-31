import { useCallback, useEffect, useRef } from "react";
import "./dna.css";

/* ── geometry ─────────────────────────────────────────────────── */

const TURNS = 2.6;          // helix revolutions over the full height
const PAIRS = 46;           // base pairs
const RADIUS = 96;          // helix radius, px at unit scale
const HEIGHT = 460;
const BEAD = 5.2;
const ROD_W = 1.7;

/* Base colours. Deliberately clinical rather than the usual
   rainbow - green/amber map onto current/superseded elsewhere. */
const BASES = [
  [34, 211, 154],  // green
  [77, 141, 255],  // blue
  [167, 139, 250], // violet
  [245, 165, 36],  // amber
];

/* ── tiny 3D helpers ──────────────────────────────────────────── */

type V3 = [number, number, number];

const rotY = (p: V3, a: number): V3 => [
  p[0] * Math.cos(a) - p[2] * Math.sin(a),
  p[1],
  p[0] * Math.sin(a) + p[2] * Math.cos(a),
];
const rotX = (p: V3, a: number): V3 => [
  p[0],
  p[1] * Math.cos(a) - p[2] * Math.sin(a),
  p[1] * Math.sin(a) + p[2] * Math.cos(a),
];
const project = (p: V3, w: number, h: number, depth: number) => {
  // perspective divide; `depth` is camera distance
  const scale = depth / (depth - p[2]);
  return {
    x: w / 2 + p[0] * scale,
    y: h / 2 + p[1] * scale,
    z: p[2],
    s: scale,
  };
};

export default function DNAHelix() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const stageRef = useRef<HTMLDivElement>(null);

  // interaction state, kept in refs so the RAF loop never re-renders
  const spin = useRef(0);
  const vel = useRef(0.0016);
  const tilt = useRef(0.42);
  const dragging = useRef(false);
  const lastX = useRef(0);
  const lastY = useRef(0);
  const hovered = useRef(false);
  const hueShift = useRef(0);
  const impulse = useRef(0);
  const raf = useRef(0);

  const emitRipple = useCallback((x: number, y: number) => {
    const stage = stageRef.current;
    if (!stage) return;
    const el = document.createElement("div");
    el.className = "dna-ripple";
    el.style.left = `${x}px`;
    el.style.top = `${y}px`;
    stage.appendChild(el);
    el.animate(
      [
        { transform: "translate(-50%,-50%) scale(1)", opacity: 0.9 },
        { transform: "translate(-50%,-50%) scale(26)", opacity: 0 },
      ],
      { duration: 760, easing: "cubic-bezier(0.22,1,0.36,1)" },
    ).onfinish = () => el.remove();
  }, []);

  useEffect(() => {
    const canvas = canvasRef.current;
    const stage = stageRef.current;
    if (!canvas || !stage) return;
    const ctx = canvas.getContext("2d", { alpha: true });
    if (!ctx) return;

    let width = 0;
    let height = 0;
    let dpr = 1;

    const resize = () => {
      const rect = stage.getBoundingClientRect();
      // Cap DPR at 2. Beyond that the cost is real and the gain is not.
      dpr = Math.min(window.devicePixelRatio || 1, 2);
      width = rect.width;
      height = rect.height;
      canvas.width = Math.round(width * dpr);
      canvas.height = Math.round(height * dpr);
      canvas.style.width = `${width}px`;
      canvas.style.height = `${height}px`;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    };

    resize();
    const ro = new ResizeObserver(resize);
    ro.observe(stage);

    // Precompute base-pair endpoints once, not every frame.
    const points: Array<{ a: V3; b: V3; t: number; i: number }> = [];
    for (let i = 0; i < PAIRS; i++) {
      const t = i / (PAIRS - 1);
      const y = t * HEIGHT - HEIGHT / 2;
      const phase = t * TURNS * Math.PI * 2;
      points.push({
        a: [Math.cos(phase) * RADIUS, y, Math.sin(phase) * RADIUS],
        b: [-Math.cos(phase) * RADIUS, y, -Math.sin(phase) * RADIUS],
        t,
        i,
      });
    }

    const CAMERA = 1250;

    const draw = () => {
      ctx.clearRect(0, 0, width, height);

      // spring the spin back to a resting velocity
      const target = hovered.current ? 0.0006 : 0.0016;
      vel.current += (target - vel.current) * 0.04;
      spin.current += vel.current + impulse.current;
      impulse.current *= 0.94;
      hueShift.current += 0.0035;

      const ax = spin.current;
      const ay = tilt.current;
      const project3 = (p: V3) => project(p, width, height, CAMERA);

      const projected = points.map((pt) => {
        const a = rotY(rotX(pt.a, ay), ax);
        const b = rotY(rotX(pt.b, ay), ax);
        return { ...pt, pa: project3(a), pb: project3(b) };
      });

      // depth sort: furthest first, so near strands overlap correctly
      projected.sort((m, n) => m.pa.z - n.pa.z);

      // ── rungs (base-pair rods) ──────────────────────────────
      ctx.lineCap = "round";
      for (const p of projected) {
        const depth = 1 - (p.pa.z + RADIUS) / (2 * RADIUS);
        const alpha = 0.1 + depth * 0.4;
        ctx.strokeStyle = `rgba(120,160,220,${alpha.toFixed(3)})`;
        ctx.lineWidth = ROD_W * p.pa.s * 0.9;
        ctx.beginPath();
        ctx.moveTo(p.pa.x, p.pa.y);
        ctx.lineTo(p.pb.x, p.pb.y);
        ctx.stroke();
      }

      // ── two sugar-phosphate backbones ───────────────────────
      for (const strand of [0, 1] as const) {
        ctx.beginPath();
        projected.forEach((p, idx) => {
          const pt = strand === 0 ? p.pa : p.pb;
          if (idx === 0) ctx.moveTo(pt.x, pt.y);
          else ctx.lineTo(pt.x, pt.y);
        });
        ctx.strokeStyle = "rgba(90,150,255,0.55)";
        ctx.lineWidth = 2.4;
        ctx.stroke();
        ctx.strokeStyle = "rgba(160,210,255,0.28)";
        ctx.lineWidth = 5.5;
        ctx.stroke();
      }

      // ── nucleotide beads ────────────────────────────────────
      for (const p of projected) {
        for (const [pt, isA] of [
          [p.pa, true],
          [p.pb, false],
        ] as const) {
          const depth = 1 - (pt.z + RADIUS) / (2 * RADIUS);
          const base = BASES[(p.i + (isA ? 0 : 2)) % BASES.length];
          const [r, g, b] = base;
          const size = BEAD * pt.s * (0.75 + depth * 0.55);

          // glow
          ctx.beginPath();
          ctx.arc(pt.x, pt.y, size * 2.5, 0, Math.PI * 2);
          ctx.fillStyle = `rgba(${r},${g},${b},${(0.05 + depth * 0.09).toFixed(3)})`;
          ctx.fill();

          // body
          ctx.beginPath();
          ctx.arc(pt.x, pt.y, size, 0, Math.PI * 2);
          ctx.fillStyle = `rgba(${r},${g},${b},${(0.45 + depth * 0.55).toFixed(3)})`;
          ctx.fill();

          // specular
          ctx.beginPath();
          ctx.arc(pt.x - size * 0.3, pt.y - size * 0.3, size * 0.34, 0, Math.PI * 2);
          ctx.fillStyle = `rgba(255,255,255,${(0.2 + depth * 0.4).toFixed(3)})`;
          ctx.fill();
        }
      }

      // hue shift on click, as a thin overlay wash
      if (hueShift.current > 0.02) {
        ctx.globalCompositeOperation = "lighter";
      }

      raf.current = requestAnimationFrame(draw);
    };

    raf.current = requestAnimationFrame(draw);

    // ── pointer interaction ──────────────────────────────────
    const onDown = (e: PointerEvent) => {
      dragging.current = true;
      lastX.current = e.clientX;
      lastY.current = e.clientY;
      const rect = stage.getBoundingClientRect();
      impulse.current = 0.06;
      emitRipple(e.clientX - rect.left, e.clientY - rect.top);
      stage.setPointerCapture(e.pointerId);
    };

    const onMove = (e: PointerEvent) => {
      if (!dragging.current) return;
      const dx = e.clientX - lastX.current;
      const dy = e.clientY - lastY.current;
      lastX.current = e.clientX;
      lastY.current = e.clientY;
      spin.current += dx * 0.006;
      tilt.current = Math.max(-1.1, Math.min(1.1, tilt.current + dy * 0.004));
    };

    const onUp = (e: PointerEvent) => {
      dragging.current = false;
      vel.current = 0.0016;
      try {
        stage.releasePointerCapture(e.pointerId);
      } catch {
        /* pointer already released */
      }
    };

    const onEnter = () => (hovered.current = true);
    const onLeave = () => (hovered.current = false);

    stage.addEventListener("pointerdown", onDown);
    stage.addEventListener("pointermove", onMove);
    stage.addEventListener("pointerup", onUp);
    stage.addEventListener("pointercancel", onUp);
    stage.addEventListener("pointerenter", onEnter);
    stage.addEventListener("pointerleave", onLeave);

    return () => {
      cancelAnimationFrame(raf.current);
      ro.disconnect();
      stage.removeEventListener("pointerdown", onDown);
      stage.removeEventListener("pointermove", onMove);
      stage.removeEventListener("pointerup", onUp);
      stage.removeEventListener("pointercancel", onUp);
      stage.removeEventListener("pointerenter", onEnter);
      stage.removeEventListener("pointerleave", onLeave);
    };
  }, [emitRipple]);

  return (
    <div className="dna-stage" ref={stageRef} aria-hidden="true">
      <canvas className="dna-canvas" ref={canvasRef} />
      <div className="dna-hint">drag to rotate · click to pulse</div>
    </div>
  );
}
