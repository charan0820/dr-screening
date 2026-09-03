import React, { useEffect, useRef, useState } from "react";

// ── Smooth cubic-bezier path through polyline points ──────────────────────
// Uses midpoint control points — gives genuinely rounded joints at every data point
function smoothPath(pts: string): string {
  const parsed = pts.trim().split(/\s+/).map(p => {
    const [x, y] = p.split(",").map(Number);
    return [x, y] as [number, number];
  });
  if (parsed.length === 0) return "";
  let d = `M${parsed[0][0]},${parsed[0][1]}`;
  for (let i = 1; i < parsed.length; i++) {
    const [x0, y0] = parsed[i - 1];
    const [x1, y1] = parsed[i];
    const mx = (x0 + x1) / 2;
    d += ` C${mx},${y0} ${mx},${y1} ${x1},${y1}`;
  }
  return d;
}

// Same path but closed back to baseline at y=100 (for filled areas)
function smoothArea(pts: string): string {
  const parsed = pts.trim().split(/\s+/).map(p => {
    const [x, y] = p.split(",").map(Number);
    return [x, y] as [number, number];
  });
  if (parsed.length === 0) return "";
  let d = `M${parsed[0][0]},100 L${parsed[0][0]},${parsed[0][1]}`;
  for (let i = 1; i < parsed.length; i++) {
    const [x0, y0] = parsed[i - 1];
    const [x1, y1] = parsed[i];
    const mx = (x0 + x1) / 2;
    d += ` C${mx},${y0} ${mx},${y1} ${x1},${y1}`;
  }
  const last = parsed[parsed.length - 1];
  d += ` L${last[0]},100 Z`;
  return d;
}

export default function Dashboard() {
  return (
    <div className="dark min-h-[100dvh] h-screen w-screen bg-background text-foreground overflow-hidden flex flex-col font-mono p-3 gap-2.5 box-border">
      {/* Top Bar */}
      <header className="h-10 shrink-0 flex items-center justify-between px-2">
        <div className="flex items-center gap-2 text-tech-primary font-black text-base tracking-tight">
          <span>◈</span> mission_control
        </div>
        <div className="flex items-center gap-2 text-xs font-bold">
          <div className="px-2.5 py-1 rounded-full border border-tech-orange/20 text-tech-orange bg-tech-orange/10">ENV: PROD</div>
          <div className="px-2.5 py-1 rounded-full border border-tech-primary/20 text-tech-primary bg-tech-primary/10">BUILD: #4821</div>
          <div className="px-2.5 py-1 rounded-full border border-tech-green/20 text-tech-green bg-tech-green/10 flex items-center gap-1.5">
            <div className="w-1.5 h-1.5 rounded-full bg-tech-green animate-pulse" />
            STATUS: NOMINAL
          </div>
        </div>
      </header>

      {/* Bento Grid */}
      <main className="flex-1 min-h-0 grid grid-cols-4 grid-rows-3 gap-2.5">
        {/* ROW 1 */}
        <Cell className="col-span-2" label="// system_metrics"><SystemMetricsChart /></Cell>
        <Cell label="## uptime"><UptimeStat /></Cell>
        <Cell label="// disk_usage"><DiskUsageChart /></Cell>

        {/* ROW 2 */}
        <Cell label="// error_rate"><ErrorRateChart /></Cell>
        <Cell className="col-span-2" label="## requests_per_sec"><RequestsChart /></Cell>
        <Cell label="// active_nodes"><ActiveNodesGrid /></Cell>

        {/* ROW 3 */}
        <Cell label="// anomaly_detection"><AnomalyChart /></Cell>
        <Cell label="// commit_activity"><CommitHeatmap /></Cell>
        <Cell className="col-span-2" label="## build_pipeline"><BuildPipeline /></Cell>
      </main>

      <style>{`
        @keyframes draw { to { stroke-dashoffset: 0; } }
        @keyframes expand { from { transform: scaleX(0); } to { transform: scaleX(1); } }
      `}</style>
    </div>
  );
}

// ── Cell shell ─────────────────────────────────────────────────────────────
function Cell({ children, className = "", label }: { children: React.ReactNode; className?: string; label?: string }) {
  return (
    <div className={`relative bg-card rounded-2xl border border-border p-4 flex flex-col overflow-hidden ${className}`}>
      {label && (
        <div className="absolute top-3 left-4 font-mono font-bold text-[10px] text-tech-muted z-10 tracking-wider">
          {label}
        </div>
      )}
      <div className="flex-1 min-h-0 mt-5 relative w-full">
        {children}
      </div>
    </div>
  );
}

// ── 1. System Metrics ──────────────────────────────────────────────────────
function SystemMetricsChart() {
  const pts1 = "0,80 10,72 20,58 30,65 40,38 50,44 60,28 70,34 80,18 90,28 100,8";
  const pts2 = "0,60 10,52 20,38 30,50 40,24 50,42 60,18 70,32 80,14 90,26 100,4";

  return (
    <div className="w-full h-full flex flex-col">
      <div className="flex gap-4 text-[10px] font-bold font-mono text-tech-muted mb-1 self-end">
        <span className="flex items-center gap-1.5"><span className="inline-block w-3 h-0.5 rounded bg-tech-primary" /> CPU</span>
        <span className="flex items-center gap-1.5"><span className="inline-block w-3 h-0.5 rounded bg-tech-purple" /> MEM</span>
      </div>
      <div className="flex-1 min-h-0 relative">
        <svg viewBox="0 0 100 100" preserveAspectRatio="none" className="w-full h-full">
          <defs>
            <linearGradient id="g1" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="var(--color-tech-primary)" stopOpacity="0.25" />
              <stop offset="100%" stopColor="var(--color-tech-primary)" stopOpacity="0" />
            </linearGradient>
            <linearGradient id="g2" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="var(--color-tech-purple)" stopOpacity="0.2" />
              <stop offset="100%" stopColor="var(--color-tech-purple)" stopOpacity="0" />
            </linearGradient>
          </defs>
          {[25, 50, 75].map(y => (
            <line key={y} x1="0" y1={y} x2="100" y2={y} stroke="var(--border)" strokeWidth="0.4" strokeDasharray="2,3" />
          ))}
          <path d={smoothArea(pts1)} fill="url(#g1)" />
          <path d={smoothArea(pts2)} fill="url(#g2)" />
          <path d={smoothPath(pts1)} fill="none" stroke="var(--color-tech-primary)" strokeWidth="1.5"
            strokeLinecap="round" strokeLinejoin="round" vectorEffect="non-scaling-stroke"
            style={{ strokeDasharray: 350, strokeDashoffset: 350, animation: "draw 1.4s ease-out forwards" }} />
          <path d={smoothPath(pts2)} fill="none" stroke="var(--color-tech-purple)" strokeWidth="1.5"
            strokeLinecap="round" strokeLinejoin="round" vectorEffect="non-scaling-stroke"
            style={{ strokeDasharray: 350, strokeDashoffset: 350, animation: "draw 1.4s ease-out 0.2s forwards" }} />
        </svg>
      </div>
    </div>
  );
}

// ── 2. Uptime — arc gauge ──────────────────────────────────────────────────
function UptimeStat() {
  // Arc gauge: 270° sweep, starts bottom-left, ends bottom-right
  const cx = 50, cy = 54, r = 36;
  const circ = 2 * Math.PI * r;
  const sweep = 0.75; // 270° of the circle used as gauge range
  const arcLen = circ * sweep;
  const filled = arcLen * 0.9997;
  const gap = circ - arcLen; // the 90° gap at the bottom

  // Incident timeline — 30 days, a few blips
  const incidents = [3, 11, 19]; // day indices with incident

  return (
    <div className="w-full h-full flex flex-col items-center justify-center relative">
      <svg viewBox="0 0 100 100" className="w-full h-full max-h-full">
        {/* Track arc */}
        <circle cx={cx} cy={cy} r={r} fill="none"
          stroke="var(--border)" strokeWidth="9"
          strokeDasharray={`${arcLen} ${circ}`}
          strokeDashoffset={-gap / 2}
          strokeLinecap="round"
          transform={`rotate(135 ${cx} ${cy})`} />

        {/* Filled arc — green */}
        <circle cx={cx} cy={cy} r={r} fill="none"
          stroke="var(--color-tech-green)" strokeWidth="9"
          strokeDasharray={`${filled} ${circ}`}
          strokeDashoffset={-gap / 2}
          strokeLinecap="round"
          transform={`rotate(135 ${cx} ${cy})`}
          style={{
            filter: "drop-shadow(0 0 4px rgba(63,185,80,0.5))",
            strokeDashoffset: arcLen - gap / 2,
            animation: "draw 1.2s ease-out forwards",
          }} />

        {/* Incident dots on the track */}
        {incidents.map(day => {
          const pct = day / 30;
          const angle = 135 + pct * 270;
          const rad = (angle * Math.PI) / 180;
          const dx = cx + r * Math.cos(rad);
          const dy = cy + r * Math.sin(rad);
          return (
            <circle key={day} cx={dx} cy={dy} r={2.5}
              fill="var(--color-tech-orange)"
              style={{ filter: "drop-shadow(0 0 3px rgba(247,129,102,0.8))" }} />
          );
        })}

        {/* Center: percentage */}
        <text x={cx} y={cy - 6} fill="var(--foreground)" fontSize="14"
          fontFamily="var(--font-mono)" fontWeight="900" textAnchor="middle" dominantBaseline="middle">
          99.97%
        </text>
        <text x={cx} y={cy + 8} fill="var(--color-tech-muted)" fontSize="5.5"
          fontFamily="var(--font-mono)" fontWeight="700" textAnchor="middle" dominantBaseline="middle"
          letterSpacing="0.8">UPTIME</text>

        {/* SLA label */}
        <text x={cx} y={cy + 20} fill="var(--color-tech-green)" fontSize="5"
          fontFamily="var(--font-mono)" fontWeight="700" textAnchor="middle" dominantBaseline="middle"
          letterSpacing="0.5">SLA 99.9% ✓</text>
      </svg>

      {/* Incident count */}
      <div className="absolute bottom-0 left-0 right-0 flex justify-between px-2 text-[9px] font-mono font-bold">
        <span className="text-tech-muted">30d</span>
        <span className="text-tech-orange">3 incidents</span>
      </div>
    </div>
  );
}

// ── 3. Disk Usage ──────────────────────────────────────────────────────────
function DiskUsageChart() {
  const r = 36, cx = 50, cy = 48;
  const circ = 2 * Math.PI * r;
  const used = 0.62, cached = 0.23;

  return (
    <div className="w-full h-full flex flex-col items-center justify-between pb-1">
      <div className="flex-1 min-h-0 w-full flex items-center justify-center">
        <svg viewBox="0 0 100 100" className="w-full h-full max-h-full">
          {/* Track */}
          <circle cx={cx} cy={cy} r={r} fill="none" stroke="var(--border)" strokeWidth="14" />
          {/* Used */}
          <circle cx={cx} cy={cy} r={r} fill="none" stroke="var(--color-tech-primary)" strokeWidth="14"
            strokeDasharray={circ} strokeDashoffset={circ * (1 - used)}
            strokeLinecap="butt" transform={`rotate(-90 ${cx} ${cy})`}
            style={{ strokeDashoffset: circ, animation: `draw 1s ease-out forwards` }} />
          {/* Cached */}
          <circle cx={cx} cy={cy} r={r} fill="none" stroke="var(--color-tech-purple)" strokeWidth="14"
            strokeDasharray={circ} strokeDashoffset={circ * (1 - cached)}
            strokeLinecap="butt" transform={`rotate(${-90 + used * 360} ${cx} ${cy})`}
            style={{ strokeDashoffset: circ, animation: `draw 1s ease-out 0.3s forwards` }} />
          <text x={cx} y={cy - 5} fill="var(--foreground)" fontSize="13" fontFamily="var(--font-mono)"
            fontWeight="900" textAnchor="middle" dominantBaseline="middle">2.4T</text>
          <text x={cx} y={cy + 9} fill="var(--color-tech-muted)" fontSize="6" fontFamily="var(--font-mono)"
            fontWeight="700" textAnchor="middle" dominantBaseline="middle" letterSpacing="1">TOTAL</text>
        </svg>
      </div>
      <div className="flex justify-around w-full text-[9px] font-mono font-bold pt-1">
        <span className="flex flex-col items-center gap-0.5"><span className="text-tech-primary">62%</span><span className="text-tech-muted">USED</span></span>
        <span className="flex flex-col items-center gap-0.5"><span className="text-tech-purple">23%</span><span className="text-tech-muted">CACHE</span></span>
        <span className="flex flex-col items-center gap-0.5"><span className="text-tech-muted">15%</span><span className="text-tech-muted">FREE</span></span>
      </div>
    </div>
  );
}

// ── 4. Error Rate ─ vertical LED spectrum ─────────────────────────────────
const ERROR_DATA = [
  { code: "200", val: 85, hex: "#3fb950" },
  { code: "301", val:  8, hex: "#58a6ff" },
  { code: "404", val:  4, hex: "#f78166" },
  { code: "429", val:  2, hex: "#bc8cff" },
  { code: "500", val:  1, hex: "#ff7b72" },
];

function ErrorRateChart() {
  const [live, setLive] = useState(false);
  useEffect(() => { const t = setTimeout(() => setLive(true), 80); return () => clearTimeout(t); }, []);

  return (
    <div className="w-full h-full flex gap-2 pt-1">
      {ERROR_DATA.map((d, i) => (
        <div key={d.code} className="flex-1 h-full flex flex-col items-center">
          {/* Percentage */}
          <div className="font-mono font-bold text-[10px] mb-1 tabular-nums"
            style={{ color: d.hex }}>{d.val}%</div>

          {/* Bar track */}
          <div className="flex-1 w-full relative rounded-sm overflow-hidden"
            style={{ background: "rgba(255,255,255,0.03)", border: "1px solid rgba(255,255,255,0.05)" }}>

            {/* Filled bar — grows from bottom */}
            <div className="absolute bottom-0 left-0 right-0 rounded-sm"
              style={{
                height: live ? `${d.val}%` : "0%",
                transition: `height 0.9s cubic-bezier(0.34,1.4,0.64,1) ${i * 0.07}s`,
                background: `linear-gradient(to top, ${d.hex}55 0%, ${d.hex}cc 60%, ${d.hex} 100%)`,
                boxShadow: `0 -3px 12px ${d.hex}99, 0 -1px 4px ${d.hex}`,
              }} />

            {/* LED segment overlay — horizontal lines create the segmented look */}
            <div className="absolute inset-0 pointer-events-none" style={{
              backgroundImage: `repeating-linear-gradient(
                to top,
                transparent 0px, transparent 5px,
                rgba(14,17,23,0.55) 5px, rgba(14,17,23,0.55) 7px
              )`,
            }} />

            {/* Top-of-bar bloom dot */}
            {live && d.val > 0 && (
              <div className="absolute left-0 right-0 h-[3px] rounded-full"
                style={{
                  bottom: `calc(${d.val}% - 2px)`,
                  background: d.hex,
                  boxShadow: `0 0 8px 3px ${d.hex}`,
                  transition: `bottom 0.9s cubic-bezier(0.34,1.4,0.64,1) ${i * 0.07}s`,
                }} />
            )}
          </div>

          {/* HTTP code label */}
          <div className="font-mono font-bold text-[9px] text-tech-muted mt-1.5 tracking-wide">
            {d.code}
          </div>
        </div>
      ))}
    </div>
  );
}

// ── 5. Requests Per Sec ────────────────────────────────────────────────────
function RequestsChart() {
  const p50 = "0,88 12,84 24,78 36,68 48,72 60,56 72,80 84,58 96,76 100,82";
  const p95 = "0,68 12,62 24,54 36,44 48,48 60,30 72,58 84,36 96,52 100,60";
  const p99 = "0,48 12,40 24,42 36,20 48,28 60,10 72,38 84,16 96,30 100,38";

  return (
    <div className="w-full h-full flex flex-col">
      <div className="flex gap-4 text-[10px] font-bold font-mono mb-1 self-end">
        <span className="text-tech-teal flex items-center gap-1.5"><span className="inline-block w-3 h-0.5 rounded bg-tech-teal"/>p50</span>
        <span className="text-tech-primary flex items-center gap-1.5"><span className="inline-block w-3 h-0.5 rounded bg-tech-primary"/>p95</span>
        <span className="text-tech-orange flex items-center gap-1.5"><span className="inline-block w-3 h-0.5 rounded bg-tech-orange"/>p99</span>
      </div>
      <div className="flex-1 min-h-0 relative">
        <svg viewBox="0 0 100 100" preserveAspectRatio="none" className="w-full h-full">
          <defs>
            <linearGradient id="rg1" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="var(--color-tech-teal)" stopOpacity="0.25"/>
              <stop offset="100%" stopColor="var(--color-tech-teal)" stopOpacity="0"/>
            </linearGradient>
            <linearGradient id="rg2" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="var(--color-tech-primary)" stopOpacity="0.2"/>
              <stop offset="100%" stopColor="var(--color-tech-primary)" stopOpacity="0"/>
            </linearGradient>
            <linearGradient id="rg3" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="var(--color-tech-orange)" stopOpacity="0.15"/>
              <stop offset="100%" stopColor="var(--color-tech-orange)" stopOpacity="0"/>
            </linearGradient>
          </defs>
          {[25, 50, 75].map(y => (
            <line key={y} x1="0" y1={y} x2="100" y2={y} stroke="var(--border)" strokeWidth="0.4" />
          ))}
          <path d={smoothArea(p99)} fill="url(#rg3)" />
          <path d={smoothArea(p95)} fill="url(#rg2)" />
          <path d={smoothArea(p50)} fill="url(#rg1)" />
          <path d={smoothPath(p99)} fill="none" stroke="var(--color-tech-orange)" strokeWidth="1.5"
            strokeLinecap="round" strokeLinejoin="round" vectorEffect="non-scaling-stroke"
            style={{ strokeDasharray: 350, strokeDashoffset: 350, animation: "draw 1.6s ease-out forwards" }} />
          <path d={smoothPath(p95)} fill="none" stroke="var(--color-tech-primary)" strokeWidth="1.5"
            strokeLinecap="round" strokeLinejoin="round" vectorEffect="non-scaling-stroke"
            style={{ strokeDasharray: 350, strokeDashoffset: 350, animation: "draw 1.6s ease-out 0.1s forwards" }} />
          <path d={smoothPath(p50)} fill="none" stroke="var(--color-tech-teal)" strokeWidth="1.5"
            strokeLinecap="round" strokeLinejoin="round" vectorEffect="non-scaling-stroke"
            style={{ strokeDasharray: 350, strokeDashoffset: 350, animation: "draw 1.6s ease-out 0.2s forwards" }} />
        </svg>
      </div>
      <div className="flex justify-between text-[9px] font-mono font-bold text-tech-muted mt-1.5 border-t border-border/50 pt-1.5">
        {["14:00","14:15","14:30","14:45","15:00"].map(t => <span key={t}>{t}</span>)}
      </div>
    </div>
  );
}

// ── 6. Active Nodes — network topology ────────────────────────────────────
function ActiveNodesGrid() {
  // Hub + 6 surrounding nodes
  const hub = { id: "hub", x: 50, y: 50, label: "LB", status: "ok" };
  const nodes = [
    { id: "42", x: 50, y: 14, label: "42", status: "ok" },
    { id: "43", x: 81, y: 30, label: "43", status: "ok" },
    { id: "44", x: 81, y: 68, label: "44", status: "sync" },
    { id: "45", x: 50, y: 84, label: "45", status: "ok" },
    { id: "46", x: 19, y: 68, label: "46", status: "err" },
    { id: "47", x: 19, y: 30, label: "47", status: "ok" },
  ];

  const statusColor: Record<string, string> = {
    ok:   "#3fb950",
    sync: "#58a6ff",
    err:  "#f78166",
  };

  // Edge connections: hub→each, plus lateral ring
  const edges = [
    ...nodes.map(n => ({ x1: hub.x, y1: hub.y, x2: n.x, y2: n.y, status: n.status })),
    { x1: nodes[0].x, y1: nodes[0].y, x2: nodes[1].x, y2: nodes[1].y, status: "ok" },
    { x1: nodes[1].x, y1: nodes[1].y, x2: nodes[2].x, y2: nodes[2].y, status: nodes[2].status },
    { x1: nodes[4].x, y1: nodes[4].y, x2: nodes[5].x, y2: nodes[5].y, status: "ok" },
  ];

  return (
    <div className="w-full h-full flex flex-col">
      <div className="flex-1 min-h-0">
        <svg viewBox="0 0 100 100" className="w-full h-full">
          {/* Edges */}
          {edges.map((e, i) => (
            <line key={i}
              x1={e.x1} y1={e.y1} x2={e.x2} y2={e.y2}
              stroke={e.status === "err" ? "rgba(247,129,102,0.3)" : e.status === "sync" ? "rgba(88,166,255,0.25)" : "rgba(255,255,255,0.08)"}
              strokeWidth="0.8" vectorEffect="non-scaling-stroke" />
          ))}

          {/* Hub */}
          <circle cx={hub.x} cy={hub.y} r={7} fill="var(--card)"
            stroke="rgba(255,255,255,0.15)" strokeWidth="1" vectorEffect="non-scaling-stroke" />
          <text x={hub.x} y={hub.y} fill="var(--foreground)" fontSize="4.5"
            fontFamily="var(--font-mono)" fontWeight="900" textAnchor="middle" dominantBaseline="middle">LB</text>

          {/* Nodes */}
          {nodes.map(n => (
            <g key={n.id}>
              {/* Outer glow ring for active states */}
              {n.status !== "ok" && (
                <circle cx={n.x} cy={n.y} r={6.5}
                  fill="none" stroke={statusColor[n.status]} strokeWidth="0.5"
                  strokeOpacity="0.4" vectorEffect="non-scaling-stroke" />
              )}
              <circle cx={n.x} cy={n.y} r={5} fill="var(--card)"
                stroke={statusColor[n.status]} strokeWidth="1.2"
                vectorEffect="non-scaling-stroke" />
              <text x={n.x} y={n.y} fill={statusColor[n.status]} fontSize="3.8"
                fontFamily="var(--font-mono)" fontWeight="900"
                textAnchor="middle" dominantBaseline="middle">{n.label}</text>
            </g>
          ))}
        </svg>
      </div>

      {/* Status summary strip */}
      <div className="flex justify-between text-[9px] font-mono font-bold pt-1 border-t border-border/40">
        <span className="text-tech-green">4 ok</span>
        <span className="text-tech-primary">1 sync</span>
        <span className="text-tech-orange">1 err</span>
      </div>
    </div>
  );
}

// ── 7. Anomaly Detection ───────────────────────────────────────────────────
function AnomalyChart() {
  // Wider viewBox so dots fill the rectangular cell naturally
  const normals = Array.from({ length: 50 }, (_, i) => ({
    x: 5 + (Math.sin(i * 123) * 0.5 + 0.5) * 140,
    y: 8 + (Math.cos(i * 321) * 0.5 + 0.5) * 84,
    r: Math.abs(Math.sin(i)) * 1.8 + 1,
  }));
  const anomalies = [
    { x: 128, y: 12, r: 3.5 },
    { x: 138, y: 72, r: 3   },
    { x: 14,  y:  9, r: 4   },
  ];
  return (
    <div className="w-full h-full relative">
      <svg viewBox="0 0 150 100" preserveAspectRatio="none" className="w-full h-full">
        <g stroke="var(--border)" strokeWidth="0.4" strokeDasharray="1,4" vectorEffect="non-scaling-stroke">
          {[25,50,75].map(v => <line key={`h${v}`} x1="0" y1={v} x2="150" y2={v} vectorEffect="non-scaling-stroke"/>)}
          {[30,60,90,120].map(v => <line key={`v${v}`} x1={v} y1="0" x2={v} y2="100" vectorEffect="non-scaling-stroke"/>)}
        </g>
        {normals.map((p, i) => (
          <circle key={i} cx={p.x} cy={p.y} r={p.r}
            fill="var(--color-tech-primary)" fillOpacity="0.5" vectorEffect="non-scaling-stroke" />
        ))}
        {anomalies.map((p, i) => (
          <g key={i}>
            <circle cx={p.x} cy={p.y} r={p.r + 3.5} fill="none"
              stroke="var(--color-tech-orange)" strokeWidth="0.8" opacity="0.35" vectorEffect="non-scaling-stroke" />
            <circle cx={p.x} cy={p.y} r={p.r} fill="var(--color-tech-orange)" vectorEffect="non-scaling-stroke" />
          </g>
        ))}
      </svg>
      <div className="absolute top-0 right-0 text-[9px] font-mono font-bold text-tech-orange flex items-center gap-1 bg-tech-orange/10 px-1.5 py-0.5 rounded border border-tech-orange/20">
        <span className="w-1.5 h-1.5 rounded-full bg-tech-orange inline-block" /> 3 anomalies
      </div>
    </div>
  );
}

// ── 8. Commit Heatmap ──────────────────────────────────────────────────────
function CommitHeatmap() {
  const COLS = 12, ROWS = 7;
  const cells = Array.from({ length: COLS * ROWS }, (_, i) => {
    const v = (Math.sin(i * 1.5) * Math.cos(i * 3.2) + 1) / 2;
    return v > 0.8 ? 4 : v > 0.6 ? 3 : v > 0.4 ? 2 : v > 0.2 ? 1 : 0;
  });
  const colors = ["transparent", "rgba(63,185,80,0.2)", "rgba(63,185,80,0.4)", "rgba(63,185,80,0.65)", "rgba(63,185,80,1)"];
  return (
    <div className="w-full h-full flex flex-col justify-between py-1">
      <div className="flex-1 min-h-0 grid gap-[3px]"
        style={{ gridTemplateColumns: `repeat(${COLS}, 1fr)`, gridTemplateRows: `repeat(${ROWS}, 1fr)` }}>
        {cells.map((v, i) => (
          <div key={i} className="rounded-[2px] transition-all hover:scale-110"
            style={{ backgroundColor: colors[v], border: v === 0 ? "1px solid var(--border)" : "none" }} />
        ))}
      </div>
      <div className="flex items-center justify-between text-[9px] font-mono font-bold text-tech-muted pt-2">
        <span>Less</span>
        <div className="flex gap-1">
          {colors.map((c, i) => (
            <div key={i} className="w-2.5 h-2.5 rounded-[2px]"
              style={{ backgroundColor: c, border: i === 0 ? "1px solid var(--border)" : "none" }} />
          ))}
        </div>
        <span>More</span>
      </div>
    </div>
  );
}

// ── 9. Build Pipeline — CI flow ────────────────────────────────────────────
function BuildPipeline() {
  const [live, setLive] = useState(false);
  useEffect(() => { const t = setTimeout(() => setLive(true), 120); return () => clearTimeout(t); }, []);

  const stages = [
    { name: "Lint",   dur: 10, hex: "#8b949e", status: "done"    },
    { name: "Test",   dur: 25, hex: "#bc8cff", status: "done"    },
    { name: "Build",  dur: 20, hex: "#58a6ff", status: "done"    },
    { name: "Scan",   dur: 15, hex: "#f78166", status: "done"    },
    { name: "Deploy", dur: 20, hex: "#39d353", status: "running" },
    { name: "Verify", dur: 10, hex: "#8b949e", status: "pending" },
  ];
  const total = stages.reduce((a, s) => a + s.dur, 0);

  return (
    <div className="w-full h-full flex flex-col justify-center gap-3 px-1">

      {/* Stage blocks — proportional width, connected */}
      <div className="flex items-stretch gap-0 w-full" style={{ height: 52 }}>
        {stages.map((s, i) => {
          const isDone    = s.status === "done";
          const isRunning = s.status === "running";
          const isPending = s.status === "pending";
          return (
            <React.Fragment key={s.name}>
              {/* Connector arrow */}
              {i > 0 && (
                <div className="flex items-center shrink-0" style={{ width: 14, zIndex: 1 }}>
                  <svg viewBox="0 0 14 10" style={{ width: 14, height: 10 }}>
                    <path d="M0,5 L10,5 M7,2 L11,5 L7,8" fill="none"
                      stroke={isDone || isRunning ? stages[i-1].hex : "rgba(255,255,255,0.12)"}
                      strokeWidth="1.2" strokeLinecap="round" strokeLinejoin="round" />
                  </svg>
                </div>
              )}

              {/* Stage block */}
              <div
                className="relative flex flex-col items-center justify-center rounded overflow-hidden shrink-0"
                style={{
                  flex: s.dur,
                  background: isPending
                    ? "rgba(255,255,255,0.03)"
                    : `${s.hex}18`,
                  border: `1px solid ${isPending ? "rgba(255,255,255,0.07)" : s.hex + "40"}`,
                  opacity: live ? 1 : 0,
                  transform: live ? "translateY(0)" : "translateY(6px)",
                  transition: `opacity 0.4s ease ${i * 0.08}s, transform 0.4s ease ${i * 0.08}s`,
                  boxShadow: isRunning ? `0 0 12px ${s.hex}44` : "none",
                }}
              >
                {/* Running shimmer */}
                {isRunning && (
                  <div className="absolute inset-0 opacity-20"
                    style={{
                      background: `linear-gradient(90deg, transparent 0%, ${s.hex} 50%, transparent 100%)`,
                      animation: "shimmer 1.6s linear infinite",
                      backgroundSize: "200% 100%",
                    }} />
                )}

                {/* Status icon */}
                <div className="text-[11px] leading-none mb-0.5" style={{ color: isPending ? "rgba(255,255,255,0.2)" : s.hex }}>
                  {isDone ? "✓" : isRunning ? "●" : "○"}
                </div>
                <div className="font-mono font-bold text-[8.5px] leading-none" style={{ color: isPending ? "rgba(255,255,255,0.25)" : s.hex }}>
                  {s.name}
                </div>
                <div className="font-mono text-[8px] mt-0.5 leading-none" style={{ color: isPending ? "rgba(255,255,255,0.15)" : `${s.hex}99` }}>
                  {s.dur}s
                </div>
              </div>
            </React.Fragment>
          );
        })}
      </div>

      {/* Timeline bar */}
      <div className="relative w-full h-1 bg-border/30 rounded-full overflow-hidden">
        {(() => {
          let acc = 0;
          return stages.map(s => {
            const left = (acc / total) * 100;
            acc += s.dur;
            const isDone    = s.status === "done";
            const isRunning = s.status === "running";
            return (
              <div key={s.name} className="absolute top-0 bottom-0 rounded-full"
                style={{
                  left: `${left}%`,
                  width: `${(s.dur / total) * 100}%`,
                  background: isDone || isRunning ? s.hex : "transparent",
                  opacity: isDone ? 0.8 : isRunning ? 1 : 0,
                  boxShadow: isRunning ? `0 0 6px ${s.hex}` : "none",
                }} />
            );
          });
        })()}
      </div>

      {/* Footer */}
      <div className="flex justify-between text-[9px] font-mono font-bold">
        <span className="text-tech-muted">build #4821</span>
        <span className="text-tech-green">4 passed</span>
        <span className="text-tech-teal animate-pulse">deploying…</span>
        <span className="text-tech-muted">{total}s total</span>
      </div>

      <style>{`
        @keyframes shimmer {
          0%   { background-position: 200% 0; }
          100% { background-position: -200% 0; }
        }
      `}</style>
    </div>
  );
}

