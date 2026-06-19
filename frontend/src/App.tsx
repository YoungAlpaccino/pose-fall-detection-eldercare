// Caregiver dashboard.
//
// Receives skeleton telemetry + discrete events over the backend WebSocket and
// renders a live 2D skeleton overlay, a fall-score gauge, and an event timeline
// with ack/dismiss. The dashboard only ever receives skeletons + events —
// never video (docs/PRIVACY.md), so it cannot "spy", only alert.

import { useState } from "react";
import { SkeletonCanvas } from "./SkeletonCanvas";
import { useDashboardSocket, type EventEntry } from "./useDashboardSocket";

const connLabel: Record<string, string> = {
  connecting: "Connecting…",
  open: "Live",
  closed: "Offline",
};

function fmtTime(ts: number) {
  return new Date(ts * 1000).toLocaleTimeString();
}

export default function App() {
  const { conn, frame, events } = useDashboardSocket();
  const [acked, setAcked] = useState<Set<number>>(new Set());

  const score = frame?.fall_score ?? 0;
  const alarming = score >= 0.6 || frame?.event === "fall";

  const ack = (ev: EventEntry) =>
    setAcked((prev) => new Set(prev).add(ev.ts));

  return (
    <main className="app">
      <header className="topbar">
        <div>
          <h1>Eldercare Fall Detection</h1>
          <p className="sub">
            Caregiver dashboard · skeleton telemetry only — no video leaves the node
          </p>
        </div>
        <span className={`status status-${conn}`}>
          <span className="dot" /> {connLabel[conn]}
        </span>
      </header>

      <section className="grid">
        <div className="panel">
          <div className="panel-head">
            <h2>Live skeleton {frame ? `· ${frame.node_id}` : ""}</h2>
            {alarming && <span className="badge badge-fall">FALL</span>}
          </div>
          <SkeletonCanvas frame={frame} alarming={alarming} />
          <div className="gauge">
            <div className="gauge-label">
              <span>Fall score</span>
              <span>{score.toFixed(2)}</span>
            </div>
            <div className="gauge-track">
              <div
                className={`gauge-fill${alarming ? " hot" : ""}`}
                style={{ width: `${Math.min(score, 1) * 100}%` }}
              />
              <div className="gauge-tau" title="alarm threshold τ=0.60" />
            </div>
          </div>
          {!frame && conn === "open" && (
            <p className="hint">Waiting for an edge node to publish telemetry…</p>
          )}
          {conn !== "open" && (
            <p className="hint">
              Backend not reachable. Start it with{" "}
              <code>uvicorn backend.app.main:app</code>.
            </p>
          )}
        </div>

        <div className="panel">
          <div className="panel-head">
            <h2>Event timeline</h2>
            <span className="count">{events.length}</span>
          </div>
          {events.length === 0 ? (
            <p className="hint">No events yet. A fall will appear here.</p>
          ) : (
            <ul className="timeline">
              {events.map((ev, i) => (
                <li key={`${ev.ts}-${i}`} className={acked.has(ev.ts) ? "acked" : ""}>
                  <div className="ev-main">
                    <span className={`ev-tag ev-${ev.event}`}>
                      {ev.event.toUpperCase()}
                    </span>
                    <span className="ev-node">{ev.node_id}</span>
                    <span className="ev-time">{fmtTime(ev.ts)}</span>
                  </div>
                  <div className="ev-meta">
                    score {ev.fall_score.toFixed(2)}
                    {acked.has(ev.ts) ? (
                      <span className="ev-acked">✓ acknowledged</span>
                    ) : (
                      <button className="ev-ack" onClick={() => ack(ev)}>
                        Acknowledge
                      </button>
                    )}
                  </div>
                </li>
              ))}
            </ul>
          )}
        </div>
      </section>
    </main>
  );
}
