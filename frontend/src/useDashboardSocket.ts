import { useEffect, useRef, useState } from "react";
import { fromWire, type PoseFrame } from "./lib/skeleton";

export interface EventEntry {
  node_id: string;
  ts: number;
  event: string;
  fall_score: number;
}

export type ConnState = "connecting" | "open" | "closed";

const WS_URL =
  import.meta.env.VITE_WS_URL ?? "ws://localhost:8006/ws/dashboard";

/** Connect to the backend dashboard WS, auto-reconnecting, surfacing the latest
 *  pose frame and the rolling event log. The dashboard only ever receives
 *  skeletons + events — never video. */
export function useDashboardSocket() {
  const [conn, setConn] = useState<ConnState>("connecting");
  const [frame, setFrame] = useState<PoseFrame | null>(null);
  const [events, setEvents] = useState<EventEntry[]>([]);
  const wsRef = useRef<WebSocket | null>(null);

  useEffect(() => {
    let stopped = false;
    let retry: ReturnType<typeof setTimeout>;

    const connect = () => {
      setConn("connecting");
      const ws = new WebSocket(WS_URL);
      wsRef.current = ws;

      ws.onopen = () => setConn("open");
      ws.onclose = () => {
        setConn("closed");
        if (!stopped) retry = setTimeout(connect, 1000);
      };
      ws.onerror = () => ws.close();
      ws.onmessage = (e) => {
        const msg = JSON.parse(e.data);
        if (msg.type === "event_log") {
          setEvents(msg.events ?? []);
          return;
        }
        const f = fromWire(msg);
        setFrame(f);
        if (f.event && f.event !== "none") {
          setEvents((prev) => [
            { node_id: f.node_id, ts: f.ts, event: f.event, fall_score: f.fall_score },
            ...prev,
          ].slice(0, 100));
        }
      };
    };

    connect();
    return () => {
      stopped = true;
      clearTimeout(retry);
      wsRef.current?.close();
    };
  }, []);

  return { conn, frame, events };
}
