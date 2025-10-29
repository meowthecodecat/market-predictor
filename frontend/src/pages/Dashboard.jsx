// frontend/src/pages/Dashboard.jsx
import React, { useEffect, useMemo, useState } from "react";
import { getSummary } from "../lib/api.js";
import StatCards from "../components/StatCards.jsx";
import SummaryTable from "../components/SummaryTable.jsx";
import { Button, Page } from "../components/UI.jsx";
import "./styles/Dashboard.css";
import LiquidPane from "../components/LiquidPane.jsx";
import { useUiSounds } from "../hooks/UseUISounds.jsx";

function parseTs(s) { return new Date(s).getTime() || 0; }

export default function Dashboard() {
  const [rowsAll, setRowsAll] = useState([]);
  const [loading, setLoading] = useState(false);
  const { playStart, playDone } = useUiSounds();

  const load = async () => {
    setLoading(true);
    playStart();
    try {
      const data = await getSummary();
      setRowsAll(Array.isArray(data) ? data : []);
    } finally {
      setLoading(false);
      playDone();
    }
  };
  useEffect(() => { load(); }, []);

  const { latestTsStr, rows } = useMemo(() => {
    if (!rowsAll.length) return { latestTsStr: null, rows: [] };
    const maxTs = rowsAll.reduce((m, r) => Math.max(m, parseTs(r.ts_utc)), 0);
    const latest = rowsAll.filter(r => parseTs(r.ts_utc) === maxTs);
    latest.sort((a,b) => String(a.symbol).localeCompare(String(b.symbol)));
    return { latestTsStr: latest[0]?.ts_utc || null, rows: latest };
  }, [rowsAll]);

  const avgDpct = rows.length ? rows.reduce((a, r) => a + (+r.d_pct || 0), 0) / rows.length : 0;
  const gainers = rows.filter(r => (+r.d_pct || 0) > 0).length;
  const losers  = rows.filter(r => (+r.d_pct || 0) < 0).length;
  const ok = rows.filter(r => String(r.status).toUpperCase() === "OK").length;
  const fail = rows.filter(r => String(r.status).toUpperCase() === "FAIL").length;
  const accuracy = rows.length ? (ok / (ok + fail || 1)) * 100 : 0;

  const avgConfidence = useMemo(() => {
    const vals = rows.map(r => Number(r.confidence)).filter(v => Number.isFinite(v));
    if (!vals.length) return null;
    return vals.reduce((a,b)=>a+b,0) / vals.length;
  }, [rows]);

  return (
    <Page
      title="Dashboard"
      action={<Button onClick={load} disabled={loading}>{loading ? "Chargement…" : "Rafraîchir"}</Button>}
    >
      <div className="dash__banner">
        <div className="dash__badge">Dernier run</div>
        <div className="dash__when">{latestTsStr ? latestTsStr : "—"}</div>
        <div className="dash__count">{rows.length} lignes</div>
      </div>

      <LiquidPane className="dash__card">
        <StatCards
          avgDpct={avgDpct}
          countOk={gainers}
          countFail={losers}
          total={rows.length}
          accuracy={accuracy}
        />
      </LiquidPane>

      <LiquidPane className="dash__card">
        <SummaryTable rows={rows} compact />
      </LiquidPane>
    </Page>
  );
}
