import React, { useMemo, useState } from 'react'
import { BarChart, Bar, XAxis, YAxis, ResponsiveContainer } from 'recharts'
import './styles/SummaryTable.css'

function Arrow({ v }) {
  if (!Number.isFinite(v)) return <span>—</span>
  if (v > 0) return <span className="arrow up">▲</span>
  if (v < 0) return <span className="arrow down">▼</span>
  return <span className="arrow eq">•</span>
}

const logoSrc = (symbol) => `/logos/${String(symbol || '').toUpperCase()}.png`

export default function SummaryTable({ rows, compact = false }) {
  const [query, setQuery] = useState('')
  const [sortKey, setSortKey] = useState('symbol')
  const [sortDir, setSortDir] = useState('asc')

  const filtered = useMemo(() => {
    const q = query.trim().toUpperCase()
    let out = rows || []
    if (q) out = out.filter(r => String(r.symbol || '').toUpperCase().includes(q))
    out = [...out].sort((a,b) => {
      const va = a[sortKey], vb = b[sortKey]
      if (va < vb) return sortDir === 'asc' ? -1 : 1
      if (va > vb) return sortDir === 'asc' ?  1 : -1
      return 0
    })
    return out
  }, [rows, query, sortKey, sortDir])

  const onSort = (k) => { setSortKey(k); setSortDir(sortDir === 'asc' ? 'desc' : 'asc') }

  return (
    <div className="summary-table">
      <div className="summary-table__controls">
        <input
          className="summary-table__filter"
          placeholder="Filtre ticker (ex: AAPL)"
          value={query}
          onChange={e => setQuery(e.target.value)}
        />
      </div>

      <div className="summary-table__container">
        <table className="summary-table__table">
          <thead>
            <tr>
              <th onClick={()=>onSort('symbol')}>Ticker</th>
              <th className="num" onClick={()=>onSort('last_close')}>Dernier</th>
              <th className="num" onClick={()=>onSort('pred_close')}>Prédiction</th>
              <th className="num" onClick={()=>onSort('d_pct')}>Écart %</th>
              {!compact && <th>Mini-graph</th>}
              <th>Statut</th>
            </tr>
          </thead>
          <tbody>
            {filtered.map((r, idx) => (
              <tr key={`${r.symbol}-${idx}`}>
                <td className="tick">
                  <img
                    className="tick__logo"
                    src={logoSrc(r.symbol)}
                    alt=""
                    loading="lazy"
                    onError={(e)=>{ e.currentTarget.style.display='none' }}
                  />
                  <span className="tick__text">{r.symbol}</span>
                </td>
                <td className="num">{Number.isFinite(+r.last_close) ? (+r.last_close).toFixed(2) : '-'}</td>
                <td className="num">{Number.isFinite(+r.pred_close) ? (+r.pred_close).toFixed(2) : '-'}</td>
                <td className={'num dpct ' + ((+r.d_pct||0)>0?'pos':(+r.d_pct||0)<0?'neg':'eq')}>
                  <Arrow v={+r.d_pct} /> {Number.isFinite(+r.d_pct) ? (+r.d_pct).toFixed(2) : '-'}
                </td>
                {!compact && (
                  <td>
                    <div className="summary-table__spark">
                      <ResponsiveContainer width="100%" height="100%">
                        <BarChart data={[{x:'d', y:+r.d_pct}]}>
                          <XAxis dataKey="x" hide />
                          <YAxis hide />
                          <Bar dataKey="y" />
                        </BarChart>
                      </ResponsiveContainer>
                    </div>
                  </td>
                )}
                <td><span className={'badge ' + (String(r.status).toUpperCase()==='OK'?'ok':'ko')}>{r.status}</span></td>
              </tr>
            ))}
            {filtered.length === 0 && (
              <tr><td className="empty" colSpan={compact?5:6}>Aucune donnée</td></tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  )
}
