import React, { useMemo, useState } from 'react'
import { BarChart, Bar, XAxis, YAxis, ResponsiveContainer } from 'recharts'
import './styles/SummaryTable.css'

export default function SummaryTable({ rows }) {
  const [query, setQuery] = useState('')
  const [sortKey, setSortKey] = useState('ts_utc')
  const [sortDir, setSortDir] = useState('desc')

  const filtered = useMemo(() => {
    const q = query.trim().toUpperCase()
    let out = rows || []
    if (q) out = out.filter(r => String(r.symbol || '').toUpperCase().includes(q))
    out = [...out].sort((a,b) => {
      const va = a[sortKey], vb = b[sortKey]
      if (va < vb) return sortDir === 'asc' ? -1 : 1
      if (va > vb) return sortDir === 'asc' ? 1 : -1
      return 0
    })
    return out
  }, [rows, query, sortKey, sortDir])

  const onSort = (key) => {
    if (key === sortKey) setSortDir(sortDir === 'asc' ? 'desc' : 'asc')
    else { setSortKey(key); setSortDir('desc') }
  }

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
              <th onClick={() => onSort('ts_utc')}>ts_utc</th>
              <th onClick={() => onSort('symbol')}>symbol</th>
              <th onClick={() => onSort('last_close')} className="num">last_close</th>
              <th onClick={() => onSort('pred_close')} className="num">pred_close</th>
              <th onClick={() => onSort('d_pct')} className="num">d_pct</th>
              <th>spark</th>
              <th>status</th>
            </tr>
          </thead>
          <tbody>
            {filtered.map((r, idx) => (
              <tr key={`${r.ts_utc}-${r.symbol}-${idx}`}>
                <td>{r.ts_utc}</td>
                <td>{r.symbol}</td>
                <td className="num">{Number.isFinite(r.last_close) ? r.last_close.toFixed(2) : '-'}</td>
                <td className="num">{Number.isFinite(r.pred_close) ? r.pred_close.toFixed(2) : '-'}</td>
                <td className={`num ${r.d_pct >= 0 ? 'pos' : 'neg'}`}>
                  {Number.isFinite(r.d_pct) ? r.d_pct.toFixed(2) : '-'}
                </td>
                <td>
                  <div className="summary-table__spark">
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart data={[{x:'d', y:r.d_pct}]}>
                        <XAxis dataKey="x" hide />
                        <YAxis hide />
                        <Bar dataKey="y" />
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                </td>
                <td><span className="summary-table__badge">{r.status}</span></td>
              </tr>
            ))}
            {filtered.length === 0 && (
              <tr>
                <td className="empty" colSpan={7}>Aucune donnée</td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  )
}
