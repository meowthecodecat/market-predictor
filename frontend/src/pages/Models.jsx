import React, { useEffect, useState } from 'react'
import { getModels } from '../lib/api.js'
import { Page, Button } from '../components/UI.jsx'
import './styles/Models.css'
import LiquidPane from '../components/LiquidPane.jsx'

export default function Models() {
  const [rows, setRows] = useState([])
  const [loading, setLoading] = useState(false)

  const load = async () => {
    setLoading(true)
    try { setRows((await getModels()) || []) }
    finally { setLoading(false) }
  }

  useEffect(() => { load() }, [])

  return (
    <Page
      title="Models"
      action={<Button onClick={load} disabled={loading}>{loading ? 'Chargement…' : 'Rafraîchir'}</Button>}
    >
      <div className="models__grid">
        {rows.map((r) => (
          <LiquidPane key={r.symbol} className="models__pane">
            <div className="models__symbol">{r.symbol}</div>
            <div className="models__flags">
              <span className={'models__badge lg-chip ' + (r.lstm ? 'ok' : 'na')}>LSTM {r.lstm ? 'OK' : '—'}</span>
              <span className={'models__badge lg-chip ' + (r.scaler ? 'ok' : 'na')}>Scaler {r.scaler ? 'OK' : '—'}</span>
            </div>
            <div className="models__ts">updated_at: {r.updated_at || 'NA'}</div>
          </LiquidPane>
        ))}
        {rows.length === 0 && <div className="models__empty">Aucun artefact détecté</div>}
      </div>
    </Page>
  )
}
