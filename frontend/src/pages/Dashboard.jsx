import React, { useEffect, useState } from 'react'
import { getSummary } from '../lib/api.js'
import StatCards from '../components/StatCards.jsx'
import SummaryTable from '../components/SummaryTable.jsx'
import { Button, Card, Page } from '../components/UI.jsx'
import './styles/Dashboard.css'

export default function Dashboard() {
  const [rows, setRows] = useState([])
  const [loading, setLoading] = useState(false)

  const load = async () => {
    setLoading(true)
    try {
      const data = await getSummary()
      setRows(data || [])
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => { load() }, [])

  const avgDpct = rows.length ? rows.reduce((a, r) => a + (Number(r.d_pct) || 0), 0) / rows.length : 0
  const countOk = rows.filter(r => String(r.status).toUpperCase() === 'OK').length
  const countFail = rows.filter(r => String(r.status).toUpperCase() === 'FAIL').length

  return (
    <Page
      title="Dashboard"
      action={<Button onClick={load} disabled={loading}>{loading ? 'Chargement…' : 'Rafraîchir'}</Button>}
    >
      <Card className="dash__card">
        <StatCards avgDpct={avgDpct} countOk={countOk} countFail={countFail} total={rows.length} />
      </Card>
      <Card className="dash__card">
        <SummaryTable rows={rows} />
      </Card>
    </Page>
  )
}
