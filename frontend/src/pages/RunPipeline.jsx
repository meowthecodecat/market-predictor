import React, { useEffect, useRef, useState } from 'react'
import { postRun, getRunLogs, getRunStatus } from '../lib/api.js'
import LogsConsole from '../components/LogsConsole.jsx'
import { Link } from 'react-router-dom'
import { Button, Card, Page } from '../components/UI.jsx'
import './styles/RunPipeline.css'

export default function RunPipeline() {
  const [jobId, setJobId] = useState(null)
  const [running, setRunning] = useState(false)
  const [exitCode, setExitCode] = useState(null)
  const [lines, setLines] = useState([])
  const pollRef = useRef(null)

  const stopPolling = () => { if (pollRef.current) { clearInterval(pollRef.current); pollRef.current = null } }

  const poll = (jid) => {
    stopPolling()
    pollRef.current = setInterval(async () => {
      try {
        const [logs, status] = await Promise.all([ getRunLogs(jid, 300), getRunStatus(jid) ])
        setLines(logs?.lines || [])
        setRunning(Boolean(status?.running))
        setExitCode(status?.exit_code ?? null)
        if (!status?.running) stopPolling()
      } catch {}
    }, 1500)
  }

  const onRun = async () => {
    setLines([]); setExitCode(null); setRunning(true)
    const res = await postRun()
    setJobId(res.job_id)
    poll(res.job_id)
  }

  useEffect(() => () => stopPolling(), [])

  return (
    <Page
      title="Run pipeline"
      action={<Button onClick={onRun} disabled={running}>{running ? 'En cours…' : 'Run'}</Button>}
    >
      {jobId && <div className="run__job">job_id: <span className="run__mono">{jobId}</span></div>}
      <Card><LogsConsole lines={lines} /></Card>

      {!running && exitCode !== null && (
        <div className="run__footer">
          <div className={'run__status ' + (exitCode === 0 ? 'ok' : 'ko')}>
            Terminé. exit_code = {exitCode}
          </div>
          <Link to="/" className="ui-button">Voir Dashboard</Link>
        </div>
      )}
    </Page>
  )
}
