import React, { useEffect, useRef } from 'react'
import './styles/LogsConsole.css'

export default function LogsConsole({ lines }) {
  const ref = useRef(null)
  useEffect(() => { if (ref.current) ref.current.scrollTop = ref.current.scrollHeight }, [lines])
  return <pre ref={ref} className="logs-console">{(lines || []).join('\n')}</pre>
}
