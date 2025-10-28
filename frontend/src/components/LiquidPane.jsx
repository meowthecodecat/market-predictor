import React from 'react'
import './styles/LiquidPane.css'

export default function LiquidPane({ className = '', children, style }) {
  return (
    <section className={`lg-pane ${className}`} style={style}>
      <div style={{ padding: 18 }}>{children}</div>
    </section>
  )
}
