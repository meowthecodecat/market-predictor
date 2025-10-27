import React from 'react'
import './styles/StatCards.css'

export default function StatCards({ avgDpct = 0, countOk = 0, countFail = 0, total = 0 }) {
  return (
    <div className="stat-cards">
      <div className="stat-cards__item">
        <div className="stat-cards__label">Moyenne d_pct</div>
        <div className="stat-cards__value">{avgDpct.toFixed(3)}%</div>
      </div>
      <div className="stat-cards__item">
        <div className="stat-cards__label">OK</div>
        <div className="stat-cards__value">{countOk}</div>
      </div>
      <div className="stat-cards__item">
        <div className="stat-cards__label">FAIL</div>
        <div className="stat-cards__value">{countFail}</div>
      </div>
      <div className="stat-cards__item">
        <div className="stat-cards__label">Total lignes</div>
        <div className="stat-cards__value">{total}</div>
      </div>
    </div>
  )
}
