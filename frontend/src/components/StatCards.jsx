import React from 'react'
import './styles/StatCards.css'

export default function StatCards({ avgDpct = 0, countOk = 0, countFail = 0, total = 0, accuracy = 0 }) {
  return (
    <div className="stat-cards">
      <div className="stat-cards__item">
        <div className="stat-cards__label">Écart moyen</div>
        <div className="stat-cards__value">{avgDpct.toFixed(3)}%</div>
      </div>
      <div className="stat-cards__item">
        <div className="stat-cards__label">Hausse</div>
        <div className="stat-cards__value">{countOk}</div>
      </div>
      <div className="stat-cards__item">
        <div className="stat-cards__label">Baisse</div>
        <div className="stat-cards__value">{countFail}</div>
      </div>
      <div className="stat-cards__item">
        <div className="stat-cards__label">Précision</div>
        <div className="stat-cards__value">{accuracy.toFixed(1)}%</div>
      </div>
    </div>
  )
}
