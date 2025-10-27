import React from 'react'
import './styles/UI.css'

export function Card({ children }) {
  return <div className="ui-card">{children}</div>
}

export function Button({ children, ...props }) {
  return <button className="ui-button" {...props}>{children}</button>
}

export function Page({ title, action, children }) {
  return (
    <div className="ui-page">
      <div className="ui-page__header">
        <h2 className="ui-page__title">{title}</h2>
        {action}
      </div>
      <div className="ui-page__content">{children}</div>
    </div>
  )
}
