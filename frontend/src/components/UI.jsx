import React from 'react'
import { motion } from 'framer-motion'
import './styles/UI.css'

export function Card({ children }) {
  return <div className="ui-card">{children}</div>
}

export function Button({ children, ...props }) {
  return <button className="ui-button" {...props}>{children}</button>
}

export function Page({ title, action, children }) {
  return (
    <motion.main
      initial={{ opacity: 0, y: 14, filter: 'blur(6px)' }}
      animate={{ opacity: 1, y: 0, filter: 'blur(0px)' }}
      exit={{ opacity: 0, y: -10, filter: 'blur(6px)' }}
      transition={{ duration: 0.35, ease: [0.22, 1, 0.36, 1] }}
      style={{ padding: '84px 24px 24px' }}
    >
      <div className="ui-page__header">
        {title && <h2 className="ui-page__title">{title}</h2>}
        {action}
      </div>
      <div className="ui-page__content">{children}</div>
    </motion.main>
  )
}
