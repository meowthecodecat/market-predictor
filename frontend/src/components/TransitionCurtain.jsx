import React, { useEffect, useState } from 'react'
import { useLocation } from 'react-router-dom'
import { motion, AnimatePresence } from 'framer-motion'
import './styles/TransitionCurtain.css'

const ENTER = 0.28
const EXIT  = 0.32
const TOTAL = (ENTER + EXIT) * 1000

export default function TransitionCurtain() {
  const { pathname } = useLocation()
  const [active, setActive] = useState(false)
  const [key, setKey] = useState(0)

  useEffect(() => {
    setActive(true)
    setKey(k => k + 1)
    const t = setTimeout(() => setActive(false), TOTAL)
    return () => clearTimeout(t)
  }, [pathname])

  if (!active) return null

  return (
    <AnimatePresence mode="wait" initial={false}>
      <motion.div
        key={key}
        className="lc-curtain"
        initial={{ x: '100%' }}
        animate={{ x: '0%', transition: { duration: ENTER, ease: [0.22,1,0.36,1] } }}
        exit={{ x: '-100%', transition: { duration: EXIT, ease: [0.22,1,0.36,1] } }}
      >
        <div className="lc-sheen" />
      </motion.div>
    </AnimatePresence>
  )
}
