import React, { useEffect } from 'react'
import { Routes, Route, useLocation } from 'react-router-dom'
import { AnimatePresence, motion } from 'framer-motion'
import Navbar from './components/Navbar.jsx'
import DarkVeil from './components/DarkVeil.jsx'
import Dashboard from './pages/Dashboard.jsx'
import RunPipeline from './pages/RunPipeline.jsx'
import Models from './pages/Models.jsx'

/* Transition : immédiate, fluide, zéro hachure
   => apparition par spring naturel et sortie quasi instant */
const variants = {
  initial: {
    opacity: 0,
    scale: 0.992,
    filter: 'blur(8px)',
  },
  animate: {
    opacity: 1,
    scale: 1,
    filter: 'blur(0px)',
    transition: {
      type: 'spring',
      stiffness: 180,
      damping: 22,
      mass: 0.5,
    },
  },
  exit: {
    opacity: 0,
    scale: 1.008,
    filter: 'blur(10px)',
    transition: {
      duration: 0.15,
      ease: [0.4, 0, 0.2, 1],
    },
  },
}

export default function App() {
  const location = useLocation()

  useEffect(() => {
    import('./pages/Dashboard.jsx')
    import('./pages/RunPipeline.jsx')
    import('./pages/Models.jsx')
  }, [])

  return (
    <div className="app-shell">
      <DarkVeil />
      <Navbar />
      <main className="app-shell__main" style={{ paddingTop: '96px' }}>
        <div className="app-shell__container">
          <AnimatePresence mode="wait" initial={false}>
            <motion.div
              key={location.pathname}
              variants={variants}
              initial="initial"
              animate="animate"
              exit="exit"
              style={{
                transformOrigin: 'center',
                willChange: 'opacity, transform, filter',
              }}
            >
              <Routes location={location}>
                <Route path="/" element={<Dashboard />} />
                <Route path="/run" element={<RunPipeline />} />
                <Route path="/models" element={<Models />} />
              </Routes>
            </motion.div>
          </AnimatePresence>
        </div>
      </main>
    </div>
  )
}
