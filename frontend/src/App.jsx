import React, { Suspense, lazy } from 'react'
import { Routes, Route } from 'react-router-dom'
import Navbar from './components/Navbar.jsx'
import DarkVeil from './components/DarkVeil.jsx'

const Dashboard  = lazy(() => import('./pages/Dashboard.jsx'))
const RunPipeline= lazy(() => import('./pages/RunPipeline.jsx'))
const Models     = lazy(() => import('./pages/Models.jsx'))

export default function App() {
  return (
    <div className="app-shell">
      <DarkVeil /> {/* fond animé */}
      <Navbar />
      <main className="app-shell__main" style={{ paddingTop: '96px' }}>
        <div className="app-shell__container">
          <Suspense fallback={null}>
            <Routes>
              <Route path="/" element={<Dashboard />} />
              <Route path="/run" element={<RunPipeline />} />
              <Route path="/models" element={<Models />} />
            </Routes>
          </Suspense>
        </div>
      </main>
    </div>
  )
}
