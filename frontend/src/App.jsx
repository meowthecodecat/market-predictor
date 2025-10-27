import React from 'react'
import { Routes, Route } from 'react-router-dom'
import Navbar from './components/Navbar.jsx'
import Dashboard from './pages/Dashboard.jsx'
import RunPipeline from './pages/RunPipeline.jsx'
import Models from './pages/Models.jsx'
import './styles/components/AppShell.css'

export default function App() {
  return (
    <div className="app-shell">
      <Navbar />
      <main className="app-shell__main">
        <div className="app-shell__container">
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/run" element={<RunPipeline />} />
            <Route path="/models" element={<Models />} />
          </Routes>
        </div>
      </main>
    </div>
  )
}
