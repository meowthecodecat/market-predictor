import React, { useState } from 'react'
import { NavLink } from 'react-router-dom'
import './styles/Navbar.css'

export default function Navbar() {
  const [open, setOpen] = useState(false)

  const linkClass = ({ isActive }) =>
    'navbar__link' + (isActive ? ' navbar__link--active' : '')

  return (
    <nav className="navbar">
      <div className="navbar__inner">
        <div className="navbar__brand">
          <div className="navbar__logo">📈</div>
          <span className="navbar__title">Market Scanner</span>
        </div>

        <div className="navbar__links">
          <NavLink to="/" className={linkClass}>Dashboard</NavLink>
          <NavLink to="/run" className={linkClass}>Run pipeline</NavLink>
          <NavLink to="/models" className={linkClass}>Models</NavLink>
        </div>

        <button
          className="navbar__menuBtn"
          onClick={() => setOpen(v => !v)}
          aria-label="menu"
        >☰</button>
      </div>

      {open && (
        <div className="navbar__mobile">
          <NavLink to="/" className={linkClass} onClick={()=>setOpen(false)}>Dashboard</NavLink>
          <NavLink to="/run" className={linkClass} onClick={()=>setOpen(false)}>Run pipeline</NavLink>
          <NavLink to="/models" className={linkClass} onClick={()=>setOpen(false)}>Models</NavLink>
        </div>
      )}
    </nav>
  )
}
