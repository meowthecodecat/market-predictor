import React, { useRef, useEffect, useState } from 'react'
import { useNavigate, useLocation } from 'react-router-dom'
import './styles/GooeyNav.css'

export default function GooeyNav({
  items = [
    { label: 'Dashboard', href: '/' },
    { label: 'Run', href: '/run' },
    { label: 'Models', href: '/models' },
  ],
  initialActiveIndex = 0
}) {
  const containerRef = useRef(null)
  const navRef = useRef(null)
  const sliderRef = useRef(null)
  const [activeIndex, setActiveIndex] = useState(initialActiveIndex)
  const navigate = useNavigate()
  const location = useLocation()

  // positionne la pastille sous l'élément actif
  const moveSliderTo = (liEl) => {
    if (!containerRef.current || !sliderRef.current || !liEl) return
    const c = containerRef.current.getBoundingClientRect()
    const r = liEl.getBoundingClientRect()
    Object.assign(sliderRef.current.style, {
      left: `${r.x - c.x}px`,
      top: `${r.y - c.y}px`,
      width: `${r.width}px`,
      height: `${r.height}px`,
      opacity: 1,
    })
  }

  const handleClick = (e, index, href) => {
    e.preventDefault()
    if (!navRef.current) return
    const liEl = navRef.current.querySelectorAll('li')[index]
    setActiveIndex(index)
    moveSliderTo(liEl)
    navigate(href)
  }

  // sync sur changement de route et resize
  useEffect(() => {
    if (!navRef.current) return
    // déduire l'index actif depuis l'URL
    const idx = Math.max(0, items.findIndex(i => i.href === location.pathname))
    setActiveIndex(idx === -1 ? 0 : idx)
    const liEl = navRef.current.querySelectorAll('li')[idx === -1 ? 0 : idx]
    moveSliderTo(liEl)

    const ro = new ResizeObserver(() => {
      const li = navRef.current?.querySelectorAll('li')[idx === -1 ? 0 : idx]
      moveSliderTo(li)
    })
    ro.observe(containerRef.current)
    return () => ro.disconnect()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [location.pathname])

  return (
    <div className="gooey-nav" ref={containerRef}>
      <nav>
        <ul ref={navRef}>
          {items.map((item, index) => (
            <li key={item.href} className={activeIndex === index ? 'active' : ''}>
              <a
                href={item.href}
                onClick={(e)=>handleClick(e, index, item.href)}
              >
                {item.label}
              </a>
            </li>
          ))}
        </ul>
      </nav>
      {/* Pastille unique qui slide */}
      <span className="slider" ref={sliderRef} />
    </div>
  )
}
