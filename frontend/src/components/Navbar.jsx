import React from 'react'
import GooeyNav from './GooeyNav.jsx'

// précharge les chunks au hover
const preloads = {
  '/':      () => import('../pages/Dashboard.jsx'),
  '/run':   () => import('../pages/RunPipeline.jsx'),
  '/models':() => import('../pages/Models.jsx'),
}

export default function Navbar() {
  const items = [
    { label: 'Dashboard', href: '/' },
    { label: 'Run',       href: '/run' },
    { label: 'Models',    href: '/models' },
  ]
  return (
    <GooeyNav
      items={items.map(it => ({
        ...it,
        onHover: () => preloads[it.href]?.()
      }))}
    />
  )
}
