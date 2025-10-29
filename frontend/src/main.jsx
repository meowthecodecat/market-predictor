import React from 'react'
import ReactDOM from 'react-dom/client'
import { BrowserRouter } from 'react-router-dom'
import App from './App.jsx'
import './index.css'
import './styles/global.css'
import DarkVeil from './components/DarkVeil.jsx'

ReactDOM.createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <BrowserRouter>
      <div className="app-root">
        <DarkVeil />
        <App />
      </div>
    </BrowserRouter>
  </React.StrictMode>
)
