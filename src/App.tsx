import { useState, useEffect } from 'react'
import { WatchPage } from './ui/watch/WatchPage'
import { InstinctBattleground } from './ui/instincts/InstinctBattleground'

type Tab = 'watch' | 'instincts'

function App() {
  // Initialize tab from URL path
  const getInitialTab = (): Tab => {
    const path = window.location.pathname.replace('/gogogo/', '').replace('/gogogo', '').replace('/', '')
    if (path === 'instincts') {
      return 'instincts'
    }
    return 'watch'
  }

  const [tab, setTab] = useState<Tab>(getInitialTab())

  // Sync URL with tab state
  useEffect(() => {
    const path = tab === 'watch' ? '/' : `/${tab}`
    const fullPath = (import.meta.env?.BASE_URL || '/') + path.slice(1)
    if (window.location.pathname !== fullPath) {
      window.history.pushState(null, '', fullPath + window.location.search)
    }
  }, [tab])

  return (
    <div className="app">
      <header className="app-header">
        <h1>GoGoGo AI Lab</h1>
        <nav className="nav">
          <button
            onClick={() => setTab('watch')}
            className={tab === 'watch' ? 'active' : ''}
          >
            Watch AI
          </button>
          <button
            onClick={() => setTab('instincts')}
            className={tab === 'instincts' ? 'active' : ''}
          >
            Instincts
          </button>
        </nav>
      </header>

      {tab === 'instincts' ? (
        <InstinctBattleground />
      ) : (
        <WatchPage />
      )}
    </div>
  )
}

export default App
