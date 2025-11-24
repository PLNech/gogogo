import { useState, useEffect } from 'react'
import { BoardView } from './ui/board/BoardView'
import { CurrencyDisplay } from './ui/game/CurrencyDisplay'
import { Milestone1 } from './ui/game/Milestone1'
import { Milestone2 } from './ui/game/Milestone2'
import { UpgradesPanel } from './ui/game/UpgradesPanel'
import { WatchPage } from './ui/watch/WatchPage'
import { IdlePanel } from './ui/idle/IdlePanel'
import { useGameStore } from './state/gameStore'
import { getCurrentMoveCount, getUnlockedBoardSizes } from './domain/upgrades/upgrades'
import { useUpgradeStore } from './state/upgradeStore'

type Tab = 'game' | 'upgrades' | 'idle' | 'watch' | 'debug'

function App() {
  // Initialize tab from URL path
  const getInitialTab = (): Tab => {
    const path = window.location.pathname.slice(1) // Remove leading '/'
    if (path === 'watch' || path === 'upgrades' || path === 'idle' || path === 'debug') {
      return path as Tab
    }
    return 'game'
  }

  const [tab, setTab] = useState<Tab>(getInitialTab())
  const phase = useGameStore((state) => state.phase)
  const purchased = useUpgradeStore((state) => state.purchased)

  // Sync URL with tab state
  useEffect(() => {
    const path = tab === 'game' ? '/' : `/${tab}`
    if (window.location.pathname !== path) {
      window.history.pushState(null, '', path + window.location.search)
    }
  }, [tab])

  const moveCount = getCurrentMoveCount(purchased)
  const boardSizes = getUnlockedBoardSizes(purchased)
  const maxBoardSize = Math.max(...boardSizes)

  return (
    <div className="app">
      <CurrencyDisplay />

      <nav className="nav">
        <button
          onClick={() => setTab('game')}
          className={tab === 'game' ? 'active' : ''}
        >
          Game
        </button>
        <button
          onClick={() => setTab('upgrades')}
          className={tab === 'upgrades' ? 'active' : ''}
        >
          Upgrades
        </button>
        <button
          onClick={() => setTab('idle')}
          className={tab === 'idle' ? 'active' : ''}
        >
          Idle
        </button>
        <button
          onClick={() => setTab('watch')}
          className={tab === 'watch' ? 'active' : ''}
        >
          Watch
        </button>
        <button
          onClick={() => setTab('debug')}
          className={tab === 'debug' ? 'active' : ''}
        >
          Debug
        </button>
      </nav>

      {tab === 'debug' ? (
        <div>
          <h1>Test Board</h1>
          <BoardView initialSize={5} showDebug={true} aiEnabled={true} />
        </div>
      ) : tab === 'watch' ? (
        <WatchPage />
      ) : tab === 'idle' ? (
        <IdlePanel />
      ) : tab === 'upgrades' ? (
        <UpgradesPanel />
      ) : (
        <div>
          {/* Milestone 1: Intro and Capture */}
          {(phase === 'M1_INTRO' || phase === 'M1_CAPTURE') && <Milestone1 />}

          {/* Milestone 2: Practice and AI Opponent */}
          {(phase.startsWith('M2_PRACTICE') || phase === 'M2_AI_OPPONENT') && <Milestone2 />}

          {/* Free Play (unlocked after M2) */}
          {phase === 'FREE_PLAY' && (
            <div>
              <h1>Free Play</h1>
              <p>Max moves: {moveCount} | Unlocked boards: {boardSizes.join(', ')}</p>
              <BoardView initialSize={maxBoardSize} showDebug={true} aiEnabled={true} />
            </div>
          )}
        </div>
      )}
    </div>
  )
}

export default App
