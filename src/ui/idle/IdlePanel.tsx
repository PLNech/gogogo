import { useEffect, useState } from 'react'
import { useIdleStore } from '../../state/idleStore'
import { useCurrencyStore } from '../../state/currencyStore'
import { useStatsStore } from '../../state/statsStore'
import { IDLE_SOURCES, getAvailableIdleSources } from '../../domain/idle/sources'
import type { IdleSourceId } from '../../domain/idle/types'
import './IdlePanel.css'

export function IdlePanel() {
  const sources = useIdleStore(state => state.sources)
  const purchaseSource = useIdleStore(state => state.purchaseSource)
  const calculatePending = useIdleStore(state => state.calculatePending)
  const collectIdle = useIdleStore(state => state.collectIdle)

  const stones = useCurrencyStore(state => state.stones)
  const canAfford = useCurrencyStore(state => state.canAfford)
  const spendStones = useCurrencyStore(state => state.spendStones)
  const earnStones = useCurrencyStore(state => state.earnStones)

  const gamesPlayed = useStatsStore(state => state.gamesPlayed)
  const wins = useStatsStore(state => state.wins)

  const [pending, setPending] = useState({ pendingStones: 0, elapsedSeconds: 0, ratePerSecond: 0 })
  const [justCollected, setJustCollected] = useState(0)

  // Update pending every second
  useEffect(() => {
    const interval = setInterval(() => {
      setPending(calculatePending())
    }, 1000)

    return () => clearInterval(interval)
  }, [calculatePending])

  const handlePurchase = (id: IdleSourceId) => {
    const source = IDLE_SOURCES[id]
    if (!source) return

    if (purchaseSource(id, source.cost, spendStones)) {
      // Success!
    }
  }

  const handleCollect = () => {
    const collected = collectIdle(earnStones)
    if (collected > 0) {
      setJustCollected(collected)
      setTimeout(() => setJustCollected(0), 2000)
    }
  }

  const availableSources = getAvailableIdleSources(sources, {
    gamesPlayed,
    wins,
    totalEarned: 0 // totalEarned is not yet tracked in currency store
  })

  const purchasedSources = Array.from(sources).map(id => IDLE_SOURCES[id]!).filter(Boolean)

  return (
    <div className="idle-panel">
      <header className="idle-header">
        <h1 className="idle-title">The Path of Stillness</h1>
        <p className="idle-subtitle">
          "Stones gather in silence. Wisdom flows in patience."
        </p>
      </header>

      {pending.ratePerSecond > 0 && (
        <div className="idle-progress">
          <div className="pending-display">
            <div className="pending-amount">{pending.pendingStones} stones</div>
            <div className="pending-label">waiting to be collected</div>
            <div className="idle-rate">
              +{pending.ratePerSecond.toFixed(1)} stones/sec
            </div>
          </div>
          <button
            onClick={handleCollect}
            className="collect-button"
            disabled={pending.pendingStones === 0}
          >
            {pending.pendingStones > 0 ? `Collect ${pending.pendingStones} Stones` : 'Nothing to collect yet...'}
          </button>
          {justCollected > 0 && (
            <div className="collected-notification">
              +{justCollected} stones collected! 🪨
            </div>
          )}
        </div>
      )}

      {purchasedSources.length > 0 && (
        <section className="owned-sources">
          <h2>Your Sources</h2>
          <div className="source-list">
            {purchasedSources.map(source => (
              <div key={source.id} className="source-card owned">
                <div className="source-name">{source.name}</div>
                <div className="source-rate">+{source.rate} stones/sec</div>
                <div className="source-quote">"{source.quote}"</div>
              </div>
            ))}
          </div>
        </section>
      )}

      {availableSources.length > 0 ? (
        <section className="available-sources">
          <h2>Available Sources</h2>
          <div className="source-list">
            {availableSources.map(id => {
              const source = IDLE_SOURCES[id]!
              const affordable = canAfford(source.cost)

              return (
                <div key={id} className={`source-card ${affordable ? 'affordable' : 'expensive'}`}>
                  <div className="source-header">
                    <div className="source-name">{source.name}</div>
                    <div className="source-cost">{source.cost} stones</div>
                  </div>
                  <div className="source-description">{source.description}</div>
                  <div className="source-rate">+{source.rate} stones/sec</div>
                  <div className="source-quote">"{source.quote}"</div>
                  {source.unlockCondition && (
                    <div className="unlock-condition">
                      Requires: {source.unlockCondition.type === 'games_played' ? `${source.unlockCondition.value} games` :
                               source.unlockCondition.type === 'wins' ? `${source.unlockCondition.value} wins` :
                               `${source.unlockCondition.value} total stones earned`}
                    </div>
                  )}
                  <button
                    onClick={() => handlePurchase(id)}
                    disabled={!affordable}
                    className="purchase-button"
                  >
                    {affordable ? 'Purchase' : `Need ${source.cost - stones} more`}
                  </button>
                </div>
              )
            })}
          </div>
        </section>
      ) : sources.size === 0 ? (
        <div className="empty-state">
          <p>The path awaits. Play games and win to unlock idle sources.</p>
          <p className="empty-quote">"The first step is always the hardest. The rest flow like water."</p>
        </div>
      ) : (
        <div className="all-purchased">
          <p>You have mastered all known paths of stillness.</p>
          <p className="mastery-quote">"When all is unlocked, the real journey begins."</p>
        </div>
      )}
    </div>
  )
}
