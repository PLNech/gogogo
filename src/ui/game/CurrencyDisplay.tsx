import { useState } from 'react'
import { useCurrencyStore } from '../../state/currencyStore'
import { UpgradesPanel } from './UpgradesPanel'
import './CurrencyDisplay.css'

export function CurrencyDisplay() {
  const stones = useCurrencyStore((state) => state.stones)
  const [showUpgrades, setShowUpgrades] = useState(false)

  return (
    <>
      <div className="currency-display">
        <div className="currency-item">
          <span className="currency-icon">⚫</span>
          <span className="currency-amount">{stones}</span>
          <span className="currency-label">Stones</span>
        </div>
        <button
          onClick={() => setShowUpgrades(true)}
          style={{
            marginLeft: '1rem',
            padding: '0.5rem 1rem',
            background: 'rgba(76, 175, 80, 0.2)',
            border: '2px solid rgba(76, 175, 80, 0.4)',
            borderRadius: '4px',
            color: '#4CAF50',
            cursor: 'pointer',
            fontSize: '0.9rem',
            fontWeight: 'bold'
          }}
        >
          📈 Upgrades
        </button>
      </div>

      {/* Upgrades Modal */}
      {showUpgrades && (
        <div
          style={{
            position: 'fixed',
            top: 0,
            left: 0,
            right: 0,
            bottom: 0,
            background: 'rgba(0, 0, 0, 0.85)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            zIndex: 1000,
            padding: '2rem',
            overflowY: 'auto'
          }}
          onClick={() => setShowUpgrades(false)}
        >
          <div
            style={{
              background: '#1a1a2e',
              border: '2px solid rgba(255, 255, 255, 0.2)',
              borderRadius: '8px',
              maxWidth: '800px',
              width: '100%',
              maxHeight: '90vh',
              overflowY: 'auto',
              position: 'relative'
            }}
            onClick={(e) => e.stopPropagation()}
          >
            <button
              onClick={() => setShowUpgrades(false)}
              style={{
                position: 'absolute',
                top: '1rem',
                right: '1rem',
                background: 'rgba(255, 0, 0, 0.2)',
                border: '2px solid rgba(255, 0, 0, 0.4)',
                borderRadius: '4px',
                color: '#ff4444',
                cursor: 'pointer',
                padding: '0.5rem 1rem',
                fontSize: '1rem',
                fontWeight: 'bold',
                zIndex: 10
              }}
            >
              ✕ Close
            </button>
            <UpgradesPanel />
          </div>
        </div>
      )}
    </>
  )
}
