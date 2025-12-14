import { useMemo } from 'react'
import { useUpgradeStore } from '../../state/upgradeStore'
import { useCurrencyStore } from '../../state/currencyStore'
import { UPGRADES, getAvailableUpgrades } from '../../domain/upgrades/upgrades'
import './UpgradesPanel.css'

export function UpgradesPanel() {
  const purchased = useUpgradeStore((state) => state.purchased)
  const availableIds = useMemo(() => getAvailableUpgrades(purchased), [purchased])
  const purchaseUpgrade = useUpgradeStore((state) => state.purchaseUpgrade)
  const canAfford = useCurrencyStore((state) => state.canAfford)

  if (availableIds.length === 0) {
    return (
      <div className="upgrades-panel">
        <h2>Upgrades</h2>
        <p className="no-upgrades">All available upgrades purchased! 🎉</p>
      </div>
    )
  }

  return (
    <div className="upgrades-panel">
      <h2>Upgrades</h2>
      <div className="upgrades-grid">
        {availableIds.map((id) => {
          const upgrade = UPGRADES[id]!
          const affordable = canAfford(upgrade.cost)

          return (
            <div key={id} className={`upgrade-card ${affordable ? 'affordable' : 'expensive'}`}>
              <h3 className="upgrade-name">{upgrade.name}</h3>
              <p className="upgrade-description">{upgrade.description}</p>
              <div className="upgrade-footer">
                <span className="upgrade-cost">
                  ⚫ {upgrade.cost}
                </span>
                <button
                  className="upgrade-button"
                  onClick={() => purchaseUpgrade(id)}
                  disabled={!affordable}
                >
                  {affordable ? 'Purchase' : 'Not enough stones'}
                </button>
              </div>
            </div>
          )
        })}
      </div>
    </div>
  )
}
