import { useCurrencyStore } from '../../state/currencyStore'
import './CurrencyDisplay.css'

export function CurrencyDisplay() {
  const stones = useCurrencyStore((state) => state.stones)

  return (
    <div className="currency-display">
      <div className="currency-item">
        <span className="currency-icon">⚫</span>
        <span className="currency-amount">{stones}</span>
        <span className="currency-label">Stones</span>
      </div>
    </div>
  )
}
