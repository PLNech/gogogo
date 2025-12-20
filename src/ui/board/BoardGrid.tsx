import './BoardGrid.css'

export interface CellData {
  stone: 'black' | 'white' | null
  territory?: 'black' | 'white' | null
  highlight?: string // CSS class to add
  marker?: string // Text marker to show on stone
}

export interface BoardGridProps {
  size: number
  getCellData: (row: number, col: number) => CellData
  onCellClick?: (row: number, col: number) => void
  interactive?: boolean
  variant?: 'default' | 'mini'
  className?: string
}

/**
 * Shared board grid component for Go board rendering.
 * Used by BoardView, WatchPage, and other board displays.
 */
export function BoardGrid({
  size,
  getCellData,
  onCellClick,
  interactive = false,
  variant = 'default',
  className = '',
}: BoardGridProps) {
  const isMini = variant === 'mini'
  const gridSize = isMini
    ? Math.min(150, size * 12)
    : Math.min(600, size * 60)

  const handleClick = (row: number, col: number) => {
    if (interactive && onCellClick) {
      onCellClick(row, col)
    }
  }

  return (
    <div
      className={`board-grid ${isMini ? 'mini' : ''} ${interactive ? 'interactive' : ''} ${className}`}
      style={{
        gridTemplateColumns: `repeat(${size}, 1fr)`,
        width: gridSize,
        height: gridSize,
      }}
    >
      {Array.from({ length: size * size }).map((_, idx) => {
        const row = Math.floor(idx / size)
        const col = idx % size
        const cell = getCellData(row, col)

        return (
          <div
            key={`${row}-${col}`}
            className={`board-cell ${cell.territory ? `territory-${cell.territory}` : ''} ${cell.highlight || ''}`}
            onClick={() => handleClick(row, col)}
          >
            {cell.stone && (
              <div className={`stone ${cell.stone}`}>
                {cell.marker && <span className="stone-marker">{cell.marker}</span>}
              </div>
            )}
          </div>
        )
      })}
    </div>
  )
}

export default BoardGrid
