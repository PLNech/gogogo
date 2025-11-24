import { useState, useEffect } from 'react'
import { createBoard, placeStone, captureStones, countTerritory, getStone, getTerritoryMap, getGroup, countLiberties } from '../../core/go/board'
import { getAIMove } from '../../core/ai/simpleAI'
import { useCurrencyStore } from '../../state/currencyStore'
import { useGameStore } from '../../state/gameStore'
import type { Board } from '../../core/go/types'
import './BoardView.css'

interface BoardViewProps {
  initialSize?: number
  showDebug?: boolean
  aiEnabled?: boolean
}

export function BoardView({ initialSize = 3, showDebug = false, aiEnabled = false }: BoardViewProps) {
  const [board, setBoard] = useState<Board>(() => createBoard(initialSize))
  const [previousBoard, setPreviousBoard] = useState<Board | undefined>(undefined)
  const [currentPlayer, setCurrentPlayer] = useState<'black' | 'white'>('black')
  const [captures, setCaptures] = useState({ black: 0, white: 0 })
  const [boardSize, setBoardSize] = useState(initialSize)
  const [aiActive, setAIActive] = useState(aiEnabled)
  const [showTerritory, setShowTerritory] = useState(false)
  const earnStones = useCurrencyStore((state) => state.earnStones)
  const resetGame = useGameStore((state) => state.reset)

  // AI move effect
  useEffect(() => {
    if (aiActive && currentPlayer === 'white') {
      const timer = setTimeout(() => {
        const aiMovePos = getAIMove(board, 'white', captures)
        if (aiMovePos) {
          handleMove(aiMovePos.row, aiMovePos.col, 'white')
        }
      }, 500) // Small delay for better UX

      return () => clearTimeout(timer)
    }
  }, [board, currentPlayer, aiActive, captures])

  const handleMove = (row: number, col: number, player: 'black' | 'white') => {
    // Ko rule: pass previous board to placeStone
    const newBoard = placeStone(board, row, col, player, previousBoard)
    if (!newBoard) return // Invalid move (including Ko violation)

    // Check for captures FIRST
    const { board: boardAfterCaptures, captured } = captureStones(newBoard, row, col, player)

    // Now check for self-capture (after captures are removed)
    // If the move doesn't capture anything, it cannot be self-capture
    if (captured === 0) {
      const group = getGroup(boardAfterCaptures, row, col)
      const liberties = countLiberties(boardAfterCaptures, group)
      if (liberties === 0) {
        return // Self-capture is illegal
      }
    }

    if (captured > 0) {
      setCaptures(prev => ({
        ...prev,
        [player]: prev[player] + captured
      }))
    }

    // Update board history for Ko rule
    setPreviousBoard(board)
    setBoard(boardAfterCaptures)
    setCurrentPlayer(player === 'black' ? 'white' : 'black')
  }

  const handleCellClick = (row: number, col: number) => {
    // Only allow player moves for black (or both if AI disabled)
    if (aiActive && currentPlayer === 'white') return

    handleMove(row, col, currentPlayer)
  }

  const handleSizeChange = (size: number) => {
    setBoard(createBoard(size))
    setPreviousBoard(undefined)
    setBoardSize(size)
    setCurrentPlayer('black')
    setCaptures({ black: 0, white: 0 })
  }

  const territory = countTerritory(board)
  const territoryMap = showTerritory ? getTerritoryMap(board) : null
  const gridSize = Math.min(600, board.size * 60)

  return (
    <div className="board-view">
      {showDebug && (
        <div className="debug-controls">
          <h3>Debug Controls</h3>
          <div>
            <label>Board Size: </label>
            {[1, 3, 5, 7, 9].map(size => (
              <button
                key={size}
                onClick={() => handleSizeChange(size)}
                disabled={size === boardSize}
              >
                {size}x{size}
              </button>
            ))}
          </div>
          <div>
            <label>
              <input
                type="checkbox"
                checked={aiActive}
                onChange={(e) => setAIActive(e.target.checked)}
              />
              {' '}Enable AI (White)
            </label>
          </div>
          <div>
            <label>
              <input
                type="checkbox"
                checked={showTerritory}
                onChange={(e) => setShowTerritory(e.target.checked)}
              />
              {' '}Show Territory
            </label>
          </div>
          <div>
            <button onClick={() => handleSizeChange(boardSize)}>Reset Board</button>
            <button onClick={() => earnStones(10)}>+10 Stones (test)</button>
            <button
              onClick={() => {
                if (confirm('Reset game progression to M1? (keeps currency)')) {
                  resetGame()
                }
              }}
              style={{ background: 'rgba(255, 0, 0, 0.2)', borderColor: 'rgba(255, 0, 0, 0.4)' }}
            >
              🔄 Reset Progression
            </button>
          </div>
        </div>
      )}

      <div className="game-info">
        <div>Current Player: <span style={{ color: currentPlayer }}>{currentPlayer}</span></div>
        <div>Captures - Black: {captures.black}, White: {captures.white}</div>
        <div>Territory - Black: {territory.black}, White: {territory.white}, Neutral: {territory.neutral}</div>
      </div>

      <div
        className="board-grid"
        style={{
          gridTemplateColumns: `repeat(${board.size}, 1fr)`,
          width: `${gridSize}px`,
          height: `${gridSize}px`,
        }}
      >
        {Array.from({ length: board.size * board.size }).map((_, idx) => {
          const row = Math.floor(idx / board.size)
          const col = idx % board.size
          const stone = getStone(board, row, col)
          const territoryOwner = territoryMap?.get(`${row},${col}`)

          return (
            <div
              key={idx}
              className={`board-cell ${stone || ''} ${territoryOwner ? `territory-${territoryOwner}` : ''}`}
              onClick={() => handleCellClick(row, col)}
            >
              {stone && <div className={`stone ${stone}`} />}
            </div>
          )
        })}
      </div>
    </div>
  )
}
