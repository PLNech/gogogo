import { useState, useEffect } from 'react'
import { createBoard, placeStone, captureStones, getStone } from '../../core/go/board'
import { useGameStore } from '../../state/gameStore'
import { useCurrencyStore } from '../../state/currencyStore'
import type { Board } from '../../core/go/types'
import './Milestone.css'

export function Milestone1() {
  const phase = useGameStore((state) => state.phase)
  const completePhase = useGameStore((state) => state.completePhase)
  const earnStones = useCurrencyStore((state) => state.earnStones)

  const [board, setBoard] = useState<Board>(() => createBoard(1))
  const [message, setMessage] = useState('A journey of ten thousand games begins with a single stone.')
  const [showBoard, setShowBoard] = useState(true)

  useEffect(() => {
    if (phase === 'M1_INTRO') {
      setBoard(createBoard(1))
      setMessage('A journey of ten thousand games begins with a single stone.')
      setShowBoard(true)
    } else if (phase === 'M1_CAPTURE') {
      // Set up 3x3 board with white stone that has ONE liberty
      let newBoard = createBoard(3)
      newBoard = placeStone(newBoard, 1, 1, 'white')!
      newBoard = placeStone(newBoard, 0, 1, 'black')!
      newBoard = placeStone(newBoard, 1, 0, 'black')!
      newBoard = placeStone(newBoard, 1, 2, 'black')!
      // Leave (2,1) empty - this is where player must place to capture
      setBoard(newBoard)
      setMessage('The board expands. A white stone clings to life. One move will capture it.')
      setShowBoard(true)
    }
  }, [phase])

  const handleCellClick = (row: number, col: number) => {
    if (phase === 'M1_INTRO') {
      // Place black stone on 1x1
      const newBoard = placeStone(board, row, col, 'black')
      if (!newBoard) return

      setBoard(newBoard)
      setMessage('Victory! +1 Stone')
      earnStones(1)

      // Auto-advance after delay
      setTimeout(() => {
        completePhase()
      }, 1500)
    } else if (phase === 'M1_CAPTURE') {
      // Player captures the white stone
      const newBoard = placeStone(board, row, col, 'black')
      if (!newBoard) return

      const { board: finalBoard, captured } = captureStones(newBoard, row, col, 'black')

      if (captured > 0) {
        setBoard(finalBoard)
        setMessage('The white stone is captured! +5 Stones')
        earnStones(5)

        // Complete M1
        setTimeout(() => {
          completePhase()
        }, 2000)
      } else {
        setBoard(finalBoard)
      }
    }
  }

  if (phase !== 'M1_INTRO' && phase !== 'M1_CAPTURE') {
    return null
  }

  const gridSize = board.size === 1 ? 100 : Math.min(300, board.size * 80)

  return (
    <div className="milestone">
      <h1 className="milestone-title">GoGoGo</h1>
      <p className="milestone-message">{message}</p>

      {showBoard && (
        <div
          className="milestone-board"
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

            return (
              <div
                key={idx}
                className={`milestone-cell ${stone || ''}`}
                onClick={() => handleCellClick(row, col)}
              >
                {stone && <div className={`stone ${stone}`} />}
              </div>
            )
          })}
        </div>
      )}
    </div>
  )
}
