import { useState, useEffect } from 'react'
import { createBoard, placeStone, captureStones, countTerritory, getStone, getGroup, countLiberties } from '../../core/go/board'
import { getAIMove } from '../../core/ai/simpleAI'
import { useGameStore } from '../../state/gameStore'
import { useCurrencyStore } from '../../state/currencyStore'
import { useStatsStore } from '../../state/statsStore'
import type { Board } from '../../core/go/types'
import './Milestone.css'

export function Milestone2() {
  const phase = useGameStore((state) => state.phase)
  const completePhase = useGameStore((state) => state.completePhase)
  const earnStones = useCurrencyStore((state) => state.earnStones)
  const recordGame = useStatsStore((state) => state.recordGame)

  const [board, setBoard] = useState<Board>(() => createBoard(3))
  const [previousBoard, setPreviousBoard] = useState<Board | undefined>(undefined)
  const [currentPlayer, setCurrentPlayer] = useState<'black' | 'white'>('black')
  const [message, setMessage] = useState('')
  const [captures, setCaptures] = useState({ black: 0, white: 0 })
  const [gameOver, setGameOver] = useState(false)

  const isPractice = phase.startsWith('M2_PRACTICE')
  const isAIOpponent = phase === 'M2_AI_OPPONENT'

  useEffect(() => {
    if (isPractice) {
      setBoard(createBoard(3))
      setPreviousBoard(undefined)
      setCurrentPlayer('black')
      setMessage('Practice your placement. Click anywhere on the board.')
      setGameOver(false)
    } else if (isAIOpponent) {
      setBoard(createBoard(5))
      setPreviousBoard(undefined)
      setCurrentPlayer('black')
      setMessage('You\'ve learned placement. Now learn opposition.')
      setCaptures({ black: 0, white: 0 })
      setGameOver(false)
    }
  }, [phase])

  // AI move
  useEffect(() => {
    if (isAIOpponent && currentPlayer === 'white' && !gameOver) {
      const timer = setTimeout(() => {
        const aiMovePos = getAIMove(board, 'white', captures)
        if (aiMovePos) {
          handleMove(aiMovePos.row, aiMovePos.col, 'white')
        }
      }, 500)

      return () => clearTimeout(timer)
    }
  }, [board, currentPlayer, isAIOpponent, gameOver, captures])

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
    if (isPractice) {
      // Practice mode: just place one stone and complete
      const newBoard = placeStone(board, row, col, 'black')
      if (!newBoard) return

      setBoard(newBoard)
      setMessage('Well done! +10 Stones')
      earnStones(10)

      setTimeout(() => {
        completePhase()
      }, 1500)
    } else if (isAIOpponent && currentPlayer === 'black' && !gameOver) {
      handleMove(row, col, 'black')
    }
  }

  const handlePass = () => {
    if (!isAIOpponent || gameOver) return

    // Simple end game: whoever has more territory + captures wins
    const territory = countTerritory(board)
    const blackScore = territory.black + captures.black
    const whiteScore = territory.white + captures.white

    const winner = blackScore > whiteScore ? 'black' : blackScore < whiteScore ? 'white' : 'draw'

    if (winner === 'black') {
      setMessage('Victory! +50 Stones')
      earnStones(50)
      recordGame('win')
    } else if (winner === 'white') {
      setMessage('Defeat. +10 Stones')
      earnStones(10)
      recordGame('loss')
    } else {
      setMessage('Draw! +25 Stones')
      earnStones(25)
      recordGame('draw')
    }

    setGameOver(true)

    setTimeout(() => {
      completePhase()
    }, 3000)
  }

  if (!isPractice && !isAIOpponent) {
    return null
  }

  const gridSize = Math.min(400, board.size * 70)

  return (
    <div className="milestone">
      <h1 className="milestone-title">
        {isPractice ? `Practice Move ${phase.slice(-1)}` : 'First Opponent'}
      </h1>
      <p className="milestone-message">{message}</p>

      {isAIOpponent && !gameOver && (
        <div className="game-controls">
          <div>Current: <strong>{currentPlayer}</strong></div>
          <button onClick={handlePass} className="pass-button">Pass / End Game</button>
        </div>
      )}

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
    </div>
  )
}
