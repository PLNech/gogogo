import { useState, useEffect } from 'react'
import { createBoard, placeStone, captureStones, countTerritory, getStone, getTerritoryMap, getGroup, countLiberties } from '../../core/go/board'
import { getAIMove } from '../../core/ai/simpleAI'
import { useCurrencyStore } from '../../state/currencyStore'
import { useGameStore } from '../../state/gameStore'
import { useUpgradeStore } from '../../state/upgradeStore'
import { getCurrentMoveCount, getUnlockedBoardSizes } from '../../domain/upgrades/upgrades'
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
  const [moveCount, setMoveCount] = useState(0)
  const [boardSize, setBoardSize] = useState(initialSize)
  const [aiActive, setAIActive] = useState(aiEnabled)
  const [showTerritory, setShowTerritory] = useState(false)
  const [gameOver, setGameOver] = useState(false)
  const [consecutivePasses, setConsecutivePasses] = useState(0)
  const [lastPassPlayer, setLastPassPlayer] = useState<'black' | 'white' | null>(null)

  const earnStones = useCurrencyStore((state) => state.earnStones)
  const resetGame = useGameStore((state) => state.reset)
  const purchased = useUpgradeStore((state) => state.purchased)
  const maxMoves = getCurrentMoveCount(purchased)
  const unlockedSizes = getUnlockedBoardSizes(purchased)

  // AI move effect
  useEffect(() => {
    if (aiActive && currentPlayer === 'white' && !gameOver) {
      const timer = setTimeout(() => {
        const aiMovePos = getAIMove(board, 'white', captures, undefined, moveCount)
        if (aiMovePos) {
          handleMove(aiMovePos.row, aiMovePos.col, 'white')
        } else {
          // AI passes
          handlePass('white')
        }
      }, 500) // Small delay for better UX

      return () => clearTimeout(timer)
    }
  }, [board, currentPlayer, aiActive, captures, moveCount, gameOver])

  const handleMove = (row: number, col: number, player: 'black' | 'white') => {
    if (gameOver) return

    // Ko rule: pass previous board to placeStone
    const newBoard = placeStone(board, row, col, player, previousBoard)
    if (!newBoard) return // Invalid move (including Ko violation)

    // Check for captures FIRST
    const { board: boardAfterCaptures, captured } = captureStones(newBoard, row, col, player)

    // Now check for self-capture (after captures are removed)
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
    setMoveCount(prev => prev + 1)
    setConsecutivePasses(0) // Reset pass counter on move
    setLastPassPlayer(null)

    // Check if game should end (max moves)
    if (moveCount + 1 >= maxMoves) {
      endGame()
    }
  }

  const handlePass = (player: 'black' | 'white') => {
    if (gameOver) return

    // Check if both players passed consecutively
    if (lastPassPlayer && lastPassPlayer !== player) {
      // Both players passed - game over
      endGame()
      return
    }

    setLastPassPlayer(player)
    setConsecutivePasses(prev => prev + 1)
    setCurrentPlayer(player === 'black' ? 'white' : 'black')
    setMoveCount(prev => prev + 1)
  }

  const endGame = () => {
    setGameOver(true)

    // Calculate final scores
    const territory = countTerritory(board)
    const blackScore = territory.black + captures.black
    const whiteScore = territory.white + captures.white

    // Reward system:
    // - Draw: 1 stone
    // - Victory: 1 stone + 1 per point of victory margin
    let stonesToEarn = 0
    if (blackScore === whiteScore) {
      // Draw
      stonesToEarn = 1
    } else if (blackScore > whiteScore) {
      // Black wins (player)
      const margin = blackScore - whiteScore
      stonesToEarn = 1 + margin
    } else {
      // White wins (AI) - still give 1 stone for playing
      stonesToEarn = 1
    }

    console.log(`🎮 Game Over! Black: ${blackScore}, White: ${whiteScore}`)
    console.log(`💰 Earning ${stonesToEarn} stones! (${blackScore > whiteScore ? 'Victory!' : blackScore === whiteScore ? 'Draw' : 'Defeat'})`)

    earnStones(stonesToEarn)

    console.log(`✅ earnStones(${stonesToEarn}) called`)
  }

  const handleNewGame = () => {
    setBoard(createBoard(boardSize))
    setPreviousBoard(undefined)
    setCurrentPlayer('black')
    setCaptures({ black: 0, white: 0 })
    setMoveCount(0)
    setGameOver(false)
    setConsecutivePasses(0)
    setLastPassPlayer(null)
  }

  const handleCellClick = (row: number, col: number) => {
    if (gameOver) return
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
    setMoveCount(0)
    setGameOver(false)
    setConsecutivePasses(0)
    setLastPassPlayer(null)
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
            {unlockedSizes.map(size => (
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
            <button onClick={handleNewGame}>New Game</button>
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

          {/* Game State Debug Panel */}
          <div style={{
            marginTop: '1rem',
            padding: '0.5rem',
            background: 'rgba(255, 255, 255, 0.05)',
            border: '1px solid rgba(255, 255, 255, 0.1)',
            borderRadius: '4px',
            fontSize: '0.85rem'
          }}>
            <h4 style={{ margin: '0 0 0.5rem 0' }}>Game State</h4>
            <div>Move: {moveCount} / {maxMoves}</div>
            <div>Game Over: {gameOver ? 'Yes' : 'No'}</div>
            <div>Consecutive Passes: {consecutivePasses}</div>
            <div>Last Pass: {lastPassPlayer || 'none'}</div>
            <div>Black Score: {territory.black + captures.black}</div>
            <div>White Score: {territory.white + captures.white}</div>
          </div>
        </div>
      )}

      <div className="game-info">
        <div>
          <strong>Move {moveCount} / {maxMoves}</strong>
          {gameOver && <span style={{ color: '#4CAF50', marginLeft: '1rem' }}>GAME OVER</span>}
        </div>
        <div>Current Player: <span style={{ color: currentPlayer }}>{currentPlayer}</span></div>
        <div>Captures - Black: {captures.black}, White: {captures.white}</div>
        <div>Territory - Black: {territory.black}, White: {territory.white}, Neutral: {territory.neutral}</div>
        <div>Score - ⚫ {territory.black + captures.black} vs ⚪ {territory.white + captures.white}</div>
      </div>

      {/* Pass Button */}
      {!gameOver && (
        <div style={{ marginBottom: '1rem' }}>
          <button
            onClick={() => handlePass(currentPlayer)}
            disabled={aiActive && currentPlayer === 'white'}
            style={{
              padding: '0.5rem 1rem',
              fontSize: '1rem',
              background: 'rgba(255, 193, 7, 0.2)',
              border: '2px solid rgba(255, 193, 7, 0.4)',
              borderRadius: '4px',
              cursor: 'pointer',
              color: '#FFC107'
            }}
          >
            Pass {aiActive && currentPlayer === 'white' ? '(AI Turn)' : ''}
          </button>
          {lastPassPlayer && (
            <span style={{ marginLeft: '1rem', color: '#FFC107' }}>
              {lastPassPlayer === 'black' ? '⚫' : '⚪'} passed. Pass again to end game.
            </span>
          )}
        </div>
      )}

      {/* Game Over Panel */}
      {gameOver && (
        <div style={{
          marginBottom: '1rem',
          padding: '1rem',
          background: 'rgba(76, 175, 80, 0.2)',
          border: '2px solid rgba(76, 175, 80, 0.4)',
          borderRadius: '4px'
        }}>
          <h3>Game Over!</h3>
          <div>⚫ Black: {territory.black + captures.black} points</div>
          <div>⚪ White: {territory.white + captures.white} points</div>
          <div style={{ marginTop: '0.5rem', fontSize: '1.2rem', color: '#4CAF50' }}>
            Winner: {territory.black + captures.black > territory.white + captures.white ? '⚫ Black' :
                     territory.white + captures.white > territory.black + captures.black ? '⚪ White' :
                     '⚖️ Draw'}
          </div>
          <button
            onClick={handleNewGame}
            style={{
              marginTop: '1rem',
              padding: '0.5rem 1rem',
              fontSize: '1rem',
              background: 'rgba(76, 175, 80, 0.3)',
              border: '2px solid rgba(76, 175, 80, 0.6)',
              borderRadius: '4px',
              cursor: 'pointer',
              color: '#4CAF50'
            }}
          >
            New Game
          </button>
        </div>
      )}

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
