import { useState, useEffect, useRef } from 'react'
import { createBoard, placeStone, captureStones, countTerritory, getStone, getTerritoryMap } from '../../core/go/board'
import { getAIMove } from '../../core/ai/simpleAI'
import { AI_PRESETS } from '../../core/ai/types'
import { useAIStatsStore } from '../../state/aiStatsStore'
import { estimateScore } from '../../core/ai/evaluation'
import type { Board } from '../../core/go/types'
import type { AIConfig } from '../../core/ai/types'
import type { MatchResult } from '../../state/aiStatsStore'
import './WatchPage.css'

interface MoveHistory {
  moveNumber: number
  player: 'black' | 'white'
  blackScore: number
  whiteScore: number
  blackWinProb: number
}

export function WatchPage() {
  // Parse URL params with defaults
  const searchParams = new URLSearchParams(window.location.search)
  const urlBoardSize = Number(searchParams.get('size')) || 9
  const urlBlackLevel = (Number(searchParams.get('black')) || 2) as 1 | 2 | 3 | 4 | 5
  const urlWhiteLevel = (Number(searchParams.get('white')) || 2) as 1 | 2 | 3 | 4 | 5
  const urlMoveSpeed = Number(searchParams.get('speed')) || 500
  const urlMaxMoves = Number(searchParams.get('maxMoves')) || 50
  const urlMaxMovesEnabled = searchParams.get('maxMovesEnabled') !== 'false' // default true

  const [board, setBoard] = useState<Board>(() => createBoard(urlBoardSize))
  const [previousBoard, setPreviousBoard] = useState<Board | undefined>(undefined)
  const [currentPlayer, setCurrentPlayer] = useState<'black' | 'white'>('black')
  const [captures, setCaptures] = useState({ black: 0, white: 0 })
  const [moveCount, setMoveCount] = useState(0)
  const [gameOver, setGameOver] = useState(false)
  const [isPlaying, setIsPlaying] = useState(false)
  const [autoReplay, setAutoReplay] = useState(false)
  const [showStats, setShowStats] = useState(false)
  const [showTerritory, setShowTerritory] = useState(true)
  const [showAdvancedAI, setShowAdvancedAI] = useState(false)
  const [moveHistory, setMoveHistory] = useState<MoveHistory[]>([])
  const [completedBoards, setCompletedBoards] = useState<Board[]>([])

  // AI Configuration - Initialize from URL
  const [boardSize, setBoardSize] = useState(urlBoardSize)
  const [blackLevel, setBlackLevel] = useState<1 | 2 | 3 | 4 | 5>(urlBlackLevel)
  const [whiteLevel, setWhiteLevel] = useState<1 | 2 | 3 | 4 | 5>(urlWhiteLevel)
  const [moveSpeed, setMoveSpeed] = useState(urlMoveSpeed)
  const [maxMoves, setMaxMoves] = useState(urlMaxMoves)
  const [maxMovesEnabled, setMaxMovesEnabled] = useState(urlMaxMovesEnabled)

  // Custom AI params
  const [blackConfig, setBlackConfig] = useState<AIConfig>(AI_PRESETS[2])
  const [whiteConfig, setWhiteConfig] = useState<AIConfig>(AI_PRESETS[2])

  // Stats
  const addMatch = useAIStatsStore((state) => state.addMatch)
  const getStats = useAIStatsStore((state) => state.getStats)
  const getLeaderboard = useAIStatsStore((state) => state.getLeaderboard)
  const clearHistory = useAIStatsStore((state) => state.clearHistory)

  const timeoutRef = useRef<NodeJS.Timeout>()

  // Sync URL params with state
  useEffect(() => {
    const params = new URLSearchParams()

    // Only set params if they differ from defaults to keep URL clean
    if (boardSize !== 9) params.set('size', boardSize.toString())
    if (blackLevel !== 2) params.set('black', blackLevel.toString())
    if (whiteLevel !== 2) params.set('white', whiteLevel.toString())
    if (moveSpeed !== 500) params.set('speed', moveSpeed.toString())
    if (maxMoves !== 50) params.set('maxMoves', maxMoves.toString())
    if (!maxMovesEnabled) params.set('maxMovesEnabled', 'false')

    const newUrl = params.toString() ? `${window.location.pathname}?${params.toString()}` : window.location.pathname
    window.history.replaceState(null, '', newUrl)
  }, [boardSize, blackLevel, whiteLevel, moveSpeed, maxMoves, maxMovesEnabled])

  // Update configs when levels change
  useEffect(() => {
    setBlackConfig(AI_PRESETS[blackLevel])
  }, [blackLevel])

  useEffect(() => {
    setWhiteConfig(AI_PRESETS[whiteLevel])
  }, [whiteLevel])

  // Auto-play logic
  useEffect(() => {
    if (!isPlaying || gameOver) {
      if (timeoutRef.current) clearTimeout(timeoutRef.current)
      return
    }

    timeoutRef.current = setTimeout(() => {
      // Check max moves (if enabled)
      if (maxMovesEnabled && moveCount >= maxMoves) {
        endGame()
        return
      }

      const config = currentPlayer === 'black' ? blackConfig : whiteConfig
      const aiMovePos = getAIMove(board, currentPlayer, captures, config, moveCount)

      if (!aiMovePos) {
        // AI decided to pass - game over (simplified end game)
        endGame()
        return
      }

      // Ko rule: pass previous board to placeStone
      const newBoard = placeStone(board, aiMovePos.row, aiMovePos.col, currentPlayer, previousBoard)
      if (!newBoard) {
        // Invalid move (including Ko violation), skip turn
        setCurrentPlayer(currentPlayer === 'black' ? 'white' : 'black')
        return
      }

      const { board: finalBoard, captured } = captureStones(newBoard, aiMovePos.row, aiMovePos.col, currentPlayer)

      if (captured > 0) {
        setCaptures(prev => ({
          ...prev,
          [currentPlayer]: prev[currentPlayer] + captured
        }))
      }

      // Update board history for Ko rule
      setPreviousBoard(board)
      setBoard(finalBoard)
      setCurrentPlayer(currentPlayer === 'black' ? 'white' : 'black')
      setMoveCount(prev => prev + 1)

      // Record move history for probability graph
      const territory = countTerritory(finalBoard)
      const blackScore = territory.black + (captured > 0 && currentPlayer === 'black' ? captures.black + captured : captures.black)
      const whiteScore = territory.white + (captured > 0 && currentPlayer === 'white' ? captures.white + captured : captures.white)
      const scoreDiff = blackScore - whiteScore

      // Convert score difference to win probability (sigmoid-like)
      const blackWinProb = 1 / (1 + Math.exp(-scoreDiff / 10))

      setMoveHistory(prev => [...prev, {
        moveNumber: moveCount + 1,
        player: currentPlayer,
        blackScore,
        whiteScore,
        blackWinProb: blackWinProb * 100
      }])
    }, moveSpeed)

    return () => {
      if (timeoutRef.current) clearTimeout(timeoutRef.current)
    }
  }, [isPlaying, gameOver, board, currentPlayer, moveCount, moveSpeed, maxMoves, blackConfig, whiteConfig])

  const endGame = () => {
    // Calculate final scores
    const territory = countTerritory(board)
    const blackScore = territory.black + captures.black
    const whiteScore = territory.white + captures.white
    const winner = blackScore > whiteScore ? 'black' : blackScore < whiteScore ? 'white' : 'draw'

    // Save final board to completed boards (keep last 5)
    setCompletedBoards(prev => {
      const updated = [...prev, board]
      return updated.slice(-5) // Keep only last 5
    })

    // Record match result
    const matchResult: MatchResult = {
      id: `${Date.now()}-${Math.random()}`,
      timestamp: Date.now(),
      boardSize,
      maxMoves,
      winner,
      blackScore,
      whiteScore,
      blackCaptures: captures.black,
      whiteCaptures: captures.white,
      moveCount,
      blackConfig,
      whiteConfig,
    }
    addMatch(matchResult)

    setGameOver(true)
    setIsPlaying(false)
  }

  const resetGame = () => {
    setBoard(createBoard(boardSize))
    setPreviousBoard(undefined)
    setCurrentPlayer('black')
    setCaptures({ black: 0, white: 0 })
    setMoveCount(0)
    setGameOver(false)
    setMoveHistory([])

    if (autoReplay) {
      setIsPlaying(true)
    }
  }

  const handleBoardSizeChange = (size: number) => {
    setBoardSize(size)
    setBoard(createBoard(size))
    setPreviousBoard(undefined)
    setCurrentPlayer('black')
    setCaptures({ black: 0, white: 0 })
    setMoveCount(0)
    setGameOver(false)
    setIsPlaying(false)
  }

  // Auto-replay when game ends
  useEffect(() => {
    if (gameOver && autoReplay) {
      setTimeout(() => {
        resetGame()
      }, 2000)
    }
  }, [gameOver, autoReplay])

  const territory = countTerritory(board)
  const blackScore = territory.black + captures.black
  const whiteScore = territory.white + captures.white
  const gridSize = Math.min(600, boardSize * 60)
  const territoryMap = showTerritory ? getTerritoryMap(board) : null

  // Current win probability
  const currentWinProb = moveHistory.length > 0 ? moveHistory[moveHistory.length - 1]!.blackWinProb : 50

  return (
    <div className="watch-page">
      <h1 className="watch-title">AI Laboratory</h1>
      <p className="watch-subtitle">Watch artificial minds play the ancient game</p>

      <div className="watch-layout">
        <div className="watch-controls">
          <section className="control-section">
            <h3>Game Controls</h3>
            <div className="control-group">
              <button
                onClick={() => setIsPlaying(!isPlaying)}
                disabled={gameOver}
                className="control-button primary"
              >
                {isPlaying ? '⏸ Pause' : '▶ Play'}
              </button>
              <button onClick={resetGame} className="control-button">
                🔄 Reset
              </button>
              <button onClick={endGame} disabled={gameOver} className="control-button">
                ⏹ Stop
              </button>
            </div>
            <div className="control-row">
              <label>
                <input
                  type="checkbox"
                  checked={autoReplay}
                  onChange={(e) => setAutoReplay(e.target.checked)}
                />
                {' '}Auto-replay
              </label>
            </div>
          </section>

          <section className="control-section">
            <h3>Board Settings</h3>
            <div className="control-row">
              <label>Board Size:</label>
              <select value={boardSize} onChange={(e) => handleBoardSizeChange(Number(e.target.value))}>
                <option value={5}>5x5</option>
                <option value={7}>7x7</option>
                <option value={9}>9x9</option>
                <option value={13}>13x13</option>
                <option value={19}>19x19</option>
              </select>
            </div>
            <div className="control-row">
              <label>
                <input
                  type="checkbox"
                  checked={maxMovesEnabled}
                  onChange={(e) => setMaxMovesEnabled(e.target.checked)}
                />
                {' '}Max Moves:
              </label>
              <input
                type="number"
                value={maxMoves}
                onChange={(e) => setMaxMoves(Number(e.target.value))}
                min={10}
                max={500}
                step={10}
                disabled={!maxMovesEnabled}
              />
            </div>
            <div className="control-row">
              <label>Speed (ms):</label>
              <input
                type="range"
                value={moveSpeed}
                onChange={(e) => setMoveSpeed(Number(e.target.value))}
                min={10}
                max={2000}
                step={10}
              />
              <span>{moveSpeed}ms</span>
            </div>
          </section>

          <section className="control-section">
            <h3>⚫ Black AI</h3>
            <div className="control-row">
              <label>Level:</label>
              <select value={blackLevel} onChange={(e) => setBlackLevel(Number(e.target.value) as 1 | 2 | 3 | 4 | 5)}>
                <option value={1}>1 - Novice (Random, no strategy)</option>
                <option value={2}>2 - Beginner (Learns patterns)</option>
                <option value={3}>3 - Intermediate (Joseki, opening)</option>
                <option value={4}>4 - Advanced (Strategic depth)</option>
                <option value={5}>5 - Expert (Master level)</option>
              </select>
            </div>
            <div className="ai-description">
              {blackLevel === 1 && "Plays randomly with basic capture awareness. Makes mistakes often."}
              {blackLevel === 2 && "Understands basic patterns like Atari→Extend. Avoids filling own territory."}
              {blackLevel === 3 && "Knows joseki patterns, opening theory, and connection. Rarely makes mistakes."}
              {blackLevel === 4 && "Uses advanced opening patterns (Chinese fuseki). Strategic territorial play."}
              {blackLevel === 5 && "Master-level play with optimal strategy and long-term planning."}
            </div>
            <button onClick={() => setShowAdvancedAI(!showAdvancedAI)} className="control-button" style={{ marginTop: '0.5rem', fontSize: '0.85rem' }}>
              {showAdvancedAI ? '▼ Hide Advanced' : '▶ Show Advanced Parameters'}
            </button>
            {showAdvancedAI && (
              <div className="param-grid">
                <label>Capture Weight: <input type="number" value={blackConfig.captureWeight} onChange={(e) => setBlackConfig({...blackConfig, captureWeight: Number(e.target.value)})} /></label>
                <label>Territory Weight: <input type="number" value={blackConfig.territoryWeight} onChange={(e) => setBlackConfig({...blackConfig, territoryWeight: Number(e.target.value)})} /></label>
                <label>Influence Weight: <input type="number" value={blackConfig.influenceWeight} onChange={(e) => setBlackConfig({...blackConfig, influenceWeight: Number(e.target.value)})} /></label>
                <label>Randomness: <input type="number" value={blackConfig.randomness} onChange={(e) => setBlackConfig({...blackConfig, randomness: Number(e.target.value)})} step={0.1} min={0} max={1} /></label>
              </div>
            )}
          </section>

          <section className="control-section">
            <h3>⚪ White AI</h3>
            <div className="control-row">
              <label>Level:</label>
              <select value={whiteLevel} onChange={(e) => setWhiteLevel(Number(e.target.value) as 1 | 2 | 3 | 4 | 5)}>
                <option value={1}>1 - Novice (Random, no strategy)</option>
                <option value={2}>2 - Beginner (Learns patterns)</option>
                <option value={3}>3 - Intermediate (Joseki, opening)</option>
                <option value={4}>4 - Advanced (Strategic depth)</option>
                <option value={5}>5 - Expert (Master level)</option>
              </select>
            </div>
            <div className="ai-description">
              {whiteLevel === 1 && "Plays randomly with basic capture awareness. Makes mistakes often."}
              {whiteLevel === 2 && "Understands basic patterns like Atari→Extend. Avoids filling own territory."}
              {whiteLevel === 3 && "Knows joseki patterns, opening theory, and connection. Rarely makes mistakes."}
              {whiteLevel === 4 && "Uses advanced opening patterns (Chinese fuseki). Strategic territorial play."}
              {whiteLevel === 5 && "Master-level play with optimal strategy and long-term planning."}
            </div>
            {showAdvancedAI && (
              <div className="param-grid">
                <label>Capture Weight: <input type="number" value={whiteConfig.captureWeight} onChange={(e) => setWhiteConfig({...whiteConfig, captureWeight: Number(e.target.value)})} /></label>
                <label>Territory Weight: <input type="number" value={whiteConfig.territoryWeight} onChange={(e) => setWhiteConfig({...whiteConfig, territoryWeight: Number(e.target.value)})} /></label>
                <label>Influence Weight: <input type="number" value={whiteConfig.influenceWeight} onChange={(e) => setWhiteConfig({...whiteConfig, influenceWeight: Number(e.target.value)})} /></label>
                <label>Randomness: <input type="number" value={whiteConfig.randomness} onChange={(e) => setWhiteConfig({...whiteConfig, randomness: Number(e.target.value)})} step={0.1} min={0} max={1} /></label>
              </div>
            )}
          </section>

          <section className="control-section">
            <h3>📊 Statistics</h3>
            <div className="control-group">
              <button onClick={() => setShowStats(!showStats)} className="control-button">
                {showStats ? 'Hide Stats' : 'Show Stats'}
              </button>
              <button onClick={() => { if (confirm('Clear all match history?')) clearHistory() }} className="control-button">
                Clear
              </button>
            </div>
            {showStats && (
              <div className="stats-panel">
                <h4>⚫ Black AI Stats</h4>
                <div className="stats-display">
                  <div>Wins: {getStats(blackConfig).wins}</div>
                  <div>Losses: {getStats(blackConfig).losses}</div>
                  <div>Draws: {getStats(blackConfig).draws}</div>
                  <div>Win Rate: {getStats(blackConfig).winRate.toFixed(1)}%</div>
                  <div>Avg Score: {getStats(blackConfig).averageScore.toFixed(1)}</div>
                </div>

                <h4>⚪ White AI Stats</h4>
                <div className="stats-display">
                  <div>Wins: {getStats(whiteConfig).wins}</div>
                  <div>Losses: {getStats(whiteConfig).losses}</div>
                  <div>Draws: {getStats(whiteConfig).draws}</div>
                  <div>Win Rate: {getStats(whiteConfig).winRate.toFixed(1)}%</div>
                  <div>Avg Score: {getStats(whiteConfig).averageScore.toFixed(1)}</div>
                </div>

                <h4>🏆 Leaderboard</h4>
                <div className="leaderboard">
                  {getLeaderboard().slice(0, 5).map((stat, idx) => (
                    <div key={stat.configSignature} className="leaderboard-entry">
                      <div className="rank">#{idx + 1}</div>
                      <div className="config-name">{stat.configSignature}</div>
                      <div className="winrate">{stat.winRate.toFixed(0)}%</div>
                      <div className="games">({stat.totalGames}g)</div>
                    </div>
                  ))}
                  {getLeaderboard().length === 0 && (
                    <div style={{ opacity: 0.5, textAlign: 'center' }}>No data yet. Play 3+ games to appear.</div>
                  )}
                </div>
              </div>
            )}
          </section>
        </div>

        <div className="watch-game">
          <div className="game-status">
            <div className="status-item">
              <strong>Move:</strong> {moveCount}
            </div>
            <div className="status-item">
              <strong>Current:</strong> <span style={{ color: currentPlayer }}>{currentPlayer}</span>
            </div>
            <div className="status-item">
              <strong>Score:</strong> ⚫ {blackScore} vs ⚪ {whiteScore}
            </div>
          </div>

          {gameOver && (
            <div className="game-over">
              <h2>Game Over</h2>
              <p>{blackScore > whiteScore ? '⚫ Black Wins!' : whiteScore > blackScore ? '⚪ White Wins!' : 'Draw!'}</p>
              <p>Final Score: ⚫ {blackScore} - ⚪ {whiteScore}</p>
            </div>
          )}

          <div
            className="watch-board"
            style={{
              gridTemplateColumns: `repeat(${boardSize}, 1fr)`,
              width: `${gridSize}px`,
              height: `${gridSize}px`,
            }}
          >
            {Array.from({ length: boardSize * boardSize }).map((_, idx) => {
              const row = Math.floor(idx / boardSize)
              const col = idx % boardSize
              const stone = getStone(board, row, col)
              const territoryOwner = territoryMap?.get(`${row},${col}`)

              return (
                <div
                  key={idx}
                  className={`watch-cell ${territoryOwner ? `territory-${territoryOwner}` : ''}`}
                >
                  {stone && <div className={`stone ${stone}`} />}
                </div>
              )
            })}
          </div>

          <div className="territory-info">
            <div>⚫ Territory: {territory.black} | Captures: {captures.black}</div>
            <div>⚪ Territory: {territory.white} | Captures: {captures.white}</div>
            <div>Neutral: {territory.neutral}</div>
          </div>

          {moveHistory.length > 0 && (
            <div className="win-probability-section">
              <h3>Win Probability</h3>
              <div className="prob-current">
                <div className="prob-bar-container">
                  <div className="prob-bar black" style={{ width: `${currentWinProb}%` }}>
                    ⚫ {currentWinProb.toFixed(1)}%
                  </div>
                  <div className="prob-bar white" style={{ width: `${100 - currentWinProb}%` }}>
                    ⚪ {(100 - currentWinProb).toFixed(1)}%
                  </div>
                </div>
              </div>

              <div className="prob-graph">
                <svg width="100%" height="120" viewBox="0 0 600 120" preserveAspectRatio="none">
                  {/* Grid lines */}
                  <line x1="0" y1="60" x2="600" y2="60" stroke="rgba(255,255,255,0.2)" strokeWidth="1" strokeDasharray="4" />
                  <line x1="0" y1="30" x2="600" y2="30" stroke="rgba(255,255,255,0.1)" strokeWidth="1" strokeDasharray="2" />
                  <line x1="0" y1="90" x2="600" y2="90" stroke="rgba(255,255,255,0.1)" strokeWidth="1" strokeDasharray="2" />

                  {/* Win probability line */}
                  <polyline
                    points={moveHistory.map((move, idx) => {
                      const x = (idx / Math.max(moveHistory.length - 1, 1)) * 600
                      const y = 120 - (move.blackWinProb / 100) * 120
                      return `${x},${y}`
                    }).join(' ')}
                    fill="none"
                    stroke="#ffd700"
                    strokeWidth="2"
                  />

                  {/* Move markers */}
                  {moveHistory.map((move, idx) => {
                    const x = (idx / Math.max(moveHistory.length - 1, 1)) * 600
                    const y = 120 - (move.blackWinProb / 100) * 120
                    return (
                      <circle
                        key={idx}
                        cx={x}
                        cy={y}
                        r="3"
                        fill={move.player === 'black' ? '#000' : '#fff'}
                        stroke={move.player === 'black' ? '#fff' : '#000'}
                        strokeWidth="1"
                      />
                    )
                  })}
                </svg>
                <div className="graph-labels">
                  <span>⚫ Black favored</span>
                  <span>50%</span>
                  <span>⚪ White favored</span>
                </div>
              </div>
            </div>
          )}

          {completedBoards.length > 0 && (
            <div className="completed-boards-section">
              <h3>Last {completedBoards.length} Completed Games</h3>
              <div className="completed-boards-column">
                {completedBoards.map((completedBoard, idx) => {
                  const miniSize = Math.min(150, completedBoard.size * 12)
                  return (
                    <div key={idx} className="mini-board-container">
                      <div className="mini-board-label">Game #{completedBoards.length - idx}</div>
                      <div
                        className="mini-board"
                        style={{
                          gridTemplateColumns: `repeat(${completedBoard.size}, 1fr)`,
                          width: `${miniSize}px`,
                          height: `${miniSize}px`,
                        }}
                      >
                        {Array.from({ length: completedBoard.size * completedBoard.size }).map((_, cellIdx) => {
                          const row = Math.floor(cellIdx / completedBoard.size)
                          const col = cellIdx % completedBoard.size
                          const stone = getStone(completedBoard, row, col)
                          return (
                            <div key={cellIdx} className="mini-cell">
                              {stone && <div className={`mini-stone ${stone}`} />}
                            </div>
                          )
                        })}
                      </div>
                    </div>
                  )
                })}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
