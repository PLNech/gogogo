import { useState, useEffect } from 'react'
import './InstinctBattleground.css'

// Instinct data from our 2000-game Atari Go experiment
const INSTINCT_DATA = {
  hane_vs_tsuke: {
    name: 'Hane vs Tsuke',
    japanese: 'ツケにはハネ',
    description: 'When opponent attaches, wrap around with hane',
    advantage: 12.1,
    fired: 75201,
    followRate: 12.8,
    rank: 1,
    pattern: [
      '  . . . . .  ',
      '  . . B . .  ',
      '  . B W . .  ',
      '  . . H . .  ',
      '  . . . . .  ',
    ],
    verdict: 'Champion',
  },
  extend_from_atari: {
    name: 'Extend from Atari',
    japanese: 'アタリから伸びよ',
    description: 'When your stone is in atari, extend to gain liberties',
    advantage: 5.2,
    fired: 36985,
    followRate: 4.9,
    rank: 2,
    pattern: [
      '  . . . . .  ',
      '  . W W . .  ',
      '  . W B W .  ',
      '  . . E . .  ',
      '  . . . . .  ',
    ],
    verdict: 'Confirmed',
  },
  block_the_thrust: {
    name: 'Block the Thrust',
    japanese: 'ツキアタリには',
    description: 'Block when opponent thrusts into your formation',
    advantage: 3.7,
    fired: 38171,
    followRate: 5.5,
    rank: 3,
    pattern: [
      '  . . . . .  ',
      '  . B . B .  ',
      '  . . W . .  ',
      '  . . H . .  ',
      '  . . . . .  ',
    ],
    verdict: 'Confirmed',
  },
  block_the_angle: {
    name: 'Block the Angle',
    japanese: 'カケにはオサエ',
    description: 'Block opponent\'s diagonal approach',
    advantage: 3.5,
    fired: 76785,
    followRate: 21.4,
    rank: 4,
    pattern: [
      '  . . . . .  ',
      '  . . W . .  ',
      '  . . H . .  ',
      '  . B . . .  ',
      '  . . . . .  ',
    ],
    verdict: 'Works',
  },
  connect_vs_peep: {
    name: 'Connect vs Peep',
    japanese: 'ノゾキにはツギ',
    description: '"Even a moron connects against a peep"',
    advantage: 2.7,
    fired: 54683,
    followRate: 9.3,
    rank: 5,
    pattern: [
      '  . . . . .  ',
      '  . B . B .  ',
      '  . . C . .  ',
      '  . . W . .  ',
      '  . . . . .  ',
    ],
    verdict: '"Even a moron"',
  },
  stretch_from_bump: {
    name: 'Stretch from Bump',
    japanese: 'ブツカリから伸びよ',
    description: 'When opponent bumps with support, stretch away',
    advantage: 2.1,
    fired: 39619,
    followRate: 5.7,
    rank: 6,
    pattern: [
      '  . . . . .  ',
      '  . . W W .  ',
      '  . . B . .  ',
      '  . . S . .  ',
      '  . . . . .  ',
    ],
    verdict: 'Slight positive',
  },
  stretch_from_kosumi: {
    name: 'Stretch from Kosumi',
    japanese: 'コスミから伸びよ',
    description: 'Stretch away from opponent\'s diagonal contact',
    advantage: 2.0,
    fired: 76397,
    followRate: 25.8,
    rank: 7,
    pattern: [
      '  . . . . .  ',
      '  . . W . .  ',
      '  . . . B .  ',
      '  . . . S .  ',
      '  . . . . .  ',
    ],
    verdict: 'Slight positive',
  },
  hane_at_head_of_two: {
    name: 'Hane at Head of Two',
    japanese: '二子の頭にハネ',
    description: 'Play at the head of two opponent stones',
    advantage: -0.9,
    fired: 68468,
    followRate: 10.9,
    rank: 8,
    pattern: [
      '  . . . . .  ',
      '  . . H . .  ',
      '  . . W . .  ',
      '  . . W . .  ',
      '  . . . . .  ',
    ],
    verdict: 'Strategic (needs time)',
  },
}

type InstinctKey = keyof typeof INSTINCT_DATA

interface PatternBoardProps {
  pattern: string[]
  size?: number
}

function PatternBoard({ pattern, size = 5 }: PatternBoardProps) {
  const cellSize = 28
  const padding = 20
  const boardSize = cellSize * (size - 1) + padding * 2

  return (
    <svg width={boardSize} height={boardSize} className="pattern-board">
      {/* Board background */}
      <rect width={boardSize} height={boardSize} fill="#DEB887" rx={4} />

      {/* Grid lines */}
      {Array.from({ length: size }).map((_, i) => (
        <g key={i}>
          <line
            x1={padding}
            y1={padding + i * cellSize}
            x2={boardSize - padding}
            y2={padding + i * cellSize}
            stroke="#4A4036"
            strokeWidth={1}
          />
          <line
            x1={padding + i * cellSize}
            y1={padding}
            x2={padding + i * cellSize}
            y2={boardSize - padding}
            stroke="#4A4036"
            strokeWidth={1}
          />
        </g>
      ))}

      {/* Stones and markers */}
      {pattern.map((row, r) => {
        const chars = row.trim().split(' ').filter(c => c)
        return chars.map((char, c) => {
          const x = padding + c * cellSize
          const y = padding + r * cellSize

          if (char === 'B') {
            return <circle key={`${r}-${c}`} cx={x} cy={y} r={11} fill="#1a1a1a" stroke="#000" />
          } else if (char === 'W') {
            return <circle key={`${r}-${c}`} cx={x} cy={y} r={11} fill="#f5f5f5" stroke="#666" />
          } else if (char === 'H' || char === 'E' || char === 'C' || char === 'S') {
            // Highlighted move
            return (
              <g key={`${r}-${c}`}>
                <circle cx={x} cy={y} r={11} fill="#4CAF50" stroke="#2E7D32" strokeWidth={2} />
                <text x={x} y={y + 4} textAnchor="middle" fill="white" fontSize={12} fontWeight="bold">
                  {char === 'H' ? '!' : char === 'E' ? '→' : char === 'C' ? '+' : '↓'}
                </text>
              </g>
            )
          }
          return null
        })
      })}
    </svg>
  )
}

function Leaderboard({ onSelect }: { onSelect: (key: InstinctKey) => void }) {
  const sorted = Object.entries(INSTINCT_DATA).sort((a, b) => b[1].advantage - a[1].advantage)

  return (
    <div className="leaderboard">
      <h2>Instinct Leaderboard</h2>
      <p className="subtitle">Ranked by follow advantage (2000 Atari Go games)</p>

      <table>
        <thead>
          <tr>
            <th>#</th>
            <th>Instinct</th>
            <th>Advantage</th>
            <th>Fired</th>
            <th>Follow%</th>
            <th>Verdict</th>
          </tr>
        </thead>
        <tbody>
          {sorted.map(([key, data], i) => (
            <tr
              key={key}
              className={data.advantage > 5 ? 'champion' : data.advantage < 0 ? 'negative' : ''}
              onClick={() => onSelect(key as InstinctKey)}
            >
              <td className="rank">
                {i === 0 ? '🏆' : i + 1}
              </td>
              <td className="name">
                <span className="japanese">{data.japanese}</span>
                <span className="english">{data.name}</span>
              </td>
              <td className={`advantage ${data.advantage > 0 ? 'positive' : 'negative'}`}>
                {data.advantage > 0 ? '+' : ''}{data.advantage.toFixed(1)}%
              </td>
              <td className="fired">{data.fired.toLocaleString()}</td>
              <td className="follow-rate">{data.followRate.toFixed(1)}%</td>
              <td className="verdict">{data.verdict}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

function InstinctDetail({ instinctKey }: { instinctKey: InstinctKey }) {
  const data = INSTINCT_DATA[instinctKey]

  return (
    <div className="instinct-detail">
      <div className="detail-header">
        <h2>{data.japanese}</h2>
        <h3>{data.name}</h3>
        <p className="description">{data.description}</p>
      </div>

      <div className="detail-content">
        <div className="pattern-section">
          <h4>Pattern</h4>
          <PatternBoard pattern={data.pattern} />
          <p className="pattern-legend">
            <span className="legend-item"><span className="black-stone"></span> Black</span>
            <span className="legend-item"><span className="white-stone"></span> White</span>
            <span className="legend-item"><span className="move-marker"></span> Instinct move</span>
          </p>
        </div>

        <div className="stats-section">
          <h4>Statistics</h4>
          <div className="stat-grid">
            <div className="stat">
              <span className="stat-value">{data.advantage > 0 ? '+' : ''}{data.advantage.toFixed(1)}%</span>
              <span className="stat-label">Follow Advantage</span>
            </div>
            <div className="stat">
              <span className="stat-value">{data.fired.toLocaleString()}</span>
              <span className="stat-label">Times Fired</span>
            </div>
            <div className="stat">
              <span className="stat-value">{data.followRate.toFixed(1)}%</span>
              <span className="stat-label">Follow Rate</span>
            </div>
            <div className="stat">
              <span className="stat-value">#{data.rank}</span>
              <span className="stat-label">Rank</span>
            </div>
          </div>

          <div className="advantage-bar">
            <div
              className={`bar-fill ${data.advantage >= 0 ? 'positive' : 'negative'}`}
              style={{ width: `${Math.min(100, Math.abs(data.advantage) * 5)}%` }}
            />
            <span className="bar-label">
              {data.advantage >= 0 ? 'Following helps!' : 'May be situational'}
            </span>
          </div>
        </div>
      </div>

      <div className="insight">
        <h4>Insight</h4>
        <p>
          {data.advantage > 10
            ? `This instinct is a game-changer! Following ${data.name} leads to ${data.advantage.toFixed(1)}% higher win rate. When you see this pattern, respond immediately.`
            : data.advantage > 3
            ? `${data.name} consistently helps win games. The proverb is confirmed by data.`
            : data.advantage > 0
            ? `${data.name} has a slight positive effect. Good to follow when other factors are equal.`
            : `${data.name} showed negative results in Atari Go, but this may be because it's a strategic move that needs more time to pay off. In full Go, this proverb may still be valuable.`
          }
        </p>
      </div>
    </div>
  )
}

function ExperimentPanel() {
  const [running, setRunning] = useState(false)
  const [games, setGames] = useState(500)
  const [boardSize, setBoardSize] = useState(9)

  const runExperiment = () => {
    setRunning(true)
    // TODO: Connect to Python backend to run experiments
    setTimeout(() => setRunning(false), 3000)
  }

  return (
    <div className="experiment-panel">
      <h2>Run Experiment</h2>
      <p className="subtitle">Generate more data to refine instinct weights</p>

      <div className="controls">
        <label>
          Games:
          <input
            type="number"
            value={games}
            onChange={(e) => setGames(parseInt(e.target.value))}
            min={100}
            max={10000}
            step={100}
          />
        </label>

        <label>
          Board Size:
          <select value={boardSize} onChange={(e) => setBoardSize(parseInt(e.target.value))}>
            <option value={5}>5×5 (Fast)</option>
            <option value={7}>7×7</option>
            <option value={9}>9×9 (Standard)</option>
            <option value={13}>13×13</option>
          </select>
        </label>

        <button onClick={runExperiment} disabled={running}>
          {running ? 'Running...' : 'Start Experiment'}
        </button>
      </div>

      <div className="experiment-info">
        <p>
          Current data: <strong>2000 games</strong> on 9×9 Atari Go
        </p>
        <p>
          Estimated time: ~{Math.ceil(games / 300)} minutes
        </p>
      </div>
    </div>
  )
}

export function InstinctBattleground() {
  const [selectedInstinct, setSelectedInstinct] = useState<InstinctKey>('hane_vs_tsuke')

  return (
    <div className="instinct-battleground">
      <header>
        <h1>Instinct Battleground</h1>
        <p>Sensei's 8 Basic Instincts — Tested through 2000 Atari Go games</p>
      </header>

      <div className="battleground-layout">
        <div className="left-panel">
          <Leaderboard onSelect={setSelectedInstinct} />
          <ExperimentPanel />
        </div>

        <div className="right-panel">
          <InstinctDetail instinctKey={selectedInstinct} />
        </div>
      </div>

      <footer className="wisdom">
        <blockquote>
          "One experiment is a question, not an answer."
        </blockquote>
        <p>We reproduce. Vary. Let empiric tell if wisdom was right — but with time.</p>
      </footer>
    </div>
  )
}
