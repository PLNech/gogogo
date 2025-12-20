import { useState } from 'react'
import './InstinctBattleground.css'

// Instinct data from our 2000-game Atari Go experiment
// Patterns use numbered stones: even (0,2,4,6,8) = Black, odd (1,3,5,7,9) = White
// * marks the instinct move
const INSTINCT_DATA = {
  hane_vs_tsuke: {
    name: 'Hane vs Tsuke',
    japanese: 'ツケにはハネ',
    description: 'When opponent attaches to your stone, wrap around with hane',
    advantage: 13.2,
    fired: 54397,
    followRate: 19.7,
    rank: 1,
    // 0: Black stone, 1: White attaches, *: Black hane
    pattern: [
      '  . . . . .  ',
      '  . . . . .  ',
      '  . . 0 1 .  ',
      '  . . . * .  ',
      '  . . . . .  ',
    ],
    verdict: 'Champion',
  },
  extend_from_atari: {
    name: 'Extend from Atari',
    japanese: 'アタリから伸びよ',
    description: 'When your stone is in atari (1 liberty), extend to escape',
    advantage: 9.7,
    fired: 19337,
    followRate: 8.3,
    rank: 2,
    // 0: Black in center, 1,3,5,7: White surrounds → atari, *: extend
    pattern: [
      '  . . . . .  ',
      '  . 3 1 . .  ',
      '  . 5 0 7 .  ',
      '  . . * . .  ',
      '  . . . . .  ',
    ],
    verdict: 'Confirmed',
  },
  block_the_thrust: {
    name: 'Block the Thrust',
    japanese: 'ツキアタリには',
    description: 'Block when opponent thrusts into your stone formation',
    advantage: 9.6,
    fired: 59893,
    followRate: 13.3,
    rank: 3,
    // 0,2: Black formation, 3: White thrusts, *: Black blocks
    pattern: [
      '  . . . . .  ',
      '  . . * . .  ',
      '  . 0 3 . .  ',
      '  . 2 . . .  ',
      '  . . . . .  ',
    ],
    verdict: 'Confirmed',
  },
  block_the_angle: {
    name: 'Block the Angle',
    japanese: 'カケにはオサエ',
    description: 'Block when opponent makes knight\'s move approach',
    advantage: 3.5,
    fired: 64273,
    followRate: 22.5,
    rank: 4,
    // 0: Black stone, 1: White knight's move, *: Black blocks
    pattern: [
      '  . . . . .  ',
      '  . . . 0 .  ',
      '  . . * . .  ',
      '  . 1 . . .  ',
      '  . . . . .  ',
    ],
    verdict: 'Works',
  },
  connect_vs_peep: {
    name: 'Connect vs Peep',
    japanese: 'ノゾキにはツギ',
    description: '"Even a moron connects against a peep"',
    advantage: 3.4,
    fired: 65163,
    followRate: 16.2,
    rank: 5,
    // 0,2: Black stones with gap, 3: White peeps, *: Black connects
    pattern: [
      '  . . . . .  ',
      '  . 0 . 2 .  ',
      '  . . * . .  ',
      '  . . 3 . .  ',
      '  . . . . .  ',
    ],
    verdict: '"Even a moron"',
  },
  stretch_from_bump: {
    name: 'Stretch from Bump',
    japanese: 'ブツカリから伸びよ',
    description: 'When bumping opponent who has support, stretch (don\'t hane)',
    advantage: 3.2,
    fired: 52845,
    followRate: 11.5,
    rank: 6,
    // 1: White support, 3: White main, 4: Black bumps, *: stretch
    pattern: [
      '  . . . . .  ',
      '  . . * . .  ',
      '  . 1 4 . .  ',
      '  . . 3 . .  ',
      '  . . . . .  ',
    ],
    verdict: 'Slight positive',
  },
  stretch_from_kosumi: {
    name: 'Stretch from Kosumi',
    japanese: 'コスミから伸びよ',
    description: 'Stretch away from opponent\'s diagonal contact',
    advantage: 3.0,
    fired: 79665,
    followRate: 48.2,
    rank: 7,
    // 0: Black stone, 1: White diagonal contact, *: Black stretches
    pattern: [
      '  . . . . .  ',
      '  . . 1 . .  ',
      '  . . . 0 .  ',
      '  . . . * .  ',
      '  . . . . .  ',
    ],
    verdict: 'Slight positive',
  },
  hane_at_head_of_two: {
    name: 'Hane at Head of Two',
    japanese: '二子の頭にハネ',
    description: '2v2 confrontation: play at the head of opponent\'s two stones',
    advantage: 1.9,
    fired: 45171,
    followRate: 8.4,
    rank: 8,
    // 0,2: Black, 1,3: White → 2v2 forms, *: Black at head
    pattern: [
      '  . . . . .  ',
      '  . . * . .  ',
      '  . 0 1 . .  ',
      '  . 2 3 . .  ',
      '  . . . . .  ',
    ],
    verdict: 'Strategic',
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

          // Numbered stones: even (0,2,4,6,8) = Black, odd (1,3,5,7,9) = White
          const num = parseInt(char)
          if (!isNaN(num)) {
            const isBlack = num % 2 === 0
            return (
              <g key={`${r}-${c}`}>
                <circle
                  cx={x} cy={y} r={11}
                  fill={isBlack ? "#1a1a1a" : "#f5f5f5"}
                  stroke={isBlack ? "#000" : "#666"}
                />
                <text
                  x={x} y={y + 4}
                  textAnchor="middle"
                  fill={isBlack ? "#fff" : "#333"}
                  fontSize={10}
                  fontWeight="bold"
                >
                  {num}
                </text>
              </g>
            )
          } else if (char === '*') {
            // Instinct move (always Black response)
            return (
              <g key={`${r}-${c}`}>
                <circle cx={x} cy={y} r={11} fill="#4CAF50" stroke="#2E7D32" strokeWidth={2} />
                <text x={x} y={y + 4} textAnchor="middle" fill="white" fontSize={12} fontWeight="bold">
                  ✦
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

function Leaderboard({ onSelect, selected }: { onSelect: (key: InstinctKey) => void, selected: InstinctKey }) {
  const sorted = Object.entries(INSTINCT_DATA).sort((a, b) => b[1].advantage - a[1].advantage)

  return (
    <div className="leaderboard">
      <h2>Instinct Leaderboard</h2>
      <p className="subtitle">2000 Atari Go games · Ranked by follow advantage</p>

      <table>
        <thead>
          <tr>
            <th>#</th>
            <th>Instinct</th>
            <th>Adv.</th>
            <th>Fired</th>
            <th>Verdict</th>
          </tr>
        </thead>
        <tbody>
          {sorted.map(([key, data], i) => (
            <tr
              key={key}
              className={`
                ${data.advantage > 5 ? 'champion' : data.advantage < 0 ? 'negative' : ''}
                ${key === selected ? 'selected' : ''}
              `}
              onClick={() => onSelect(key as InstinctKey)}
            >
              <td className="rank">
                {i === 0 ? '🏆' : i + 1}
              </td>
              <td className="name">
                <span className="english">{data.name}</span>
                <span className="japanese">{data.japanese}</span>
              </td>
              <td className={`advantage ${data.advantage > 0 ? 'positive' : 'negative'}`}>
                {data.advantage > 0 ? '+' : ''}{data.advantage.toFixed(1)}%
              </td>
              <td className="fired">{(data.fired / 1000).toFixed(0)}k</td>
              <td className="verdict">{data.verdict}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

// Get advantage bar label based on level
function getAdvantageLabel(adv: number): string {
  if (adv > 10) return "Game-changer! Always follow."
  if (adv > 5) return "Strong advantage. Follow this."
  if (adv > 2) return "Solid positive. Good default."
  if (adv > 0) return "Slight edge. Consider context."
  if (adv > -2) return "Neutral to slight negative."
  return "Strategic — needs more moves."
}

// Get verdict color class
function getVerdictClass(adv: number): string {
  if (adv > 5) return 'verdict-champion'
  if (adv > 2) return 'verdict-confirmed'
  if (adv > 0) return 'verdict-positive'
  return 'verdict-strategic'
}

function InstinctDetail({ instinctKey }: { instinctKey: InstinctKey }) {
  const data = INSTINCT_DATA[instinctKey]

  return (
    <div className="instinct-detail">
      <div className="detail-header">
        <h2>{data.name}</h2>
        <h3>{data.japanese}</h3>
        <p className="description">{data.description}</p>
        <span className={`verdict-badge ${getVerdictClass(data.advantage)}`}>
          {data.verdict}
        </span>
      </div>

      <div className="detail-content">
        <div className="pattern-section">
          <h4>Pattern</h4>
          <PatternBoard pattern={data.pattern} />
          <p className="pattern-legend">
            <span className="legend-item"><span className="black-stone"></span> Black (0,2,4...)</span>
            <span className="legend-item"><span className="white-stone"></span> White (1,3,5...)</span>
            <span className="legend-item"><span className="move-marker"></span> Instinct</span>
          </p>
        </div>

        <div className="stats-section">
          <h4>Statistics</h4>
          <div className="stat-grid">
            <div className="stat">
              <span className={`stat-value ${data.advantage > 0 ? 'positive' : 'negative'}`}>
                {data.advantage > 0 ? '+' : ''}{data.advantage.toFixed(1)}%
              </span>
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
              style={{ width: `${Math.min(100, Math.abs(data.advantage) * 8)}%` }}
            />
            <span className="bar-label">
              {getAdvantageLabel(data.advantage)}
            </span>
          </div>
        </div>
      </div>

      <div className={`insight ${getVerdictClass(data.advantage)}`}>
        <h4>Insight</h4>
        <p>
          {data.advantage > 10
            ? <><strong>This instinct is a game-changer!</strong> Following {data.name} leads to {data.advantage.toFixed(1)}% higher win rate. When you see this pattern, respond immediately.</>
            : data.advantage > 3
            ? <><strong>Confirmed by data.</strong> {data.name} consistently helps win games. The proverb holds true.</>
            : data.advantage > 0
            ? <><strong>Slight positive effect.</strong> {data.name} helps when other factors are equal. Not the highest priority, but sound.</>
            : <><strong>Strategic move.</strong> {data.name} showed negative results in Atari Go, but this is a short-term game. Strategic moves that build influence need more moves to pay off. In full Go, this proverb remains valuable.</>
          }
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
          <Leaderboard onSelect={setSelectedInstinct} selected={selectedInstinct} />
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
