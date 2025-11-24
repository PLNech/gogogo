import type { Board, Position } from '../go/types'
import { getStone, countTerritory, getGroup, countLiberties } from '../go/board'

/**
 * Estimate the value of a position for influence/territory
 */
export function evaluateInfluence(board: Board, pos: Position, player: 'black' | 'white'): number {
  let influence = 0
  const opponent: 'black' | 'white' = player === 'black' ? 'white' : 'black'

  // Distance to friendly stones increases influence
  // Distance to enemy stones decreases influence
  for (let row = 0; row < board.size; row++) {
    for (let col = 0; col < board.size; col++) {
      const stone = getStone(board, row, col)
      if (!stone) continue

      const distance = Math.abs(pos.row - row) + Math.abs(pos.col - col)
      if (distance === 0) continue

      const weight = 1 / (distance + 1)

      if (stone === player) {
        influence += weight * 2
      } else {
        influence -= weight * 2
      }
    }
  }

  return influence
}

/**
 * Evaluate territorial security of a position
 */
export function evaluateTerritorialSecurity(board: Board, pos: Position, player: 'black' | 'white'): number {
  // Check if position is in a corner or edge (more secure)
  const distToEdge = Math.min(pos.row, pos.col, board.size - 1 - pos.row, board.size - 1 - pos.col)
  let security = 0

  // Corners and edges are valuable for territory
  if (distToEdge === 0) security += 15 // Edge
  if (distToEdge === 1) security += 10 // Near edge
  if ((pos.row === 0 || pos.row === board.size - 1) && (pos.col === 0 || pos.col === board.size - 1)) {
    security += 25 // Corner
  }

  // Check for friendly nearby stones (more secure)
  let friendlyNearby = 0
  let enemyNearby = 0
  const opponent: 'black' | 'white' = player === 'black' ? 'white' : 'black'

  for (let dr = -2; dr <= 2; dr++) {
    for (let dc = -2; dc <= 2; dc++) {
      if (dr === 0 && dc === 0) continue
      const r = pos.row + dr
      const c = pos.col + dc
      if (r < 0 || r >= board.size || c < 0 || c >= board.size) continue

      const stone = getStone(board, r, c)
      if (stone === player) friendlyNearby++
      if (stone === opponent) enemyNearby++
    }
  }

  security += friendlyNearby * 3
  security -= enemyNearby * 3

  return security
}

/**
 * Estimate current game score from player's perspective
 */
export function estimateScore(board: Board, captures: { black: number; white: number }, player: 'black' | 'white'): number {
  const territory = countTerritory(board)

  const blackScore = territory.black + captures.black
  const whiteScore = territory.white + captures.white

  // Return score difference from player's perspective
  if (player === 'black') {
    return blackScore - whiteScore
  } else {
    return whiteScore - blackScore
  }
}

/**
 * Evaluate if a stone group is alive or in danger
 */
export function evaluateGroupHealth(board: Board, group: Position[]): number {
  if (group.length === 0) return 0

  const liberties = countLiberties(board, group)

  // Dead group
  if (liberties === 0) return -100

  // In atari (danger)
  if (liberties === 1) return -50

  // Somewhat safe
  if (liberties === 2) return 0

  // Safe
  if (liberties === 3) return 20

  // Very safe
  return 40
}

/**
 * Check if a position is filling own territory (wasteful move)
 * This is a critical check - playing inside your own secure area is almost always wrong
 */
export function isFillingOwnTerritory(board: Board, pos: Position, player: 'black' | 'white'): boolean {
  const opponent: 'black' | 'white' = player === 'black' ? 'white' : 'black'

  // Check immediate adjacents
  const adjacents = [
    { row: pos.row - 1, col: pos.col },
    { row: pos.row + 1, col: pos.col },
    { row: pos.row, col: pos.col - 1 },
    { row: pos.row, col: pos.col + 1 },
  ]

  let friendlyAdjacent = 0
  let opponentAdjacent = 0
  let emptyAdjacent = 0

  for (const adj of adjacents) {
    if (adj.row < 0 || adj.row >= board.size || adj.col < 0 || adj.col >= board.size) {
      friendlyAdjacent++ // Treat edge as friendly for territory
      continue
    }

    const stone = getStone(board, adj.row, adj.col)
    if (stone === player) {
      friendlyAdjacent++
    } else if (stone === opponent) {
      opponentAdjacent++
    } else {
      emptyAdjacent++
    }
  }

  // Immediate check: If ANY opponent stones adjacent, not filling own territory
  if (opponentAdjacent > 0) {
    return false
  }

  // Check 1: Completely surrounded by friendly (obviously filling)
  if (friendlyAdjacent === 4) {
    return true
  }

  // Check 2: Mostly surrounded (3 friendly, 1 empty)
  if (friendlyAdjacent >= 3 && emptyAdjacent <= 1) {
    // Additional check: is the empty adjacent also surrounded by friendly?
    if (emptyAdjacent === 1) {
      for (const adj of adjacents) {
        if (adj.row < 0 || adj.row >= board.size || adj.col < 0 || adj.col >= board.size) continue
        if (getStone(board, adj.row, adj.col) === null) {
          // Check if this empty space is also enclosed
          const emptyNeighbors = [
            { row: adj.row - 1, col: adj.col },
            { row: adj.row + 1, col: adj.col },
            { row: adj.row, col: adj.col - 1 },
            { row: adj.row, col: adj.col + 1 },
          ]
          let emptyFriendly = 0
          for (const en of emptyNeighbors) {
            if (en.row < 0 || en.row >= board.size || en.col < 0 || en.col >= board.size) {
              emptyFriendly++
              continue
            }
            if (getStone(board, en.row, en.col) === player) {
              emptyFriendly++
            }
          }
          // If the empty space is also mostly enclosed, we're deep in territory
          if (emptyFriendly >= 3) {
            return true
          }
        }
      }
    }
    return true
  }

  // Check 3: Look at 2-space radius - are we deep inside friendly area?
  // Only apply this check on larger boards (7x7+) to avoid false positives on small boards
  if (board.size >= 7) {
    let nearbyFriendly = 0
    let nearbyOpponent = 0

    for (let dr = -2; dr <= 2; dr++) {
      for (let dc = -2; dc <= 2; dc++) {
        if (dr === 0 && dc === 0) continue
        const r = pos.row + dr
        const c = pos.col + dc
        if (r < 0 || r >= board.size || c < 0 || c >= board.size) {
          nearbyFriendly++ // Edges count as friendly
          continue
        }

        const stone = getStone(board, r, c)
        if (stone === player) {
          nearbyFriendly++
        } else if (stone === opponent) {
          nearbyOpponent++
        }
      }
    }

    // If we see a lot of friendly stones and no opponent stones nearby, probably filling territory
    if (nearbyFriendly >= 12 && nearbyOpponent === 0) {
      return true
    }
  }

  return false
}

/**
 * Check if position is on the boundary between territories (good place to play)
 */
export function isOnBoundary(board: Board, pos: Position, player: 'black' | 'white'): boolean {
  const opponent: 'black' | 'white' = player === 'black' ? 'white' : 'black'

  const adjacents = [
    { row: pos.row - 1, col: pos.col },
    { row: pos.row + 1, col: pos.col },
    { row: pos.row, col: pos.col - 1 },
    { row: pos.row, col: pos.col + 1 },
  ]

  let hasFriendly = false
  let hasOpponent = false

  for (const adj of adjacents) {
    if (adj.row < 0 || adj.row >= board.size || adj.col < 0 || adj.col >= board.size) continue

    const stone = getStone(board, adj.row, adj.col)
    if (stone === player) hasFriendly = true
    if (stone === opponent) hasOpponent = true
  }

  // On boundary if we have both friendly and opponent nearby
  return hasFriendly && hasOpponent
}

/**
 * Evaluate connection potential - prefer moves that connect groups
 */
export function evaluateConnection(board: Board, pos: Position, player: 'black' | 'white'): number {
  let connectionValue = 0
  const nearbyGroups = new Set<string>()

  // Check 2-space knight moves and adjacent for friendly stones
  for (let dr = -2; dr <= 2; dr++) {
    for (let dc = -2; dc <= 2; dc++) {
      if (dr === 0 && dc === 0) continue
      const r = pos.row + dr
      const c = pos.col + dc
      if (r < 0 || r >= board.size || c < 0 || c >= board.size) continue

      const stone = getStone(board, r, c)
      if (stone === player) {
        // Find which group this belongs to
        const group = getGroup(board, r, c)
        const groupKey = group.map(p => `${p.row},${p.col}`).sort().join('|')
        nearbyGroups.add(groupKey)
      }
    }
  }

  // More nearby groups = better connection potential
  if (nearbyGroups.size >= 2) {
    connectionValue = 15 * nearbyGroups.size
  }

  return connectionValue
}

/**
 * Check if position creates overly dense/heavy formations (bad Go)
 * Master players avoid playing in areas with too many friendly stones
 */
export function isOverlyDense(board: Board, pos: Position, player: 'black' | 'white', level: number): boolean {
  if (level < 4) return false // Only check for advanced/master levels

  // Count friendly stones in immediate area (2-space radius)
  let friendlyCount = 0
  let totalSpaces = 0

  for (let dr = -2; dr <= 2; dr++) {
    for (let dc = -2; dc <= 2; dc++) {
      if (dr === 0 && dc === 0) continue
      const r = pos.row + dr
      const c = pos.col + dc
      if (r < 0 || r >= board.size || c < 0 || c >= board.size) continue

      totalSpaces++
      const stone = getStone(board, r, c)
      if (stone === player) {
        friendlyCount++
      }
    }
  }

  // If more than 60% of nearby spaces are friendly, it's too dense
  const density = friendlyCount / totalSpaces

  // Level 5: very strict (50%), Level 4: moderate (60%)
  const threshold = level >= 5 ? 0.5 : 0.6

  return density > threshold
}

/**
 * Evaluate corner and wall (edge) control value
 * Level progression:
 * - Level 1: Doesn't understand (returns 0)
 * - Level 2: Learns corners/walls are good (moderate bonus)
 * - Level 3-4: Maximize corner/wall control (high bonus)
 * - Level 5: Knows when to ignore (moderate bonus, masters know flexibility)
 */
export function evaluateCornerWallControl(board: Board, pos: Position, player: 'black' | 'white', level: number): number {
  if (level < 2) return 0 // Level 1 doesn't understand corner/wall value

  let controlValue = 0

  // Calculate distance to edges
  const distToTop = pos.row
  const distToBottom = board.size - 1 - pos.row
  const distToLeft = pos.col
  const distToRight = board.size - 1 - pos.col
  const minDistToEdge = Math.min(distToTop, distToBottom, distToLeft, distToRight)

  // Scale corner/wall detection with board size
  // 9x9: 3 spaces, 13x13: 4 spaces, 19x19: 6 spaces
  const cornerRadius = board.size <= 9 ? 3 : board.size <= 13 ? 4 : 6
  const wallRadius = board.size <= 9 ? 2 : board.size <= 13 ? 3 : 4

  // Check if in corner (both dimensions close to edge)
  const isInCorner = (
    (distToTop <= cornerRadius || distToBottom <= cornerRadius) &&
    (distToLeft <= cornerRadius || distToRight <= cornerRadius)
  )

  // Check if on wall/edge (close to one edge but not corner)
  const isOnWall = (
    (minDistToEdge <= wallRadius) && !isInCorner
  )

  // Level-based scoring
  if (isInCorner) {
    // Corners are most valuable
    if (level === 2) {
      controlValue = 15 // Learning corners are good
    } else if (level === 3) {
      controlValue = 25 // Good understanding
    } else if (level === 4) {
      controlValue = 40 // Maximize corner control
    } else if (level === 5) {
      controlValue = 30 // Masters know corners are good but not always best
    }

    // Bonus for actual corner points (3-3, 3-4, 4-4 positions)
    if (minDistToEdge >= 2 && minDistToEdge <= 3) {
      controlValue += level >= 4 ? 15 : 10
    }
  } else if (isOnWall) {
    // Walls/edges are valuable but less than corners
    if (level === 2) {
      controlValue = 10 // Learning walls are good
    } else if (level === 3) {
      controlValue = 18 // Good understanding
    } else if (level === 4) {
      controlValue = 30 // Maximize wall control
    } else if (level === 5) {
      controlValue = 20 // Masters know walls are good but flexible
    }
  }

  return controlValue
}

/**
 * Find all opponent groups and their health status
 * Returns information about weak groups that could be attacked
 */
export interface OpponentGroupInfo {
  group: Position[]
  liberties: number
  health: 'dead' | 'atari' | 'weak' | 'safe' | 'very-safe'
  attackPoints: Position[] // Points where you can attack this group
}

export function findOpponentGroups(board: Board, player: 'black' | 'white'): OpponentGroupInfo[] {
  const opponent: 'black' | 'white' = player === 'black' ? 'white' : 'black'
  const processedPositions = new Set<string>()
  const groups: OpponentGroupInfo[] = []

  for (let row = 0; row < board.size; row++) {
    for (let col = 0; col < board.size; col++) {
      const stone = getStone(board, row, col)
      if (stone !== opponent) continue

      const posKey = `${row},${col}`
      if (processedPositions.has(posKey)) continue

      // Found a new opponent group
      const group = getGroup(board, row, col)
      const liberties = countLiberties(board, group)

      // Mark all positions in this group as processed
      for (const pos of group) {
        processedPositions.add(`${pos.row},${pos.col}`)
      }

      // Find attack points (liberties of this group)
      const attackPoints: Position[] = []
      const libertiesSet = new Set<string>()

      for (const pos of group) {
        const adjacents = [
          { row: pos.row - 1, col: pos.col },
          { row: pos.row + 1, col: pos.col },
          { row: pos.row, col: pos.col - 1 },
          { row: pos.row, col: pos.col + 1 },
        ]

        for (const adj of adjacents) {
          if (adj.row < 0 || adj.row >= board.size || adj.col < 0 || adj.col >= board.size) continue
          if (getStone(board, adj.row, adj.col) === null) {
            const key = `${adj.row},${adj.col}`
            if (!libertiesSet.has(key)) {
              libertiesSet.add(key)
              attackPoints.push(adj)
            }
          }
        }
      }

      // Classify health
      let health: 'dead' | 'atari' | 'weak' | 'safe' | 'very-safe'
      if (liberties === 0) health = 'dead'
      else if (liberties === 1) health = 'atari'
      else if (liberties === 2) health = 'weak'
      else if (liberties === 3) health = 'safe'
      else health = 'very-safe'

      groups.push({
        group,
        liberties,
        health,
        attackPoints,
      })
    }
  }

  return groups
}

/**
 * Evaluate the value of attacking an opponent group
 * Returns score bonus for attacking based on AI level and strategic value
 */
export function evaluateAttackValue(
  board: Board,
  groupInfo: OpponentGroupInfo,
  player: 'black' | 'white',
  level: number
): number {
  const opponent: 'black' | 'white' = player === 'black' ? 'white' : 'black'

  // Level 1-2: Super aggressive, attack everything weak (tactics-focused)
  if (level <= 2) {
    if (groupInfo.health === 'atari') return 80 // MUST capture!
    if (groupInfo.health === 'weak') return 60 // Attack weak groups aggressively
    return 0
  }

  // Level 3: Transitioning from tactics to strategy
  if (level === 3) {
    if (groupInfo.health === 'atari') return 70 // Still very aggressive on atari
    if (groupInfo.health === 'weak') return 40 // More selective on weak groups
    return 0
  }

  // Level 4-5: Strategic thinking - only attack if it's beneficial
  // Consider: Is killing this group actually good for territory?
  // Or should we just threaten and build territory elsewhere?

  if (groupInfo.health === 'atari') {
    // Even masters capture groups in atari, but check if it's worth it
    const groupSize = groupInfo.group.length

    // Small groups (1-2 stones) might not be worth capturing in late game
    if (groupSize <= 2 && level === 5) {
      return 40 // Still capture, but lower priority
    }

    return 60 // Generally capture atari groups
  }

  if (groupInfo.health === 'weak') {
    const groupSize = groupInfo.group.length

    // Level 5: Very strategic - might ignore weak groups to play elsewhere
    if (level === 5) {
      // Only attack weak groups if they're big (more than 4 stones)
      if (groupSize >= 4) {
        return 30 // Worth attacking for the capture value
      }
      return 0 // Ignore small weak groups, play territory instead
    }

    // Level 4: Somewhat strategic
    if (groupSize >= 3) {
      return 35
    }
    return 10 // Small bonus, but might ignore
  }

  return 0 // Safe groups don't get attack bonus
}

/**
 * Evaluate invasion/reduction opportunities in opponent territory
 * Master level players (4-5) actively invade and reduce opponent influence
 */
export function evaluateInvasionReduction(board: Board, pos: Position, player: 'black' | 'white', level: number): number {
  if (level < 4) return 0 // Only level 4-5 actively invade

  const opponent: 'black' | 'white' = player === 'black' ? 'white' : 'black'

  // Count opponent vs friendly stones in 3-space radius
  let opponentCount = 0
  let friendlyCount = 0

  for (let dr = -3; dr <= 3; dr++) {
    for (let dc = -3; dc <= 3; dc++) {
      if (dr === 0 && dc === 0) continue
      const r = pos.row + dr
      const c = pos.col + dc
      if (r < 0 || r >= board.size || c < 0 || c >= board.size) continue

      const stone = getStone(board, r, c)
      if (stone === opponent) opponentCount++
      else if (stone === player) friendlyCount++
    }
  }

  // This is opponent-influenced territory if:
  // - More opponent stones than friendly
  // - At least 3 opponent stones nearby
  if (opponentCount >= 3 && opponentCount > friendlyCount * 2) {
    // Invasion opportunity!
    const invasionValue = opponentCount * (level >= 5 ? 15 : 10)

    // Check if on edge/corner (more valuable invasion point)
    const distToEdge = Math.min(
      pos.row,
      pos.col,
      board.size - 1 - pos.row,
      board.size - 1 - pos.col
    )

    // Corner/edge invasions are more effective
    if (distToEdge <= 2) {
      return invasionValue * 1.5
    }

    return invasionValue
  }

  return 0
}

/**
 * Evaluate if position contests/reduces opponent framework
 * Level 4-5: Actively reduce opponent moyos (frameworks)
 */
export function evaluateReduction(board: Board, pos: Position, player: 'black' | 'white', level: number): number {
  if (level < 4) return 0

  const opponent: 'black' | 'white' = player === 'black' ? 'white' : 'black'

  // Check if we're in a large empty area with opponent stones at edges
  let nearbyEmpty = 0
  let opponentAtEdge = 0

  // Check 2-space radius for emptiness
  for (let dr = -2; dr <= 2; dr++) {
    for (let dc = -2; dc <= 2; dc++) {
      const r = pos.row + dr
      const c = pos.col + dc
      if (r < 0 || r >= board.size || c < 0 || c >= board.size) continue

      const stone = getStone(board, r, c)
      if (stone === null) {
        nearbyEmpty++
      }
    }
  }

  // Check 4-space radius for opponent stones (framework edge)
  for (let dr = -4; dr <= 4; dr++) {
    for (let dc = -4; dc <= 4; dc++) {
      if (Math.abs(dr) < 3 && Math.abs(dc) < 3) continue // Skip inner area
      const r = pos.row + dr
      const c = pos.col + dc
      if (r < 0 || r >= board.size || c < 0 || c >= board.size) continue

      const stone = getStone(board, r, c)
      if (stone === opponent) {
        opponentAtEdge++
      }
    }
  }

  // If lots of empty space with opponent stones at edges, this is a reduction point
  if (nearbyEmpty >= 15 && opponentAtEdge >= 4) {
    return level >= 5 ? 40 : 25 // Master level values reduction highly
  }

  return 0
}

/**
 * Count eyes (secure territory) for a group - simplified heuristic
 */
export function countEyes(board: Board, group: Position[]): number {
  if (group.length === 0) return 0

  const stone = getStone(board, group[0]!.row, group[0]!.col)
  if (!stone) return 0

  // Find empty spaces surrounded by this group
  const groupSet = new Set(group.map(p => `${p.row},${p.col}`))
  const potentialEyes = new Set<string>()

  for (const pos of group) {
    const adjacents = [
      { row: pos.row - 1, col: pos.col },
      { row: pos.row + 1, col: pos.col },
      { row: pos.row, col: pos.col - 1 },
      { row: pos.row, col: pos.col + 1 },
    ]

    for (const adj of adjacents) {
      if (adj.row < 0 || adj.row >= board.size || adj.col < 0 || adj.col >= board.size) continue

      if (getStone(board, adj.row, adj.col) === null) {
        potentialEyes.add(`${adj.row},${adj.col}`)
      }
    }
  }

  // Simplified: each empty space could be an eye
  // In reality, need to check if fully surrounded
  let eyes = 0
  for (const eyeKey of potentialEyes) {
    const [row, col] = eyeKey.split(',').map(Number)
    const adjacents = [
      { row: row! - 1, col: col! },
      { row: row! + 1, col: col! },
      { row: row!, col: col! - 1 },
      { row: row!, col: col! + 1 },
    ]

    let surroundedByGroup = true
    for (const adj of adjacents) {
      if (adj.row < 0 || adj.row >= board.size || adj.col < 0 || adj.col >= board.size) continue

      const adjStone = getStone(board, adj.row, adj.col)
      if (adjStone !== stone && adjStone !== null) {
        surroundedByGroup = false
        break
      }
    }

    if (surroundedByGroup) eyes++
  }

  return eyes
}
