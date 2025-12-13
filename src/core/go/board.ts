import type { Board, Stone, Position } from './types'

export function createBoard(size: number): Board {
  const stones: Stone[][] = []
  for (let i = 0; i < size; i++) {
    stones[i] = []
    for (let j = 0; j < size; j++) {
      stones[i]![j] = null
    }
  }
  return { size, stones }
}

export function getStone(board: Board, row: number, col: number): Stone {
  if (row < 0 || row >= board.size || col < 0 || col >= board.size) {
    return null
  }
  return board.stones[row]?.[col] ?? null
}

export function placeStone(
  board: Board,
  row: number,
  col: number,
  stone: 'black' | 'white',
  previousBoard?: Board
): Board | null {
  // Validate position
  if (row < 0 || row >= board.size || col < 0 || col >= board.size) {
    return null
  }

  // Check if position is occupied
  if (getStone(board, row, col) !== null) {
    return null
  }

  // Create new board (immutable)
  const newStones = board.stones.map(row => [...row])
  newStones[row]![col] = stone

  const newBoard = { ...board, stones: newStones }

  // Apply captures (opponent groups first)
  const { board: boardAfterCaptures, captured } = captureStones(newBoard, row, col, stone)

  // Suicide check: if placed stone's group has no liberties, it's illegal
  // (unless it captured something - then it has liberties from the capture)
  const placedGroup = getGroup(boardAfterCaptures, row, col)
  if (countLiberties(boardAfterCaptures, placedGroup) === 0) {
    return null // Suicide - illegal move
  }

  // Ko rule: check if this move would return to the previous board state
  if (previousBoard && boardsEqual(boardAfterCaptures, previousBoard)) {
    return null // Ko violation
  }

  return boardAfterCaptures
}

export function captureStones(
  board: Board,
  lastRow: number,
  lastCol: number,
  lastStone: 'black' | 'white'
): { board: Board; captured: number } {
  const opponent: Stone = lastStone === 'black' ? 'white' : 'black'
  let captured = 0
  let newBoard = board

  // Check all four adjacent positions for opponent groups
  const adjacents: Position[] = [
    { row: lastRow - 1, col: lastCol },
    { row: lastRow + 1, col: lastCol },
    { row: lastRow, col: lastCol - 1 },
    { row: lastRow, col: lastCol + 1 },
  ]

  for (const pos of adjacents) {
    if (getStone(newBoard, pos.row, pos.col) === opponent) {
      const group = getGroup(newBoard, pos.row, pos.col)
      if (countLiberties(newBoard, group) === 0) {
        // Capture this group
        captured += group.length
        newBoard = removeStones(newBoard, group)
      }
    }
  }

  return { board: newBoard, captured }
}

export function getGroup(board: Board, row: number, col: number): Position[] {
  const stone = getStone(board, row, col)
  if (stone === null) return []

  const group: Position[] = []
  const visited = new Set<string>()

  function dfs(r: number, c: number) {
    // Bounds check
    if (r < 0 || r >= board.size || c < 0 || c >= board.size) return

    const key = `${r},${c}`
    if (visited.has(key)) return
    if (getStone(board, r, c) !== stone) return

    visited.add(key)
    group.push({ row: r, col: c })

    dfs(r - 1, c)
    dfs(r + 1, c)
    dfs(r, c - 1)
    dfs(r, c + 1)
  }

  dfs(row, col)
  return group
}

export function countLiberties(board: Board, group: Position[]): number {
  const liberties = new Set<string>()

  for (const pos of group) {
    const adjacents: Position[] = [
      { row: pos.row - 1, col: pos.col },
      { row: pos.row + 1, col: pos.col },
      { row: pos.row, col: pos.col - 1 },
      { row: pos.row, col: pos.col + 1 },
    ]

    for (const adj of adjacents) {
      // Check bounds first
      if (adj.row < 0 || adj.row >= board.size || adj.col < 0 || adj.col >= board.size) {
        continue
      }
      if (getStone(board, adj.row, adj.col) === null) {
        liberties.add(`${adj.row},${adj.col}`)
      }
    }
  }

  return liberties.size
}

function removeStones(board: Board, positions: Position[]): Board {
  const newStones = board.stones.map(row => [...row])

  for (const pos of positions) {
    newStones[pos.row]![pos.col] = null
  }

  return { ...board, stones: newStones }
}

export function countTerritory(board: Board): {
  black: number
  white: number
  neutral: number
} {
  const visited = new Set<string>()
  let black = 0
  let white = 0
  let neutral = 0

  for (let row = 0; row < board.size; row++) {
    for (let col = 0; col < board.size; col++) {
      const key = `${row},${col}`
      if (visited.has(key)) continue

      const stone = getStone(board, row, col)
      if (stone !== null) {
        visited.add(key)
        continue
      }

      // Find empty region
      const region = getEmptyRegion(board, row, col)
      region.forEach(pos => visited.add(`${pos.row},${pos.col}`))

      // Determine who owns this region
      const owner = getRegionOwner(board, region)
      if (owner === 'black') {
        black += region.length
      } else if (owner === 'white') {
        white += region.length
      } else {
        neutral += region.length
      }
    }
  }

  return { black, white, neutral }
}

export function getTerritoryMap(board: Board): Map<string, Stone> {
  const territoryMap = new Map<string, Stone>()
  const visited = new Set<string>()

  for (let row = 0; row < board.size; row++) {
    for (let col = 0; col < board.size; col++) {
      const key = `${row},${col}`
      if (visited.has(key)) continue

      const stone = getStone(board, row, col)
      if (stone !== null) {
        visited.add(key)
        continue
      }

      // Find empty region
      const region = getEmptyRegion(board, row, col)
      region.forEach(pos => visited.add(`${pos.row},${pos.col}`))

      // Determine who owns this region
      const owner = getRegionOwner(board, region)

      // Mark all positions in region with owner
      region.forEach(pos => {
        territoryMap.set(`${pos.row},${pos.col}`, owner)
      })
    }
  }

  return territoryMap
}

function getEmptyRegion(board: Board, row: number, col: number): Position[] {
  const region: Position[] = []
  const visited = new Set<string>()

  function dfs(r: number, c: number) {
    // Bounds check
    if (r < 0 || r >= board.size || c < 0 || c >= board.size) return

    const key = `${r},${c}`
    if (visited.has(key)) return
    if (getStone(board, r, c) !== null) return

    visited.add(key)
    region.push({ row: r, col: c })

    dfs(r - 1, c)
    dfs(r + 1, c)
    dfs(r, c - 1)
    dfs(r, c + 1)
  }

  dfs(row, col)
  return region
}

function getRegionOwner(board: Board, region: Position[]): Stone {
  const adjacentStones = new Set<Stone>()

  for (const pos of region) {
    const adjacents: Position[] = [
      { row: pos.row - 1, col: pos.col },
      { row: pos.row + 1, col: pos.col },
      { row: pos.row, col: pos.col - 1 },
      { row: pos.row, col: pos.col + 1 },
    ]

    for (const adj of adjacents) {
      const stone = getStone(board, adj.row, adj.col)
      if (stone !== null) {
        adjacentStones.add(stone)
      }
    }
  }

  // If only one color borders this region, that color owns it
  if (adjacentStones.size === 1) {
    return adjacentStones.has('black') ? 'black' : 'white'
  }

  // Otherwise it's neutral (or empty board)
  return null
}

/**
 * Check if two boards are equal (for Ko rule detection)
 */
export function boardsEqual(board1: Board, board2: Board): boolean {
  if (board1.size !== board2.size) return false

  for (let row = 0; row < board1.size; row++) {
    for (let col = 0; col < board1.size; col++) {
      if (getStone(board1, row, col) !== getStone(board2, row, col)) {
        return false
      }
    }
  }

  return true
}

/**
 * Get adjacent positions (up, down, left, right)
 * Only returns valid positions within board bounds
 */
export function getAdjacentPositions(board: Board, row: number, col: number): Position[] {
  const adjacents: Position[] = [
    { row: row - 1, col },
    { row: row + 1, col },
    { row, col: col - 1 },
    { row, col: col + 1 }
  ]

  return adjacents.filter(
    p => p.row >= 0 && p.row < board.size && p.col >= 0 && p.col < board.size
  )
}

/**
 * Check if a position is a true eye for the given color.
 * True eye: empty point where all adjacent positions are same color
 * and at least 3/4 diagonals are controlled.
 */
export function hasEye(board: Board, row: number, col: number, color: Stone): boolean {
  if (getStone(board, row, col) !== null) return false
  if (color === null) return false

  // Check all adjacent positions are same color
  const adjacents = getAdjacentPositions(board, row, col)
  for (const adj of adjacents) {
    if (getStone(board, adj.row, adj.col) !== color) return false
  }

  // Check diagonals
  const diagonals = [
    { row: row - 1, col: col - 1 },
    { row: row - 1, col: col + 1 },
    { row: row + 1, col: col - 1 },
    { row: row + 1, col: col + 1 }
  ].filter(p => p.row >= 0 && p.row < board.size && p.col >= 0 && p.col < board.size)

  let controlled = 0
  for (const diag of diagonals) {
    const stone = getStone(board, diag.row, diag.col)
    if (stone === color || stone === null) controlled++
  }

  const minRequired = diagonals.length === 4 ? 3 : diagonals.length
  return controlled >= minRequired
}

/**
 * Count true eyes in a group
 */
export function countGroupEyes(board: Board, group: Position[]): number {
  if (group.length === 0) return 0

  const color = getStone(board, group[0]!.row, group[0]!.col)
  if (color === null) return 0

  // Find all empty adjacent positions
  const eyeCandidates = new Set<string>()
  for (const pos of group) {
    const adjacents = getAdjacentPositions(board, pos.row, pos.col)
    for (const adj of adjacents) {
      if (getStone(board, adj.row, adj.col) === null) {
        eyeCandidates.add(`${adj.row},${adj.col}`)
      }
    }
  }

  let eyes = 0
  for (const key of eyeCandidates) {
    const [row, col] = key.split(',').map(Number) as [number, number]
    if (hasEye(board, row, col, color)) eyes++
  }

  return eyes
}

/**
 * Determine if a group is alive (has 2+ eyes or enough space to make them)
 */
export function isGroupAlive(board: Board, group: Position[]): boolean {
  if (group.length === 0) return false

  const color = getStone(board, group[0]!.row, group[0]!.col)
  if (color === null) return false

  const eyes = countGroupEyes(board, group)
  const liberties = countLiberties(board, group)

  // Two eyes = unconditionally alive
  if (eyes >= 2) return true

  // Large group with many liberties likely alive
  if (group.length >= 6 && liberties >= 4) return true

  // One eye with few liberties = dead
  if (eyes === 1 && liberties <= 2) return false

  // No eyes - check if heavily surrounded
  if (eyes === 0) {
    const opponent = color === 'black' ? 'white' : 'black'

    // Get liberty positions
    const libertyPositions: Position[] = []
    for (const pos of group) {
      const adjacents = getAdjacentPositions(board, pos.row, pos.col)
      for (const adj of adjacents) {
        if (getStone(board, adj.row, adj.col) === null) {
          libertyPositions.push(adj)
        }
      }
    }

    // Count opponent stones around liberties
    let opponentSurrounding = 0
    for (const lib of libertyPositions) {
      const adjacents = getAdjacentPositions(board, lib.row, lib.col)
      for (const adj of adjacents) {
        if (getStone(board, adj.row, adj.col) === opponent) {
          opponentSurrounding++
        }
      }
    }

    // Heavily surrounded with no eyes = dead
    if (opponentSurrounding >= libertyPositions.length * 2 && liberties <= 3) {
      return false
    }
  }

  // Default: assume alive (conservative)
  return true
}

/**
 * Remove dead stones from the board.
 * Returns a new board with dead stones removed and counts of removed stones.
 */
export function removeDeadStones(board: Board): {
  board: Board
  blackRemoved: number
  whiteRemoved: number
} {
  let newBoard = board
  let blackRemoved = 0
  let whiteRemoved = 0

  const visited = new Set<string>()

  for (let row = 0; row < board.size; row++) {
    for (let col = 0; col < board.size; col++) {
      const key = `${row},${col}`
      if (visited.has(key)) continue

      const stone = getStone(newBoard, row, col)
      if (stone === null) continue

      const group = getGroup(newBoard, row, col)
      for (const pos of group) {
        visited.add(`${pos.row},${pos.col}`)
      }

      if (!isGroupAlive(newBoard, group)) {
        if (stone === 'black') {
          blackRemoved += group.length
        } else {
          whiteRemoved += group.length
        }
        newBoard = removeStones(newBoard, group)
      }
    }
  }

  return { board: newBoard, blackRemoved, whiteRemoved }
}

/**
 * Score a board with automatic dead stone removal
 */
export function scoreWithDeadRemoval(board: Board): {
  blackScore: number
  whiteScore: number
  blackStones: number
  whiteStones: number
  blackTerritory: number
  whiteTerritory: number
  blackDead: number
  whiteDead: number
} {
  const { board: cleanBoard, blackRemoved, whiteRemoved } = removeDeadStones(board)

  // Count stones
  let blackStones = 0
  let whiteStones = 0
  for (let row = 0; row < cleanBoard.size; row++) {
    for (let col = 0; col < cleanBoard.size; col++) {
      const stone = getStone(cleanBoard, row, col)
      if (stone === 'black') blackStones++
      if (stone === 'white') whiteStones++
    }
  }

  // Count territory
  const territory = countTerritory(cleanBoard)

  return {
    blackScore: blackStones + territory.black,
    whiteScore: whiteStones + territory.white,
    blackStones,
    whiteStones,
    blackTerritory: territory.black,
    whiteTerritory: territory.white,
    blackDead: blackRemoved,
    whiteDead: whiteRemoved
  }
}
