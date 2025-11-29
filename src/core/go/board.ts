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

  // Ko rule: check if this move would return to the previous board state
  // (after captures are applied)
  if (previousBoard) {
    const { board: boardAfterCaptures } = captureStones(newBoard, row, col, stone)
    if (boardsEqual(boardAfterCaptures, previousBoard)) {
      return null // Ko violation
    }
  }

  return newBoard
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
