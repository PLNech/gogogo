import type { Board, Position, Stone } from '../go/types'
import { getStone, placeStone, captureStones, countTerritory, getGroup, countLiberties } from '../go/board'
import type { AIConfig } from './types'
import { estimateScore } from './evaluation'

// MCTS Node structure
export interface MCTSNode {
  position: Position | null
  visits: number
  wins: number
  children: MCTSNode[]
  parent: MCTSNode | null
  boardState: Board
  isTerminal: boolean
  moveValue?: number // For visualization purposes
}

// MCTS search result
export interface MCTSResult {
  position: Position
  winRate: number
  visits: number
  bestPath?: Position[]
}

/**
 * Simple MCTS implementation for Go AI
 */
export class GoMCTS {
  private root!: MCTSNode
  private config: AIConfig
  private maxIterations: number
  private maxTimeMs: number

  constructor(config: AIConfig, maxIterations: number = 1000, maxTimeMs: number = 50) {
    this.config = config
    this.maxIterations = maxIterations
    this.maxTimeMs = maxTimeMs
  }

  /**
   * Main MCTS search function
   */
  public search(board: Board, player: 'black' | 'white', moveCount: number): MCTSResult {
    this.root = this.createRootNode(board)

    const startTime = Date.now()
    let iterations = 0

    // Run MCTS iterations until time limit or max iterations
    while (iterations < this.maxIterations && (Date.now() - startTime) < this.maxTimeMs) {
      this.iteration(player)
      iterations++
    }

    // Find the best move based on visit count (or win rate)
    const bestChild = this.selectBestChild(this.root)

    if (!bestChild || !bestChild.position) {
      // Fallback to simple evaluation if no good move found
      return this.fallbackToSimpleEvaluation(board, player, moveCount)
    }

    return {
      position: bestChild.position,
      winRate: bestChild.visits > 0 ? bestChild.wins / bestChild.visits : 0,
      visits: bestChild.visits,
      bestPath: this.getBestPath(bestChild)
    }
  }

  /**
   * Single MCTS iteration
   */
  private iteration(player: 'black' | 'white'): void {
    // 1. Selection: Traverse from root to leaf
    const node = this.select(this.root)

    // 2. Expansion: Add children if not terminal
    if (!node.isTerminal) {
      this.expand(node, player)
    }

    // 3. Simulation: Play out from leaf
    const result = this.simulate(node.boardState, player)

    // 4. Backpropagation: Update statistics up the tree
    this.backpropagate(node, result)
  }

  /**
   * Selection phase - traverse tree using UCB1 formula
   */
  private select(node: MCTSNode): MCTSNode {
    while (node.children.length > 0 && !node.isTerminal) {
      const bestChild = this.selectBestUCBChild(node)
      if (!bestChild) break
      node = bestChild
    }
    return node
  }

  /**
   * Expand node with children (all legal moves)
   */
  private expand(node: MCTSNode, player: 'black' | 'white'): void {
    const emptyPositions = this.getEmptyPositions(node.boardState)

    for (const pos of emptyPositions) {
      // Check if this move would be valid
      const newBoard = placeStone(node.boardState, pos.row, pos.col, player)
      if (newBoard) {
        const newNode: MCTSNode = {
          position: pos,
          visits: 0,
          wins: 0,
          children: [],
          parent: node,
          boardState: newBoard,
          isTerminal: false
        }
        node.children.push(newNode)
      }
    }
  }

  /**
   * Simulation phase - play random playouts
   */
  private simulate(board: Board, player: 'black' | 'white'): number {
    // Simple random playout for now
    // In a full implementation, this would be more sophisticated
    let currentBoard = board
    let currentPlayer: 'black' | 'white' = player
    let moveCount = 0
    const maxMoves = 200 // Prevent infinite loops

    // Play random moves until game ends or max moves reached
    while (moveCount < maxMoves) {
      const emptyPositions = this.getEmptyPositions(currentBoard)

      if (emptyPositions.length === 0) {
        break // Game over
      }

      // Choose a random move
      const randomIndex = Math.floor(Math.random() * emptyPositions.length)
      const move = emptyPositions[randomIndex]!

      const newBoard = placeStone(currentBoard, move.row, move.col, currentPlayer)
      if (!newBoard) {
        // Invalid move, try another
        currentPlayer = currentPlayer === 'black' ? 'white' : 'black'
        moveCount++
        continue
      }

      // Apply captures
      const { board: boardAfterCapture } = captureStones(newBoard, move!.row, move!.col, currentPlayer)
      currentBoard = boardAfterCapture

      currentPlayer = currentPlayer === 'black' ? 'white' : 'black'
      moveCount++
    }

    // Evaluate final position
    const territory = countTerritory(currentBoard)
    const blackScore = territory.black
    const whiteScore = territory.white

    // Return win result (1 for player win, 0 for loss, 0.5 for draw)
    if (player === 'black') {
      return blackScore > whiteScore ? 1 : blackScore < whiteScore ? 0 : 0.5
    } else {
      return whiteScore > blackScore ? 1 : whiteScore < blackScore ? 0 : 0.5
    }
  }

  /**
   * Backpropagation - update statistics up the tree
   */
  private backpropagate(node: MCTSNode, result: number): void {
    let current: MCTSNode | null = node

    while (current) {
      current.visits++
      current.wins += result

      // Store move value for visualization
      if (current.position) {
        current.moveValue = result
      }

      current = current.parent
    }
  }

  /**
   * Select best child using UCB1 formula
   */
  private selectBestUCBChild(node: MCTSNode): MCTSNode | null {
    if (node.children.length === 0) return null

    const explorationConstant = 1.41 // sqrt(2)
    let bestChild: MCTSNode | null = null
    let bestUCBValue = -Infinity

    for (const child of node.children) {
      if (child.visits === 0) {
        // Give unvisited nodes a high UCB value to encourage exploration
        return child
      }

      const ucbValue = (child.wins / child.visits) +
        explorationConstant * Math.sqrt(Math.log(node.visits) / child.visits)

      if (ucbValue > bestUCBValue) {
        bestUCBValue = ucbValue
        bestChild = child
      }
    }

    return bestChild
  }

  /**
   * Select best child based on visits (for final move selection)
   */
  private selectBestChild(node: MCTSNode): MCTSNode | null {
    if (node.children.length === 0) return null

    let bestChild: MCTSNode | null = null
    let maxVisits = -1

    for (const child of node.children) {
      if (child.visits > maxVisits) {
        maxVisits = child.visits
        bestChild = child
      }
    }

    return bestChild
  }

  /**
   * Get best path from root to leaf for visualization
   */
  private getBestPath(node: MCTSNode): Position[] {
    const path: Position[] = []
    let current: MCTSNode | null = node

    while (current && current.position) {
      path.unshift(current.position)
      current = current.parent
    }

    return path.slice(0, 10) // Limit to first 10 moves for clarity
  }

  /**
   * Fallback evaluation when MCTS fails
   */
  private fallbackToSimpleEvaluation(board: Board, player: 'black' | 'white', moveCount: number): MCTSResult {
    // Use existing simple AI evaluation as fallback
    const emptyPositions = this.getEmptyPositions(board)

    if (emptyPositions.length === 0) {
      return {
        position: { row: 0, col: 0 },
        winRate: 0,
        visits: 0
      }
    }

    // Pick a reasonable move based on existing heuristics
    const bestPosition = emptyPositions[0]! // Simple fallback

    return {
      position: bestPosition,
      winRate: 0.5,
      visits: 1
    }
  }

  /**
   * Helper to get empty positions
   */
  private getEmptyPositions(board: Board): Position[] {
    const positions: Position[] = []
    for (let row = 0; row < board.size; row++) {
      for (let col = 0; col < board.size; col++) {
        if (getStone(board, row, col) === null) {
          positions.push({ row, col })
        }
      }
    }
    return positions
  }

  /**
   * Create root node for MCTS
   */
  private createRootNode(board: Board): MCTSNode {
    return {
      position: null,
      visits: 0,
      wins: 0,
      children: [],
      parent: null,
      boardState: board,
      isTerminal: false
    }
  }
}