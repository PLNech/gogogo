/**
 * Neural network inference for Go using ONNX Runtime Web.
 *
 * Loads a trained neural network model and provides policy/value predictions.
 * This integrates with the heuristic AI system to provide Level 6+ AI.
 */

import * as ort from 'onnxruntime-web'
import type { Board, Position } from '../go/types'
import { getStone } from '../go/board'
import type { AIDecision } from './simpleAI'

// Model configuration
const MODEL_BASE_PATH = '/gogogo/play/models/go_9x9'
const MODEL_PATH = `${MODEL_BASE_PATH}/model.onnx`
const METADATA_PATH = `${MODEL_BASE_PATH}/metadata.json`

// Model metadata
interface ModelMetadata {
  name: string
  version: string
  board_size: number
  input_planes: number
  architecture: string
  training?: {
    method: string
    games: number
    hybrid?: boolean
  }
  notes?: string
}

// Singleton session and metadata
let session: ort.InferenceSession | null = null
let loadPromise: Promise<ort.InferenceSession> | null = null
let metadataPromise: Promise<ModelMetadata | null> | null = null
let modelMetadata: ModelMetadata | null = null

// Constants (updated from metadata when loaded)
const INPUT_PLANES = 17
const DEFAULT_BOARD_SIZE = 9

/**
 * Load model metadata.
 * Returns null if metadata cannot be loaded.
 */
export async function loadModelMetadata(): Promise<ModelMetadata | null> {
  if (modelMetadata) return modelMetadata

  if (metadataPromise) return metadataPromise

  metadataPromise = (async () => {
    try {
      console.log('[Neural] Loading metadata from', METADATA_PATH)
      const response = await fetch(METADATA_PATH)
      if (!response.ok) {
        console.warn('[Neural] No metadata found, using defaults')
        return null
      }
      const metadata = await response.json() as ModelMetadata
      modelMetadata = metadata
      console.log('[Neural] Metadata loaded:', metadata.name, `(${metadata.board_size}x${metadata.board_size})`)
      return metadata
    } catch (error) {
      console.warn('[Neural] Failed to load metadata:', error)
      return null
    }
  })()

  return metadataPromise
}

/**
 * Get the board size supported by the neural model.
 */
export async function getNeuralModelBoardSize(): Promise<number> {
  const metadata = await loadModelMetadata()
  return metadata?.board_size ?? DEFAULT_BOARD_SIZE
}

/**
 * Check if neural model supports a given board size.
 */
export async function supportsNeuralBoardSize(size: number): Promise<boolean> {
  const modelSize = await getNeuralModelBoardSize()
  return size === modelSize
}

/**
 * Load the neural network model.
 * Returns cached session if already loaded.
 */
export async function loadNeuralModel(): Promise<ort.InferenceSession> {
  if (session) return session

  if (loadPromise) return loadPromise

  loadPromise = (async () => {
    // Load metadata first (non-blocking if fails)
    await loadModelMetadata()

    console.log('[Neural] Loading model from', MODEL_PATH)

    try {
      // Configure ONNX Runtime
      ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/'

      const newSession = await ort.InferenceSession.create(MODEL_PATH, {
        executionProviders: ['wasm'],
        graphOptimizationLevel: 'all'
      })

      console.log('[Neural] Model loaded successfully')
      console.log('[Neural] Inputs:', newSession.inputNames)
      console.log('[Neural] Outputs:', newSession.outputNames)

      session = newSession
      return session
    } catch (error) {
      console.error('[Neural] Failed to load model:', error)
      loadPromise = null
      throw error
    }
  })()

  return loadPromise
}

/**
 * Check if neural model is loaded.
 */
export function isNeuralModelLoaded(): boolean {
  return session !== null
}

/**
 * Convert board state to input tensor.
 *
 * Feature planes (17 total):
 * - Planes 0-7: Current player's stones (with history if available)
 * - Planes 8-15: Opponent's stones (with history if available)
 * - Plane 16: Current player indicator (all 1s for black, all 0s for white)
 */
export function boardToTensor(
  board: Board,
  player: 'black' | 'white',
  moveHistory: Position[] = []
): Float32Array {
  const size = board.size
  const planes = INPUT_PLANES
  const tensor = new Float32Array(planes * size * size)

  // For simplicity, we only use current position (no history)
  // Plane 0: Current player's stones
  // Plane 8: Opponent's stones
  // Plane 16: Player to move indicator

  const currentStone = player === 'black' ? 'black' : 'white'
  const opponentStone = player === 'black' ? 'white' : 'black'

  for (let row = 0; row < size; row++) {
    for (let col = 0; col < size; col++) {
      const idx = row * size + col
      const stone = getStone(board, row, col)

      // Plane 0: Current player's stones
      if (stone === currentStone) {
        tensor[0 * size * size + idx] = 1.0
      }

      // Plane 8: Opponent's stones
      if (stone === opponentStone) {
        tensor[8 * size * size + idx] = 1.0
      }

      // Plane 16: Player to move (1 for black, 0 for white)
      tensor[16 * size * size + idx] = player === 'black' ? 1.0 : 0.0
    }
  }

  return tensor
}

/**
 * Run neural network inference.
 *
 * Returns:
 *   - policy: Probability distribution over moves (size^2 + 1 for pass)
 *   - value: Win probability for current player (-1 to 1)
 */
export async function runInference(
  board: Board,
  player: 'black' | 'white'
): Promise<{ policy: Float32Array; value: number }> {
  const sess = await loadNeuralModel()

  const size = board.size
  const inputTensor = boardToTensor(board, player)

  // Create ONNX tensor
  const onnxTensor = new ort.Tensor('float32', inputTensor, [1, INPUT_PLANES, size, size])

  // Run inference
  const results = await sess.run({ board: onnxTensor })

  // Extract outputs
  const policyTensor = results['policy']
  const valueTensor = results['value']

  if (!policyTensor || !valueTensor) {
    throw new Error('Model output missing policy or value')
  }

  const policy = policyTensor.data as Float32Array
  const valueData = valueTensor.data as Float32Array
  const value = valueData[0] ?? 0

  return { policy, value }
}

/**
 * Get AI decision using neural network.
 *
 * Uses the trained neural network to:
 * 1. Get policy (move probabilities)
 * 2. Get value (win probability)
 * 3. Select best legal move from policy
 */
export async function getNeuralAIDecision(
  board: Board,
  player: 'black' | 'white',
  temperature: number = 0.0
): Promise<AIDecision> {
  try {
    const { policy, value } = await runInference(board, player)

    const size = board.size
    const passIdx = size * size

    // Build list of legal moves with their policy values
    type MoveWithProb = { position: Position; prob: number; idx: number }
    const legalMoves: MoveWithProb[] = []

    for (let row = 0; row < size; row++) {
      for (let col = 0; col < size; col++) {
        const idx = row * size + col
        const stone = getStone(board, row, col)

        // Only consider empty positions
        if (stone === null) {
          legalMoves.push({
            position: { row, col },
            prob: policy[idx] ?? 0,
            idx
          })
        }
      }
    }

    const passProb = policy[passIdx] ?? 0

    // If no legal moves, pass
    if (legalMoves.length === 0) {
      return {
        action: 'pass',
        confidence: passProb,
        score: value * 100
      }
    }

    // Sort by probability
    legalMoves.sort((a, b) => b.prob - a.prob)

    let selectedMove: MoveWithProb = legalMoves[0]!

    if (temperature > 0 && legalMoves.length > 1) {
      // Sample with temperature
      const probs = legalMoves.map(m => Math.pow(m.prob, 1 / temperature))
      const sum = probs.reduce((a, b) => a + b, 0)
      const normalized = probs.map(p => p / sum)

      // Sample
      const rand = Math.random()
      let cumulative = 0
      let selectedIdx = 0
      for (let i = 0; i < normalized.length; i++) {
        const p = normalized[i] ?? 0
        cumulative += p
        if (rand < cumulative) {
          selectedIdx = i
          break
        }
      }
      selectedMove = legalMoves[selectedIdx] ?? legalMoves[0]!
    }

    // Check if pass has higher probability than best move
    if (passProb > selectedMove.prob * 1.5) {
      return {
        action: 'pass',
        confidence: passProb,
        score: value * 100
      }
    }

    return {
      action: 'move',
      position: selectedMove.position,
      confidence: selectedMove.prob,
      score: value * 100
    }
  } catch (error) {
    console.error('[Neural] Inference error:', error)
    // Fall back to pass on error
    return {
      action: 'pass',
      confidence: 0,
      score: 0
    }
  }
}

/**
 * Get move probabilities for visualization.
 *
 * Returns an object mapping position strings to probabilities.
 */
export async function getMoveProbs(
  board: Board,
  player: 'black' | 'white'
): Promise<Map<string, number>> {
  const { policy } = await runInference(board, player)

  const probs = new Map<string, number>()
  const size = board.size

  for (let row = 0; row < size; row++) {
    for (let col = 0; col < size; col++) {
      const idx = row * size + col
      const key = `${row},${col}`
      probs.set(key, policy[idx] ?? 0)
    }
  }

  // Add pass
  probs.set('pass', policy[size * size] ?? 0)

  return probs
}
