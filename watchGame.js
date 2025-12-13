#!/usr/bin/env node

/**
 * Watch Two AIs Play Against Each Other
 * Self-play with configurable parameters
 *
 * Usage:
 *   npm run watch              # Default 19x19 board, 50ms delay
 *   npm run watch -- -n 9      # 9x9 board
 *   npm run watch -- -n 19 -d 100  # 19x19 board, 100ms delay
 *   npm run watch -- --neural  # Use neural network AI (requires serve.py)
 *   npm run watch -- -h        # Show help
 */

import { createBoard, placeStone, getStone, countTerritory, scoreWithDeadRemoval } from './src/core/go/board.js'
import { computeMovePriors, evaluateMove } from './src/core/ai/policy.js'

// Neural server configuration
const NEURAL_SERVER = 'http://localhost:8765'
let useNeural = false
let neuralAvailable = false

// Parse CLI arguments
function parseArgs() {
    const args = process.argv.slice(2)
    const config = {
        size: 19,
        delay: 50,
        showHelp: false,
        neural: false
    }

    for (let i = 0; i < args.length; i++) {
        switch (args[i]) {
            case '-h':
            case '--help':
                config.showHelp = true
                break
            case '-n':
            case '--size':
                config.size = parseInt(args[++i])
                if (isNaN(config.size) || config.size < 3 || config.size > 19) {
                    console.error('Error: Board size must be between 3 and 19')
                    process.exit(1)
                }
                break
            case '-d':
            case '--delay':
                config.delay = parseInt(args[++i])
                if (isNaN(config.delay) || config.delay < 0) {
                    console.error('Error: Delay must be a positive number')
                    process.exit(1)
                }
                break
            case '--neural':
            case '-N':
                config.neural = true
                break
            default:
                console.error(`Unknown option: ${args[i]}`)
                console.log('Use -h for help')
                process.exit(1)
        }
    }

    return config
}

function showHelp() {
    console.log(`
╔════════════════════════════════════════╗
║   GoGoGo - AI Self-Play Viewer         ║
╚════════════════════════════════════════╝

Watch two AIs play against each other with live visualization.

USAGE:
  npm run watch                      # Default: 19x19 board, 50ms delay
  npm run watch -- [OPTIONS]

OPTIONS:
  -n, --size <N>      Board size (3-19)           [default: 19]
  -d, --delay <MS>    Delay per move (ms)         [default: 50]
  -N, --neural        Use neural network AI (requires serve.py)
  -h, --help          Show this help

NEURAL MODE:
  First start the neural server in training/:
    cd training && poetry run python serve.py

  Then run with --neural flag:
    npm run watch -- --neural

EXAMPLES:
  npm run watch -- -n 9               # 9x9 board (heuristic AI)
  npm run watch -- --neural           # 19x19 neural network
  npm run watch -- -n 19 -d 100       # 19x19 board, slower moves

`)
    process.exit(0)
}

const config = parseArgs()
if (config.showHelp) showHelp()

const DELAY_MS = config.delay
const SIZE = config.size
useNeural = config.neural

// Check neural server availability
async function checkNeuralServer() {
    if (!useNeural) return false
    try {
        const response = await fetch(`${NEURAL_SERVER}/status`)
        if (response.ok) {
            const data = await response.json()
            console.log(`🧠 Neural server connected: ${data.model.blocks} blocks, ${data.model.filters} filters`)
            if (data.model.board_size !== SIZE) {
                console.log(`⚠️  Model trained on ${data.model.board_size}x${data.model.board_size}, using that size`)
                // Can't change SIZE here as it's const, will just warn
            }
            return true
        }
    } catch (e) {
        console.log('❌ Neural server not available, falling back to heuristic AI')
        console.log('   Start server with: cd training && poetry run python serve.py')
    }
    return false
}

// Get move from neural server
async function getNeuralMove(board, player) {
    const size = board.size
    // Convert board to array format
    const boardArray = []
    for (let r = 0; r < size; r++) {
        const row = []
        for (let c = 0; c < size; c++) {
            const stone = getStone(board, r, c)
            row.push(stone === 'black' ? 1 : stone === 'white' ? -1 : 0)
        }
        boardArray.push(row)
    }

    const currentPlayer = player === 'black' ? 1 : -1

    try {
        const response = await fetch(`${NEURAL_SERVER}/move`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                size,
                board: boardArray,
                current_player: currentPlayer
            })
        })

        if (response.ok) {
            const data = await response.json()
            if (data.moves && data.moves.length > 0) {
                const best = data.moves[0]
                return {
                    row: best.row,
                    col: best.col,
                    prior: best.prob,
                    value: data.value,
                    move: best.move,
                    topMoves: data.moves.slice(0, 3)
                }
            }
        }
    } catch (e) {
        // Fall through to return null
    }
    return null
}

function visualizeBoard(board, lastMove) {
    console.clear()
    const aiType = neuralAvailable ? '🧠 Neural' : '📊 Heuristic'
    console.log('\n╔════════════════════════════════════════╗')
    console.log(`║   GoGoGo - ${aiType} (${board.size}x${board.size})`.padEnd(41) + '║')
    console.log('╚════════════════════════════════════════╝\n')

    // Column headers with two digits
    console.log('   ' + Array.from({ length: board.size }, (_, i) => String(i).padStart(2, '0')).join(' '))
    console.log('   ' + '─'.repeat(board.size * 3 - 1))

    for (let row = 0; row < board.size; row++) {
        // Row label with two digits
        let line = `${String(row).padStart(2, '0')}│`
        for (let col = 0; col < board.size; col++) {
            const stone = getStone(board, row, col)
            const isLastMove = lastMove && lastMove.row === row && lastMove.col === col

            if (stone === 'black') {
                line += isLastMove ? '◉' : '●'
            } else if (stone === 'white') {
                line += isLastMove ? '⊙' : '○'
            } else {
                line += '·'
            }
            if (col < board.size - 1) line += '  ' // Two spaces between columns
        }
        console.log(line)
    }
    console.log()
}

function showTopMoves(board, player, count = 3, neuralMoves = null) {
    const symbol = player === 'black' ? '●' : '○'

    if (neuralMoves) {
        console.log(`${symbol} Top moves (neural):`)
        for (let i = 0; i < Math.min(count, neuralMoves.length); i++) {
            const m = neuralMoves[i]
            const percent = (m.prob * 100).toFixed(1)
            console.log(`  ${i + 1}. ${m.move} - ${percent}%`)
        }
        console.log()
        return
    }

    const priors = computeMovePriors(board, player)
    const moves = []

    for (const [key, prior] of priors) {
        const [row, col] = key.split(',').map(Number)
        const pos = { row, col }
        const score = evaluateMove(board, pos, player)
        moves.push({ pos, prior, score })
    }

    moves.sort((a, b) => b.prior - a.prior)

    console.log(`${symbol} Top moves:`)
    for (let i = 0; i < Math.min(count, moves.length); i++) {
        const m = moves[i]
        const percent = (m.prior * 100).toFixed(1)
        console.log(`  ${i + 1}. (${m.pos.row},${m.pos.col}) - ${percent}% (score: ${m.score.toFixed(1)})`)
    }
    console.log()
}

function getAIMove(board, player) {
    const priors = computeMovePriors(board, player)
    let bestMove = null
    let bestPrior = -1

    for (const [key, prior] of priors) {
        if (prior > bestPrior) {
            bestPrior = prior
            const [row, col] = key.split(',').map(Number)
            bestMove = { row, col, prior }
        }
    }

    return bestMove
}

function sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms))
}

async function playGame() {
    // Check neural server if requested
    neuralAvailable = await checkNeuralServer()

    let board = createBoard(SIZE)
    let moveCount = 0
    let passCount = 0
    let currentPlayer = 'black'
    let lastMove = null

    const aiLabel = neuralAvailable ? '🧠 Neural network' : '📊 Heuristic'
    console.log(`\n🎮 Starting self-play game with ${aiLabel}...\n`)
    await sleep(1000)

    while (moveCount < SIZE * SIZE && passCount < 2) {
        visualizeBoard(board, lastMove)

        const symbol = currentPlayer === 'black' ? '●' : '○'
        console.log(`Move ${moveCount + 1} - ${symbol} ${currentPlayer} to play`)
        console.log('─'.repeat(40))

        // Try neural move first if available
        let aiMove = null
        let neuralData = null

        if (neuralAvailable) {
            neuralData = await getNeuralMove(board, currentPlayer)
            if (neuralData) {
                aiMove = { row: neuralData.row, col: neuralData.col, prior: neuralData.prior, move: neuralData.move }
                showTopMoves(board, currentPlayer, 3, neuralData.topMoves)
                console.log(`Value: ${(neuralData.value * 100).toFixed(1)}% (${neuralData.value > 0 ? 'black' : 'white'} favored)`)
            }
        }

        if (!aiMove) {
            showTopMoves(board, currentPlayer, 3)
            aiMove = getAIMove(board, currentPlayer)
        }

        if (!aiMove) {
            console.log(`${symbol} passes (no legal moves)`)
            passCount++
            currentPlayer = currentPlayer === 'black' ? 'white' : 'black'
            await sleep(DELAY_MS * 2)
            continue
        }

        // Try to play the move - with retry logic for illegal moves
        let newBoard = placeStone(board, aiMove.row, aiMove.col, currentPlayer)
        let attempts = 1
        const maxAttempts = 5

        while (!newBoard && attempts < maxAttempts) {
            const moveLabel = aiMove.move || `(${aiMove.row},${aiMove.col})`
            console.log(`  ⚠️  Illegal move ${moveLabel} (attempt ${attempts}) - trying another...`)

            // Get alternative move (use heuristic as fallback)
            const altMove = getAIMove(board, currentPlayer)
            if (!altMove) break

            aiMove = altMove
            newBoard = placeStone(board, aiMove.row, aiMove.col, currentPlayer)
            attempts++
        }

        if (!newBoard) {
            console.log(`  ❌ ${symbol} forfeits turn (${attempts} illegal moves)`)
            passCount++
            currentPlayer = currentPlayer === 'black' ? 'white' : 'black'
            await sleep(DELAY_MS * 2)
            continue
        }

        board = newBoard
        lastMove = aiMove
        moveCount++
        passCount = 0

        const moveLabel = neuralAvailable ? aiMove.move || `(${aiMove.row},${aiMove.col})` : `(${aiMove.row},${aiMove.col})`
        console.log(`${symbol} plays ${moveLabel} - ${(aiMove.prior * 100).toFixed(1)}% confidence`)

        // Switch player
        currentPlayer = currentPlayer === 'black' ? 'white' : 'black'

        await sleep(DELAY_MS)
    }

    // Final board
    visualizeBoard(board, lastMove)

    console.log('\n╔════════════════════════════════════════╗')
    console.log('║   Game Complete!                       ║')
    console.log('╚════════════════════════════════════════╝\n')

    console.log(`Board: ${SIZE}x${SIZE}`)
    console.log(`Total moves: ${moveCount}`)
    console.log(`Reason: ${passCount >= 2 ? 'Both players passed' : 'Board full'}\n`)

    // Score with dead stone removal
    const score = scoreWithDeadRemoval(board)

    console.log('Final Score (with dead stone removal):')
    if (score.blackDead > 0 || score.whiteDead > 0) {
        console.log(`  ☠️  Dead stones removed: Black=${score.blackDead}, White=${score.whiteDead}`)
    }
    console.log(`  ● Black: ${score.blackStones} stones + ${score.blackTerritory} territory = ${score.blackScore} points`)
    console.log(`  ○ White: ${score.whiteStones} stones + ${score.whiteTerritory} territory = ${score.whiteScore} points`)

    if (score.blackScore > score.whiteScore) {
        console.log(`\n🏆 Black wins by ${score.blackScore - score.whiteScore} points!`)
    } else if (score.whiteScore > score.blackScore) {
        console.log(`\n🏆 White wins by ${score.whiteScore - score.blackScore} points!`)
    } else {
        console.log('\n🤝 Draw!')
    }

    console.log(`\n⏱️  Average time per move: ${DELAY_MS}ms`)
    console.log()
}

console.log('Starting game in 1 second...')
playGame().catch(console.error)
