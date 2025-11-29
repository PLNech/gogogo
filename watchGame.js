#!/usr/bin/env node

/**
 * Watch Two AIs Play Against Each Other
 * Self-play with configurable parameters
 *
 * Usage:
 *   npm run watch              # Default 5x5 board, 50ms delay
 *   npm run watch -- -n 9      # 9x9 board
 *   npm run watch -- -n 19 -d 100  # 19x19 board, 100ms delay
 *   npm run watch -- -h        # Show help
 */

import { createBoard, placeStone, getStone, countTerritory } from './src/core/go/board.js'
import { computeMovePriors, evaluateMove } from './src/core/ai/policy.js'

// Parse CLI arguments
function parseArgs() {
    const args = process.argv.slice(2)
    const config = {
        size: 19,
        delay: 50,
        showHelp: false
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
  -h, --help          Show this help

EXAMPLES:
  npm run watch -- -n 9               # 9x9 board
  npm run watch -- -n 19 -d 100       # 19x19 board, slower moves
  npm run watch -- -n 13 -d 0         # 13x13 board, no delay

COMING SOON:
  -a, --ai <TYPE>     AI type (policy, mcts, hybrid)
  -i, --iterations    MCTS iterations per move
  -v, --verbose       Show detailed move analysis
  -q, --quiet         Minimal output
  -s, --save <FILE>   Save game to SGF file

`)
    process.exit(0)
}

const config = parseArgs()
if (config.showHelp) showHelp()

const DELAY_MS = config.delay
const SIZE = config.size

function visualizeBoard(board, lastMove) {
    console.clear()
    console.log('\n╔════════════════════════════════════════╗')
    console.log(`║   GoGoGo - AI Self-Play (${SIZE}x${SIZE})`.padEnd(41) + '║')
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

function showTopMoves(board, player, count = 3) {
    const priors = computeMovePriors(board, player)
    const moves = []

    for (const [key, prior] of priors) {
        const [row, col] = key.split(',').map(Number)
        const pos = { row, col }
        const score = evaluateMove(board, pos, player)
        moves.push({ pos, prior, score })
    }

    moves.sort((a, b) => b.prior - a.prior)

    const symbol = player === 'black' ? '●' : '○'
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
    let board = createBoard(SIZE)
    let moveCount = 0
    let passCount = 0
    let currentPlayer = 'black'
    let lastMove = null

    console.log('\n🎮 Starting self-play game...\n')
    await sleep(1000)

    while (moveCount < SIZE * SIZE && passCount < 2) {
        visualizeBoard(board, lastMove)

        const symbol = currentPlayer === 'black' ? '●' : '○'
        console.log(`Move ${moveCount + 1} - ${symbol} ${currentPlayer} to play`)
        console.log('─'.repeat(40))

        showTopMoves(board, currentPlayer, 3)

        const aiMove = getAIMove(board, currentPlayer)

        if (!aiMove) {
            console.log(`${symbol} passes (no legal moves)`)
            passCount++
            currentPlayer = currentPlayer === 'black' ? 'white' : 'black'
            await sleep(DELAY_MS * 2)
            continue
        }

        const newBoard = placeStone(board, aiMove.row, aiMove.col, currentPlayer)

        if (!newBoard) {
            console.log(`${symbol} passes (illegal move)`)
            passCount++
            currentPlayer = currentPlayer === 'black' ? 'white' : 'black'
            await sleep(DELAY_MS * 2)
            continue
        }

        board = newBoard
        lastMove = aiMove
        moveCount++
        passCount = 0

        console.log(`${symbol} plays (${aiMove.row},${aiMove.col}) - ${(aiMove.prior * 100).toFixed(1)}% confidence`)

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

    // Count stones on board
    let blackStones = 0
    let whiteStones = 0
    for (let row = 0; row < board.size; row++) {
        for (let col = 0; col < board.size; col++) {
            const stone = getStone(board, row, col)
            if (stone === 'black') blackStones++
            if (stone === 'white') whiteStones++
        }
    }

    const territory = countTerritory(board)
    const blackTotal = blackStones + territory.black
    const whiteTotal = whiteStones + territory.white

    console.log('Final Score:')
    console.log(`  ● Black: ${blackStones} stones + ${territory.black} territory = ${blackTotal} points`)
    console.log(`  ○ White: ${whiteStones} stones + ${territory.white} territory = ${whiteTotal} points`)

    if (blackTotal > whiteTotal) {
        console.log(`\n🏆 Black wins by ${blackTotal - whiteTotal} points!`)
    } else if (whiteTotal > blackTotal) {
        console.log(`\n🏆 White wins by ${whiteTotal - blackTotal} points!`)
    } else {
        console.log('\n🤝 Draw!')
    }

    console.log(`\n⏱️  Average time per move: ${DELAY_MS}ms`)
    console.log()
}

console.log('Starting game in 1 second...')
playGame().catch(console.error)
