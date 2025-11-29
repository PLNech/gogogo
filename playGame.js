#!/usr/bin/env node

/**
 * Play a game of Go against the AI
 *
 * Usage:
 *   npm run play              # Default 5x5 board
 *   npm run play -- -n 9      # 9x9 board
 *   npm run play -- -h        # Show help
 */

import { createBoard, placeStone, getStone } from './src/core/go/board.js'
import { computeMovePriors } from './src/core/ai/policy.js'
import * as readline from 'readline'

// Parse CLI arguments
function parseArgs() {
    const args = process.argv.slice(2)
    const config = {
        size: 19,
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
║   GoGoGo - Play Against AI             ║
╚════════════════════════════════════════╝

Play a game of Go against the AI.

USAGE:
  npm run play                   # Default: 19x19 board
  npm run play -- [OPTIONS]

OPTIONS:
  -n, --size <N>      Board size (3-19)           [default: 19]
  -h, --help          Show this help

EXAMPLES:
  npm run play -- -n 9            # Play on 9x9 board
  npm run play -- -n 13           # Play on 13x13 board

GAMEPLAY:
  • You are Black (●), AI is White (○)
  • Enter moves as: row,col (e.g., "2,3")
  • Type "pass" or "p" to pass
  • Type "quit" or "q" to exit

`)
    process.exit(0)
}

const config = parseArgs()
if (config.showHelp) showHelp()

const SIZE = config.size

const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout
})

function visualizeBoard(board) {
    // Column headers with two digits
    console.log('\n   ' + Array.from({ length: board.size }, (_, i) => String(i).padStart(2, '0')).join(' '))
    console.log('   ' + '─'.repeat(board.size * 3 - 1))

    for (let row = 0; row < board.size; row++) {
        // Row label with two digits
        let line = `${String(row).padStart(2, '0')}│`
        for (let col = 0; col < board.size; col++) {
            const stone = getStone(board, row, col)
            if (stone === 'black') {
                line += '●'
            } else if (stone === 'white') {
                line += '○'
            } else {
                line += '·'
            }
            if (col < board.size - 1) line += '  ' // Two spaces between columns
        }
        console.log(line)
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
            bestMove = { row, col }
        }
    }

    return bestMove
}

async function playGame() {
    console.log('\n╔════════════════════════════════════════╗')
    console.log('║   GoGoGo - Play Against AI            ║')
    console.log('╚════════════════════════════════════════╝\n')

    const SIZE = 5
    let board = createBoard(SIZE)
    let moveCount = 0

    console.log('You are Black (●). AI is White (○).')
    console.log('Enter moves as: row,col (e.g., "0,0" for top-left)')
    console.log('Type "pass" to pass, "quit" to exit\n')

    visualizeBoard(board)

    function askForMove() {
        rl.question('Your move (row,col): ', async (answer) => {
            answer = answer.trim().toLowerCase()

            if (answer === 'quit' || answer === 'q') {
                console.log('\nThanks for playing!\n')
                rl.close()
                return
            }

            if (answer === 'pass' || answer === 'p') {
                console.log('\nYou passed.')
                aiMove()
                return
            }

            const parts = answer.split(',')
            if (parts.length !== 2) {
                console.log('Invalid format. Use: row,col')
                askForMove()
                return
            }

            const row = parseInt(parts[0])
            const col = parseInt(parts[1])

            if (isNaN(row) || isNaN(col) || row < 0 || row >= SIZE || col < 0 || col >= SIZE) {
                console.log('Invalid position. Must be 0-4.')
                askForMove()
                return
            }

            if (getStone(board, row, col) !== null) {
                console.log('Position already occupied!')
                askForMove()
                return
            }

            const newBoard = placeStone(board, row, col, 'black')
            if (!newBoard) {
                console.log('Illegal move (ko rule or self-capture)')
                askForMove()
                return
            }

            board = newBoard
            moveCount++
            visualizeBoard(board)

            // AI turn
            aiMove()
        })
    }

    function aiMove() {
        console.log('AI thinking...')

        const aiMovePos = getAIMove(board, 'white')
        if (!aiMovePos) {
            console.log('AI passes (no legal moves)\n')
            console.log('Game over! Thanks for playing.\n')
            rl.close()
            return
        }

        const newBoard = placeStone(board, aiMovePos.row, aiMovePos.col, 'white')
        if (!newBoard) {
            console.log('AI passes (illegal move)\n')
            console.log('Game over! Thanks for playing.\n')
            rl.close()
            return
        }

        board = newBoard
        moveCount++

        console.log(`AI plays: (${aiMovePos.row},${aiMovePos.col})`)
        visualizeBoard(board)

        if (moveCount >= SIZE * SIZE) {
            console.log('Board full! Game over.\n')
            rl.close()
            return
        }

        askForMove()
    }

    askForMove()
}

playGame().catch(console.error)
