#!/usr/bin/env node

/**
 * Watch Two AIs Play Against Each Other
 * Self-play with 50ms delay per move
 */

import { createBoard, placeStone, getStone, countTerritory } from './src/core/go/board.js'
import { computeMovePriors, evaluateMove } from './src/core/ai/policy.js'

const DELAY_MS = 50
const SIZE = 5

function visualizeBoard(board, lastMove) {
    console.clear()
    console.log('\n╔════════════════════════════════════════╗')
    console.log('║   GoGoGo - AI Self-Play                ║')
    console.log('╚════════════════════════════════════════╝\n')

    console.log('  ' + Array.from({ length: board.size }, (_, i) => i).join(' '))
    console.log('  ' + '─'.repeat(board.size * 2 - 1))

    for (let row = 0; row < board.size; row++) {
        let line = `${row}│`
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
            if (col < board.size - 1) line += ' '
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

    console.log(`Total moves: ${moveCount}`)

    const territory = countTerritory(board)
    console.log(`\nTerritory:`)
    console.log(`  ● Black: ${territory.black} points`)
    console.log(`  ○ White: ${territory.white} points`)

    if (territory.black > territory.white) {
        console.log('\n🏆 Black wins!')
    } else if (territory.white > territory.black) {
        console.log('\n🏆 White wins!')
    } else {
        console.log('\n🤝 Draw!')
    }

    console.log()
}

console.log('Starting game in 1 second...')
playGame().catch(console.error)
