# Test Fixtures

**December 1, 2025**

---

> 定石百遍
> *Practice joseki a hundred times*

---

## Foundations Need Patterns

Testing Go AI requires positions—specific, repeatable arrangements.
Atari. Ladders. Corner enclosures. Capture races.

Creating these by hand in every test is tedious.
More importantly, it obscures intent.

What does this test?

```typescript
let board = createBoard(3)
board = placeStone(board, 0, 1, 'black')!
board = placeStone(board, 1, 0, 'black')!
board = placeStone(board, 1, 1, 'white')!
```

Versus this:

```typescript
const board = fixtures.atari_simple()
```

Intent is clear. The test reads like a sentence.

## The ASCII Approach

Go players think visually. ASCII boards match that intuition.

```typescript
// Parse ASCII into Board
export function parseAsciiBoard(ascii: string): Board {
  const lines = ascii
    .trim()
    .split('\n')
    .map(line => line.trim())
    .filter(line => line.length > 0)

  const pattern: StoneChar[][] = lines.map(line => {
    return line.split(/\s+/).map(char => {
      if (char === 'X' || char === 'B') return 'X'  // Black
      if (char === 'O' || char === 'W') return 'O'  // White
      return '.'  // Empty
    }) as StoneChar[]
  })

  return setupPosition(pattern)
}
```

Usage:

```typescript
const board = parseAsciiBoard(`
  . . .
  . X .
  O O .
`)

// Black stone at center in atari
// White can capture at (2,1)
```

The position is right there in the code.
No need to trace through `placeStone` calls.
The board reveals itself.

## Standard Fixtures

Common positions deserve names:

```typescript
// Life and death
export const atari_simple = (): Board => setupPosition([
  ['.', '.', '.'],
  ['.', 'X', '.'],
  ['O', 'O', '.']
])

export const selfCaptureCorner = (): Board => setupPosition([
  ['.', 'X', '.'],
  ['X', '.', '.'],
  ['.', '.', '.']
])

// Ladders
export const ladderWorks = (): Board => parseAsciiBoard(`
  . . . . . . . . .
  . . . . . . . . .
  . . . O . . . . .
  . . X X O . . . .
  . . . . . . . . .
  . . . . . . . . .
  . . . . . . . . .
  . . . . . . . . .
  . . . . . . . . .
`)

// Opening patterns
export const opening_44_44 = (): Board => parseAsciiBoard(`
  . . . . . . . . .
  . . . . . . . . .
  . . X . . . X . .
  . . . . . . . . .
  . . . . . . . . .
  . . . . . . . . .
  . . . . . . . . .
  . . . . . . . . .
  . . . . . . . . .
`)
```

Import and use:

```typescript
import { fixtures } from '@/test/fixtures'

it('should detect atari', () => {
  const board = fixtures.atari_simple()
  // Test atari detection logic
})

it('should avoid self-capture', () => {
  const board = fixtures.selfCaptureCorner()
  // Test self-capture avoidance
})
```

## Utilities for Custom Positions

Sometimes fixtures aren't enough. Build on the fly:

```typescript
import { createBoardWithStones } from '@/test/utils/boardSetup'

const board = createBoardWithStones(9, [
  { pos: { row: 2, col: 2 }, color: 'black' },
  { pos: { row: 2, col: 6 }, color: 'black' },
  { pos: { row: 6, col: 2 }, color: 'white' },
  { pos: { row: 6, col: 6 }, color: 'white' }
])

// Four corners occupied, ready for testing
```

Or match against a pattern:

```typescript
import { boardMatchesPattern } from '@/test/utils/boardSetup'

// After AI plays, verify result
const expected = [
  ['.', 'X', '.'],
  ['X', 'O', 'X'],
  ['.', 'X', '.']
]

expect(boardMatchesPattern(board, expected)).toBe(true)
```

## Directory Structure

```
src/test/
├── fixtures/
│   ├── index.ts                  // Central export
│   └── standardPositions.ts      // All standard fixtures
└── utils/
    └── boardSetup.ts              // Helper functions
```

Clean. Organized. One import to rule them all:

```typescript
import { fixtures } from '@/test/fixtures'
import { parseAsciiBoard, setupPosition } from '@/test/utils/boardSetup'
```

## The Benefit

Tests become readable.
Positions become reusable.
Intent becomes clear.

Before:

```typescript
it('should handle snapback', () => {
  let board = createBoard(5)
  board = placeStone(board, 1, 1, 'black')!
  board = placeStone(board, 1, 2, 'black')!
  board = placeStone(board, 1, 3, 'black')!
  // ... what pattern is this?
})
```

After:

```typescript
it('should handle snapback', () => {
  const board = fixtures.snapback()
  // Pattern is obvious, test reads cleanly
})
```

---

> 形を覚える
> *Remember the shapes*

Go is a game of patterns.
So are good tests.

The foundation is complete.
Phase 1 awaits.

---

[← Foundation](2025-11-30-foundation.md) | [Chronicle](../README.md)
