// Core Go types
export type Stone = 'black' | 'white' | null

export type Position = {
  row: number
  col: number
}

export interface Board {
  size: number
  stones: Stone[][]
}

export type GameResult = {
  winner: 'black' | 'white' | 'draw'
  blackScore: number
  whiteScore: number
  blackCaptures: number
  whiteCaptures: number
}
