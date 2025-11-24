/**
 * Idle mechanics types - "The stones find you, even in stillness"
 */

export type IdleSourceId =
  | 'contemplation'      // Basic passive income - "Still water runs deep"
  | 'meditation_1'       // +1 stone/sec - "The breath between moves"
  | 'meditation_2'       // +2 stones/sec - "The silence speaks"
  | 'meditation_3'       // +5 stones/sec - "The void contains all"
  | 'student_1'          // First student - "One who seeks to learn"
  | 'student_2'          // Second student
  | 'student_3'          // Third student
  | 'dojo_1'             // Small dojo - "A place of learning"
  | 'dojo_2'             // Medium dojo
  | 'dojo_3'             // Large dojo

export interface IdleSource {
  id: IdleSourceId
  name: string
  description: string
  quote: string // Poetic wisdom
  rate: number // Stones per second
  cost: number
  prerequisite?: IdleSourceId
  unlockCondition?: {
    type: 'games_played' | 'wins' | 'total_stones_earned'
    value: number
  }
}

export interface IdleState {
  sources: Set<IdleSourceId>
  lastCollectionTime: number
  totalIdleEarned: number
  sessionIdleEarned: number
}

export interface IdleProgress {
  pendingStones: number
  elapsedSeconds: number
  ratePerSecond: number
}
