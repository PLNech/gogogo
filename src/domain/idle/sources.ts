import type { IdleSource, IdleSourceId } from './types'

/**
 * "Patience yields stones. Time teaches all players."
 */

export const IDLE_SOURCES: Record<IdleSourceId, IdleSource> = {
  contemplation: {
    id: 'contemplation',
    name: 'Contemplation',
    description: 'The first lesson: stillness. Watch stones gather like morning dew.',
    quote: 'Still water runs deep. Empty board holds infinite potential.',
    rate: 0.1, // 0.1 stones/sec = 6 stones/min
    cost: 50,
  },
  meditation_1: {
    id: 'meditation_1',
    name: 'Deep Meditation',
    description: 'Close your eyes. See the board within. Feel the stones accumulate.',
    quote: 'The breath between moves. The space between thoughts.',
    rate: 1, // 1 stone/sec = 60 stones/min
    cost: 200,
    prerequisite: 'contemplation',
  },
  meditation_2: {
    id: 'meditation_2',
    name: 'Transcendent Meditation',
    description: 'You and the board are one. Stones flow like water.',
    quote: 'When the student is ready, the stones appear.',
    rate: 2, // 2 stones/sec
    cost: 1000,
    prerequisite: 'meditation_1',
  },
  meditation_3: {
    id: 'meditation_3',
    name: 'Eternal Stillness',
    description: 'Beyond thought. Beyond time. Only Go remains.',
    quote: 'The void contains all moves. Silence holds every game.',
    rate: 5, // 5 stones/sec
    cost: 5000,
    prerequisite: 'meditation_2',
    unlockCondition: {
      type: 'games_played',
      value: 50,
    },
  },
  student_1: {
    id: 'student_1',
    name: 'First Student',
    description: 'A seeker arrives at your door. They learn. You earn.',
    quote: 'To teach is to learn twice. Share the path.',
    rate: 0.5, // 0.5 stones/sec
    cost: 500,
    unlockCondition: {
      type: 'wins',
      value: 5,
    },
  },
  student_2: {
    id: 'student_2',
    name: 'Second Student',
    description: 'Another soul drawn to the stones. Your dojo grows.',
    quote: 'One candle lights another without diminishing.',
    rate: 0.75,
    cost: 2000,
    prerequisite: 'student_1',
  },
  student_3: {
    id: 'student_3',
    name: 'Third Student',
    description: 'Word spreads of your wisdom. Three now walk the path.',
    quote: 'The teacher learns from the student who learns from the teacher.',
    rate: 1.5,
    cost: 8000,
    prerequisite: 'student_2',
  },
  dojo_1: {
    id: 'dojo_1',
    name: 'Small Dojo',
    description: 'Four walls. Tatami mats. Goban in the center. A place of learning.',
    quote: 'A thousand mile journey begins with a single dojo.',
    rate: 3,
    cost: 10000,
    unlockCondition: {
      type: 'wins',
      value: 20,
    },
  },
  dojo_2: {
    id: 'dojo_2',
    name: 'Established Dojo',
    description: 'Your reputation grows. Students fill the rooms. Stones flow freely.',
    quote: 'Many hands, one Go. Many students, one wisdom.',
    rate: 8,
    cost: 50000,
    prerequisite: 'dojo_1',
    unlockCondition: {
      type: 'total_stones_earned',
      value: 100000,
    },
  },
  dojo_3: {
    id: 'dojo_3',
    name: 'Master Dojo',
    description: 'Pilgrims come from distant lands. Your name echoes in Go circles.',
    quote: 'When the master is ready, the students will teach.',
    rate: 20,
    cost: 250000,
    prerequisite: 'dojo_2',
    unlockCondition: {
      type: 'games_played',
      value: 200,
    },
  },
}

export function getAvailableIdleSources(
  purchased: Set<IdleSourceId>,
  stats: { gamesPlayed: number; wins: number; totalEarned: number }
): IdleSourceId[] {
  return (Object.keys(IDLE_SOURCES) as IdleSourceId[]).filter(id => {
    const source = IDLE_SOURCES[id]!

    // Already purchased
    if (purchased.has(id)) return false

    // Check prerequisite
    if (source.prerequisite && !purchased.has(source.prerequisite)) {
      return false
    }

    // Check unlock condition
    if (source.unlockCondition) {
      const { type, value } = source.unlockCondition
      if (type === 'games_played' && stats.gamesPlayed < value) return false
      if (type === 'wins' && stats.wins < value) return false
      if (type === 'total_stones_earned' && stats.totalEarned < value) return false
    }

    return true
  })
}

export function calculateIdleRate(purchased: Set<IdleSourceId>): number {
  let rate = 0
  for (const id of purchased) {
    const source = IDLE_SOURCES[id]
    if (source) {
      rate += source.rate
    }
  }
  return rate
}
