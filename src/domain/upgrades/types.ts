export type UpgradeId =
  | 'move_counter_1'  // 5 → 10 moves
  | 'move_counter_2'  // 10 → 15 moves
  | 'move_counter_3'  // 15 → 20 moves
  | 'board_size_5x5'
  | 'board_size_7x7'
  | 'board_size_9x9'
  | 'board_size_13x13'
  | 'board_size_19x19'
  | 'win_bonus_1'     // +50% stones from wins
  | 'win_bonus_2'     // +100% stones from wins
  | 'capture_bonus'   // +2 stones per capture

export interface Upgrade {
  id: UpgradeId
  name: string
  description: string
  cost: number
  effect: UpgradeEffect
  prerequisite?: UpgradeId
}

export type UpgradeEffect =
  | { type: 'move_count'; value: number }
  | { type: 'board_size'; value: number }
  | { type: 'ai_difficulty'; value: number }
  | { type: 'win_bonus'; value: number }     // percentage bonus
  | { type: 'capture_bonus'; value: number } // stones per capture

export interface UpgradeState {
  purchased: Set<UpgradeId>
  available: UpgradeId[]
}
