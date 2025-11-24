import type { Upgrade, UpgradeId } from './types'

export const UPGRADES: Record<UpgradeId, Upgrade> = {
  move_counter_1: {
    id: 'move_counter_1',
    name: 'Extend Game Length I',
    description: 'Games last 10 moves instead of 5. "Patience reveals deeper patterns."',
    cost: 100,
    effect: { type: 'move_count', value: 10 },
  },
  move_counter_2: {
    id: 'move_counter_2',
    name: 'Extend Game Length II',
    description: 'Games last 15 moves. More time to develop strategy.',
    cost: 500,
    effect: { type: 'move_count', value: 15 },
    prerequisite: 'move_counter_1',
  },
  move_counter_3: {
    id: 'move_counter_3',
    name: 'Extend Game Length III',
    description: 'Games last 20 moves. Master the full arc of play.',
    cost: 2000,
    effect: { type: 'move_count', value: 20 },
    prerequisite: 'move_counter_2',
  },
  board_size_5x5: {
    id: 'board_size_5x5',
    name: 'Unlock 5x5 Board',
    description: 'Access to 5x5 board size for practice.',
    cost: 50,
    effect: { type: 'board_size', value: 5 },
  },
  board_size_7x7: {
    id: 'board_size_7x7',
    name: 'Unlock 7x7 Board',
    description: 'Access to 7x7 board size.',
    cost: 200,
    effect: { type: 'board_size', value: 7 },
    prerequisite: 'board_size_5x5',
  },
  board_size_9x9: {
    id: 'board_size_9x9',
    name: 'Unlock 9x9 Board',
    description: 'The traditional small board.',
    cost: 1000,
    effect: { type: 'board_size', value: 9 },
    prerequisite: 'board_size_7x7',
  },
  board_size_13x13: {
    id: 'board_size_13x13',
    name: 'Unlock 13x13 Board',
    description: 'A serious challenge awaits.',
    cost: 5000,
    effect: { type: 'board_size', value: 13 },
    prerequisite: 'board_size_9x9',
  },
  board_size_19x19: {
    id: 'board_size_19x19',
    name: 'Unlock 19x19 Board',
    description: 'The full board. The traditional battlefield.',
    cost: 20000,
    effect: { type: 'board_size', value: 19 },
    prerequisite: 'board_size_13x13',
  },
  win_bonus_1: {
    id: 'win_bonus_1',
    name: 'Victory Spoils I',
    description: '+50% stones from victories. "The victor writes history."',
    cost: 300,
    effect: { type: 'win_bonus', value: 50 },
  },
  win_bonus_2: {
    id: 'win_bonus_2',
    name: 'Victory Spoils II',
    description: '+100% stones from victories. Double rewards for triumph.',
    cost: 1500,
    effect: { type: 'win_bonus', value: 100 },
    prerequisite: 'win_bonus_1',
  },
  capture_bonus: {
    id: 'capture_bonus',
    name: 'Predator Instinct',
    description: '+2 stones per captured stone. "Hunt with precision."',
    cost: 500,
    effect: { type: 'capture_bonus', value: 2 },
  },
}

export function getAvailableUpgrades(purchased: Set<UpgradeId>): UpgradeId[] {
  return (Object.keys(UPGRADES) as UpgradeId[]).filter(id => {
    const upgrade = UPGRADES[id]!

    // Already purchased
    if (purchased.has(id)) return false

    // Check prerequisite
    if (upgrade.prerequisite && !purchased.has(upgrade.prerequisite)) {
      return false
    }

    return true
  })
}

export function getCurrentMoveCount(purchased: Set<UpgradeId>): number {
  if (purchased.has('move_counter_3')) return 20
  if (purchased.has('move_counter_2')) return 15
  if (purchased.has('move_counter_1')) return 10
  return 5 // Default
}

export function getUnlockedBoardSizes(purchased: Set<UpgradeId>): number[] {
  const sizes = [3] // Always have 3x3

  if (purchased.has('board_size_5x5')) sizes.push(5)
  if (purchased.has('board_size_7x7')) sizes.push(7)
  if (purchased.has('board_size_9x9')) sizes.push(9)
  if (purchased.has('board_size_13x13')) sizes.push(13)
  if (purchased.has('board_size_19x19')) sizes.push(19)

  return sizes
}
