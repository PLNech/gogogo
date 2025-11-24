export type SchoolId = 'territory' | 'influence' | 'combat'

export interface School {
  id: SchoolId
  name: string
  title: string
  description: string
  philosophy: string
  bonuses: SchoolBonus[]
  skillTree: SkillNode[]
}

export interface SchoolBonus {
  type: 'territory_bonus' | 'influence_bonus' | 'capture_bonus' | 'komi_adjustment'
  value: number
  description: string
}

export interface SkillNode {
  id: string
  title: string
  tooltip: {
    content: string
    senseiUrl?: string // Link to Senseis Library article
  }
  children: SkillNode[]
  optional?: boolean
}

export interface SchoolState {
  selectedSchool: SchoolId | null
  skillPoints: number
  unlockedSkills: Set<string>
}

// Lesson/Puzzle data from Senseis Library
export interface Lesson {
  id: string
  title: string
  category: 'joseki' | 'fuseki' | 'tesuji' | 'tsumego' | 'endgame'
  school: SchoolId
  senseiUrl: string
  difficulty: 1 | 2 | 3 | 4 | 5
  description: string
  skillPoints: number
}
