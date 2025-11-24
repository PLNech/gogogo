# GoGoGo - Project Development Guide

## Project Overview
GoGoGo is an idle/incremental game centered around the ancient game of Go (Baduk/Weiqi). The game combines traditional Go gameplay with incremental mechanics, allowing players to earn currencies through play, unlock larger boards, longer games, and harder AI opponents. Educational content from Senseis.xmp.net and classic Go resources will be integrated to provide learning value.

## Architecture Principles

### Senior Staff Architect Standards
- **Lean MVP First**: Build minimum viable features, validate, then iterate
- **YAGNI (You Aren't Gonna Need It)**: No speculative features or over-engineering
- **Separation of Concerns**: Clear boundaries between layers
- **TDD with Late UI**: Engine and tests first, UI last
- **Clean Architecture**: Domain logic independent of frameworks

### Layer Structure
```
┌─────────────────────────────────────┐
│     UI/Presentation Layer           │  React Components
│     (React, beautiful-skill-tree)   │
├─────────────────────────────────────┤
│     State/Application Layer         │  Zustand (global)
│     (Game State, Currencies)        │  Jotai (atomic)
├─────────────────────────────────────┤
│     Domain/Business Layer           │  Game Logic
│     (Rules, Scoring, Progression)   │  Currency System
├─────────────────────────────────────┤
│     Core/Engine Layer               │  Tenuki (Go rules)
│     (Go Rules, AI, Board)           │  MCTS AI + Web Workers
├─────────────────────────────────────┤
│     Data/Content Layer              │  SGF parsing
│     (SGF, Joseki, Tsumego)          │  Content loading
└─────────────────────────────────────┘
```

## Technology Stack

### Core Technologies
- **Framework**: React 18+ with TypeScript 5+
- **Build Tool**: Vite
- **State Management**:
  - Zustand for global game state (currencies, unlocks, settings)
  - Jotai for atomic rapid-update state (board position, AI thinking)
- **Testing**: Vitest + React Testing Library + vitest-localstorage-mock

### Go Game Engine
- **Rules Engine**: Tenuki (ko, superko, scoring, seki, dead stones, multiple rulesets)
- **Board Rendering**: WGo.js or reactive-goban (evaluate during implementation)
- **SGF Support**: sgf-ts (TypeScript-native SGF parser)
- **AI Engine**:
  - MCTS implementation (evaluate: SethPipho, jsmcts, dsesclei)
  - Simple heuristic AI (random + capture priority + liberty counting)
  - Web Workers for background computation

### Idle Game Mechanics
- **Patterns**: Incremental Game Template (IGT) patterns
- **Big Numbers**: break_eternity.js
- **Skill Tree UI**: beautiful-skill-tree
- **Persistence**: localStorage

### Educational Content Sources
- **Primary**: Senseis.xmp.net (26,490+ pages)
  - Beginner Study Section structure
  - Joseki and Fuseki patterns
  - Tsumego problems
- **Joseki Databases**:
  - OGS godojo-server (Neo4j backend, JSON exports)
  - Kogo's Joseki Dictionary (SGF)
  - Waltheri's pattern search (70,000 pro games)
- **Tsumego**: Multiple databases with 20,000+ problems in SGF format

## Development Methodology

### MVP Phase 1 - Core Engine
**Test First, UI Last**
1. Implement Go rules integration (Tenuki)
   - Unit tests for capture, ko, scoring
   - Test multiple board sizes (9x9, 13x13, 19x19)
2. Implement basic AI
   - Simple random + heuristic AI
   - MCTS with configurable iterations
   - Web Worker integration tests
3. Implement currency earning
   - Game outcome → currency calculation
   - Persistence tests

### MVP Phase 2 - Incremental Mechanics
**Test First, UI Last**
1. Upgrade system
   - Board size unlocks (9x9 → 13x13 → 19x19)
   - Move count increases
   - AI difficulty progression
   - Game speed modifiers
2. Currency system
   - Multiple currency types (TBD: stones, experience, wisdom)
   - BigNumber integration
   - Save/load with localStorage

### MVP Phase 3 - UI Implementation
**After engine validation**
1. Board display
   - Responsive goban rendering
   - Move input and validation feedback
   - Game state visualization
2. Upgrade UI
   - Skill tree display (beautiful-skill-tree)
   - Currency display
   - Purchase/unlock flows
3. Game loop UI
   - New game flow
   - Auto-play options
   - Settings and persistence

### Phase 4+ - Educational Content
**Iterative content integration**
1. Tsumego problem integration
   - SGF parsing and rendering
   - Problem selection and hints
   - Reward integration
2. Joseki learning mode
   - Pattern recognition mini-games
   - Database integration
   - Progress tracking
3. Famous games replay
   - SGF viewer integration
   - Commentary display
   - Learning rewards

## Code Quality Standards

### TypeScript
- Strict mode enabled
- No `any` types without explicit justification
- Prefer interfaces over types for objects
- Use discriminated unions for state variants

### Testing
- Engine layer: 100% coverage target
- Domain layer: 90%+ coverage target
- UI layer: Critical paths tested, avoid implementation details
- Integration tests for key user flows
- Performance tests for MCTS iterations

### File Organization
```
src/
├── core/           # Go engine, AI, rules (no React deps)
│   ├── go/         # Tenuki integration
│   ├── ai/         # MCTS, heuristics, Web Workers
│   └── sgf/        # SGF parsing
├── domain/         # Game logic, currencies, upgrades
│   ├── game/       # Game state, progression
│   ├── currency/   # Currency system
│   └── upgrades/   # Upgrade definitions and logic
├── state/          # Zustand stores, Jotai atoms
├── ui/             # React components
│   ├── board/      # Board rendering components
│   ├── upgrades/   # Skill tree, upgrade UI
│   └── game/       # Game flow components
├── data/           # Content loading, SGF files
└── utils/          # Shared utilities
```

### Performance Budgets
- Initial load: < 100KB gzipped JS
- AI move calculation: < 500ms (adjust MCTS iterations)
- Board rendering: 60fps
- Web Worker communication: < 50ms overhead

## Feature Flags (Post-MVP)
When implementing variant rules or experimental features:
- Feature flag pattern in Zustand store
- A/B testing support
- Easy enable/disable for testing

## Variant Game Modes (Future)
Examples for late-game content:
- "Dumb but Fast": AI plays 2 moves per player move
- "Capture Race": Score based on captures only
- "Territory Master": Pure territory scoring
- "Speed Go": Time pressure mechanics
- Additional creative variants TBD

## Development Commands
```bash
npm run dev          # Vite dev server
npm run build        # Production build
npm run test         # Vitest test runner
npm run test:ui      # Vitest UI mode
npm run test:coverage # Coverage report
npm run lint         # ESLint check
npm run type-check   # TypeScript check
```

## Git Workflow
- Never push to main/develop without explicit request
- Feature branches only: `feature/board-rendering`, `feature/mcts-ai`
- Commit messages: Clear, imperative mood ("Add MCTS worker", "Fix ko rule detection")
- Small, focused commits preferred

## AI Development Notes
- Use `rg` (ripgrep) for fast codebase search
- Use `sg` (ast-grep) for precise AST-based navigation
- Prioritize reading existing code before modifications
- Test first, implement second, UI last

## PLAN.md Tracking (MANDATORY)
- **MUST maintain PLAN.md** with ALL features from complete game design (all 7 milestones)
- Track each feature with status: **DONE** / **TODO** / **LATER** / **DISCUSS**
- PLAN.md is the single source of truth for implementation progress
- Update PLAN.md as features are completed or priorities shift
- PLAN.md ensures the complete vision is documented and nothing is forgotten

## Content Integration Strategy
1. **Phase 1**: Download and parse representative SGF samples
2. **Phase 2**: Implement content loaders (tsumego, joseki, games)
3. **Phase 3**: Build content selection and progression systems
4. **Phase 4**: Integrate rewards with main game progression

## Game Progression Design
**Direction Artistique: Poetic, immersive, educational**

### Milestone 1: Initial Stepping Stone (1A - "Place First Stone")
- **1x1 board**: Place one black stone → instant win → earn 1 Stone
- **Poetic message**: "A journey of ten thousand games begins with a single stone"
- **Board expands to 3x3**: White stone appears (dead), player captures it
- **Reward**: +5 stones, unlock "Play Again" button
- **Purpose**: Introduce core loop, first dopamine hit, poetic tone

### Milestone 2: Opponent Introduction (2B - "After 3 Practice Moves")
- Player places 3 stones on empty boards first (earning 10 stones each)
- **Narrative trigger**: "You've learned placement. Now learn opposition."
- **5x5 board**: AI with simple heuristic (avoid eyes, try to capture)
- **Purpose**: Gradual learning curve, narrative pacing

### Milestone 3: First Upgrade (3B - "Move Counter")
- **Unlock at**: 100 stones
- Games start at **5 moves max**, upgrade to 10 moves
- **Key feature**: End-game board shows **territory gradient visualization** (grey shading based on territory heuristic)
- More moves = see more of the game, learn territory concept, earn more per game
- **Message**: "Patience reveals deeper patterns"
- **Purpose**: Games end quick initially, but show full Go concepts (territory, influence)

### Milestone 4: School Specialization (4A - "Three Philosophies")
- **Unlock at**: 1,000 stones
- **School of Territory (Komi)**: Efficiency, area control
  - Bonuses: +50% territory scoring rewards
  - Unlocks: Joseki training modules
- **School of Influence (Moyo)**: Potential, large-scale thinking
  - Bonuses: +50% capture rewards
  - Unlocks: Fuseki patterns library
- **School of Combat (Sabaki)**: Fighting, tactical play
  - Bonuses: +50% tsumego rewards
  - Unlocks: Life/death puzzles
- **Purpose**: Focus learning, bonus specialization, thematic progression

### Milestone 5: Idle Mechanics (5A + 5C - Hybrid)
- **Unlock at**: 5,000 stones

#### 5A: Auto-Play Training Grounds
- Your AI plays against weak AIs in background
- Earn stones/minute while focusing on harder games
- Watch games like OpenGo.com
- Upgrade: simultaneous games, AI strength

#### 5C: Dojo Students System
- Recruit students who train on old boards/puzzles you've beaten
- Each student earns % of original reward passively
- **Managery idle game feel**
- Upgrade: more students, efficiency, student AI level

### Milestone 6: Puzzle/Lesson Unlocks (6A + 6B + 6C - Hybrid)
- **6A Primary**: Achievement-based unlocks
  - Win 10 games → Unlock Beginner Tsumego Pack 1
  - Capture 100 stones → "Basics of Capture" joseki
  - Play on 9x9 → "Opening Principles" lessons
- **6B Secondary**: Currency purchases
  - "100 Beginner Tsumego" costs 1000 stones
  - "Classic Joseki Library" costs 5000 stones
  - Some packs locked behind achievements
- **6C Bonus**: Contextual discovery
  - Get captured? → Related tsumego unlocks
  - Play joseki? → "You discovered Keima approach!"
  - Organic, contextual learning deep links

### Milestone 7: Late Game (7A - "AI Laboratory")
- **Unlock at**: 100,000 stones
- **Meta-game**: Train multiple AIs with different personalities/strategies
- Send AIs to compete in tournaments while you manage
- **Expose AI hyperparameters**: MCTS iterations, exploration constant, heuristics
- **Experimental AI exploration**: Genetic algorithms, evolutionary strategies
- Strategy shifts to: breeding AIs, optimizing training, tournament management
- **You become**: Tournament director & AI researcher
- **FUNKY meta-game revealing underlying AI mechanics!**

## Currency System
- **Primary**: Stones (earned from games)
- **Secondary** (Post-School Specialization):
  - Territory School: Efficiency Points
  - Influence School: Moyo Potential
  - Combat School: Fighting Spirit

## Key Design Decisions
- ✅ Direction Artistique: Poetic, immersive, educational
- ✅ Quick initial games with territory visualization
- ✅ Three Schools specialization (Territory/Influence/Combat)
- ✅ Hybrid idle mechanics (auto-play + dojo students)
- ✅ Achievement + purchase + discovery unlock system
- ✅ Late game becomes AI management meta-game
- ✅ Scoring → currency includes territory gradient visualization

---

**Remember**: MVP first. Build the engine, validate with tests, then add UI. No feature creep. Clean separation. Fast iteration.
