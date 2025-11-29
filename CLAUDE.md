# GoGoGo - Project Development Guide

## Project Overview
GoGoGo is an idle/incremental game centered around the ancient game of Go (Baduk/Weiqi). The game combines traditional Go gameplay with incremental mechanics, allowing players to earn currencies through play, unlock larger boards, longer games, and harder AI opponents. Educational content from Senseis.xmp.net and classic Go resources will be integrated to provide learning value.

## Architecture Principles

### CRITICAL: Test-Driven Development (TDD) & Separation of Concerns (SOC)
**ALL development MUST follow TDD methodology:**
1. **Write tests FIRST** - Never write implementation before tests
2. **Run tests to see them fail** - Verify test is testing what you think
3. **Write minimal implementation** - Make tests pass with simplest code
4. **Refactor** - Clean up while keeping tests green
5. **Repeat** - Every feature, every change follows this cycle

**Separation of Concerns is NON-NEGOTIABLE:**
- **Core/Engine Layer** (`src/core/`): ZERO React dependencies, pure TypeScript
- **Domain Layer** (`src/domain/`): Business logic, NO UI concerns
- **State Layer** (`src/state/`): State management bridges, minimal logic
- **UI Layer** (`src/ui/`): React components only, NO business logic

### Senior Staff Architect Standards
- **Lean MVP First**: Build minimum viable features, validate, then iterate
- **YAGNI (You Aren't Gonna Need It)**: No speculative features or over-engineering
- **Clean Architecture**: Domain logic independent of frameworks
- **Make it work, make it right, make it fast**: In that order

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

### CURRENT FOCUS: Neural Network Go AI Training
**Goal: Train AlphaZero-style neural network AI using self-play**

**CRITICAL: Follow PLAN.md for all neural training work**
- PLAN.md contains the detailed implementation roadmap
- Update PLAN.md status as work progresses
- All training decisions documented there

**Two Modes:**
1. **Training Mode** (Developer): Self-play → train → checkpoint → evaluate → repeat
2. **Playing Mode** (Users): Pre-trained models exported to TensorFlow.js for browser

**Hardware:** RTX 4080 12GB - sufficient for 9x9 training, iterative 19x19

**Key Directories:**
```
training/           # Python training code (PyTorch)
├── models/         # Network architectures
├── mcts/           # Neural MCTS implementation
├── selfplay/       # Game generation
├── train/          # Training loop
└── checkpoints/    # Saved model states

src/core/ai/        # TypeScript inference (browser)
├── neural/         # TensorFlow.js inference
└── models/         # Exported ONNX/TFJS models
```

**Training Loop:**
```
1. Self-play games (N games with current model)
2. Store (board, policy, value) tuples
3. Train network on replay buffer
4. Checkpoint model
5. Evaluate vs previous checkpoint
6. If stronger → promote, else → continue training
7. Export to TFJS periodically for browser testing
```

**Existing Heuristics → Features:**
- `src/core/ai/policy.ts` heuristics can inform feature engineering
- Not used directly in neural network - superseded by learned features

### MANDATORY: Blog Post for Every Deliverable
**CRITICAL RULE**: Each phase deliverable MUST include a blog post entry

**Blog Post Requirements**:
1. **FORMAT**: MARKDOWN (.md) files ONLY - NEVER HTML
   - File format: `blog/posts/YYYY-MM-DD-title.md`
   - Use proper Markdown syntax (headers, code blocks, quotes)
2. **LENGTH**: SHORT. CONCISE. POETIC.
   - More white space, less text
   - Short paragraphs (1-3 sentences max)
   - Let code breathe
   - No long explanations - show, don't tell
3. **Tone**: Subtle, evasive, go-inspired, evocative, spare prose
4. **Structure**:
   - Philosophical opening (go proverb or haiku)
   - Brief system explanation
   - 1-2 key code samples (small, focused)
   - Brief closing reflection
5. **Purpose**: Grounds delivery in functional requirements completion

**Blog Infrastructure**:
- Minimal GitHub Pages blog
- Posts are Markdown: `blog/posts/YYYY-MM-DD-title.md`
- Landing page links to posts
- Simple, elegant theme that honors Go's aesthetic

**DATES**: Always use current/recent dates relative to project context

**Phase 1: Simplify & Stabilize**
1. **Audit existing AI code**
   - Identify what works, what doesn't
   - Consolidate multiple AI implementations
   - Fix failing tests
2. **Define single target strength**
   - Pick ONE difficulty level to perfect
   - Focus on correctness over variety
   - Establish measurable benchmarks

**Phase 2: TDD-Driven AI Improvement**
1. **Test-first feature extraction**
   - Write tests for board state representation
   - Implement basic feature extraction (liberties, captures, territory)
   - Test edge cases (ko, seki, life/death)
2. **Test-first heuristic evaluation**
   - Write tests for position evaluation
   - Implement clean, testable heuristics
   - Validate against known positions
3. **Test-first MCTS integration**
   - Write tests for MCTS search
   - Implement clean MCTS with proper UCT
   - Test convergence and move selection

**Phase 3: Visualization & Analysis**
1. **CLI-based game viewer**
   - ASCII board representation
   - Move-by-move playback
   - Statistics and analysis
2. **Self-play evaluation**
   - Automated game generation
   - Win rate tracking
   - Performance benchmarking

### MVP Phase 1 - Core Engine (LATER)
**Test First, UI Last**
1. Implement Go rules integration
   - Unit tests for capture, ko, scoring
   - Test multiple board sizes (9x9, 13x13, 19x19)
2. Implement currency earning
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

### CRITICAL: Commit Early, Commit Often
- **Commit after every logical unit of work** - don't batch up changes
- **Push periodically** - SSH is configured, push to share progress
- Commit when:
  - Tests go from red to green
  - A feature is complete (even if small)
  - Files are created/moved/deleted
  - After writing blog posts
  - Before starting risky refactoring

### Branch Strategy
- Never push to main/develop without explicit request
- Feature branches for new work: `feature/board-rendering`, `feature/mcts-ai`
- Current work can stay on master if it's iterative development

### Commit Messages
- Clear, imperative mood: "Add MCTS worker", "Fix ko rule detection"
- Start with verb: Add, Fix, Update, Remove, Refactor
- Reference what changed, not why (why goes in comments/docs)
- Small, focused commits preferred

### Example Session
```bash
# After fixing tests
git add src/core/ai/simpleAI.ts
git commit -m "Fix missing imports in simpleAI.ts"

# After creating fixtures
git add src/test/fixtures/ src/test/utils/
git commit -m "Add test fixtures and boardSetup utilities"

# After blog post
git add blog/posts/2025-11-28-first-stone.md
git commit -m "Add 'First Stone' blog post"

# Push periodically (every 3-5 commits or end of session)
git push
```

## AI Development Notes
- Use `rg` (ripgrep) for fast codebase search
- Use `sg` (ast-grep) for precise AST-based navigation
- Prioritize reading existing code before modifications
- Test first, implement second, UI last
- **ALWAYS use Poetry** for Python package management (never pip/venv directly)

## Meta: Capturing Learnings
**CRITICAL**: When you make an error or realize a mistake:
1. **Immediately update CLAUDE.md** with guidance to prevent recurrence
2. **Document the pattern**, not just the specific instance
3. **Add it to the relevant section** (or create new section if needed)
4. **Make it actionable** - clear rules, not vague suggestions

**Examples of learnings to capture**:
- File format mistakes (HTML vs Markdown)
- Architectural violations (UI in core/)
- Test patterns that work/don't work
- Common pitfalls in the domain (Go rules edge cases)
- Workflow improvements discovered

**This file is living documentation** - it should grow with the project

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
