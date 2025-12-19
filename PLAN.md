# GoGoGo - Complete Implementation Plan

**Status Legend:**
- **DONE**: Implemented and tested
- **TODO**: Current sprint, actively working on
- **LATER**: Planned for future sprints
- **DISCUSS**: Needs design decision or clarification

---

## Foundation & Infrastructure

### Project Setup
- [x] DONE - Initialize Vite + React + TypeScript
- [x] DONE - Setup Vitest + React Testing Library
- [x] DONE - Configure vitest-localstorage-mock
- [x] DONE - Setup ESLint + TypeScript strict mode
- [x] DONE - Create directory structure (core/domain/state/ui/data/utils)
- [ ] LATER - Setup Git hooks and commit conventions
- [ ] LATER - Setup CI/CD pipeline
- [ ] LATER - Setup Vercel/Netlify deployment

### Development Tools
- [x] DONE - Research Go libraries and AI engines
- [x] DONE - Research idle game patterns
- [x] DONE - Research state management solutions
- [x] DONE - Create CLAUDE.md architecture guide
- [x] DONE - Create PLAN.md tracking document

---

## Milestone 1: Initial Stepping Stone (1A - "Place First Stone")

### Core Engine - 1x1 Board
- [x] DONE - Implement 1x1 board representation
- [x] DONE - Stone placement on 1x1
- [x] DONE - Auto-win detection on 1x1
- [x] DONE - Tests for 1x1 board logic

### Core Engine - 3x3 Board
- [x] DONE - Implement 3x3 board representation
- [x] DONE - Stone placement validation (no overlap)
- [x] DONE - Capture detection (liberty counting)
- [x] DONE - Remove captured stones
- [x] DONE - Tests for capture mechanics

### Currency System
- [x] DONE - Stone currency data model
- [x] DONE - Earn stones from game outcomes
- [x] DONE - Currency persistence to localStorage
- [x] DONE - Tests for currency system

### UI - Milestone 1 Flow
- [x] DONE - 1x1 board UI with click handler
- [x] DONE - Poetic message display system
- [x] DONE - "A journey of ten thousand games..." message
- [x] DONE - Board expansion animation (1x1 → 3x3)
- [x] DONE - Pre-placed dead white stone on 3x3
- [x] DONE - Capture animation/feedback
- [x] DONE - "+5 stones" reward notification
- [x] DONE - "Play Again" button unlock (auto-advance)
- [x] DONE - Currency display (Stones counter)

### Testing & Polish
- [ ] LATER - End-to-end test: Complete M1 flow
- [x] DONE - Visual polish: Stone rendering
- [ ] DISCUSS - Sound effects (optional)

---

## Milestone 2: Opponent Introduction (2B - "After 3 Practice Moves")

### Core Engine - 5x5 Board
- [x] DONE - Implement 5x5 board representation
- [x] DONE - Extend capture logic to 5x5
- [x] DONE - Basic win condition (most territory)
- [x] DONE - Tests for 5x5 gameplay

### Territory Calculation
- [x] DONE - Simple territory heuristic algorithm
- [x] DONE - Territory ownership calculation (black/white/neutral)
- [ ] LATER - Territory gradient data (for visualization)
- [x] DONE - Tests for territory calculation

### Simple AI
- [x] DONE - AI interface/contract
- [x] DONE - Random move generator
- [x] DONE - Basic heuristic: avoid filling own eyes
- [x] DONE - Basic heuristic: prefer captures
- [x] DONE - AI move selection logic
- [x] DONE - Tests for AI behavior

### Game Loop
- [x] DONE - Turn-based game state machine
- [x] DONE - Player move → AI move → repeat
- [x] DONE - Game end detection
- [x] DONE - Score calculation and winner determination
- [x] DONE - Reward calculation (stones from wins)

### Progression State
- [x] DONE - Track games played
- [x] DONE - Track practice moves completed
- [x] DONE - Unlock trigger: 3 practice moves → AI opponent
- [x] DONE - Persistence of progression state

### UI - Milestone 2 Flow
- [x] DONE - Empty board practice mode (3 games)
- [x] DONE - "You've learned placement. Now learn opposition." message
- [x] DONE - 5x5 board UI
- [x] DONE - Turn indicator (your turn/AI thinking)
- [x] DONE - AI move visualization
- [x] DONE - Game end screen with winner
- [ ] LATER - Territory gradient visualization (grey shading)
- [x] DONE - Reward notification (+X stones)
- [x] DONE - Play again flow (auto-advance)

### Testing & Polish
- [ ] LATER - End-to-end test: Complete M2 flow
- [x] DONE - AI move delay/animation
- [ ] LATER - Territory visualization polish

---

## Milestone 3: First Upgrade (3B - "Move Counter")

### Upgrade System Foundation
- [ ] LATER - Upgrade data model (id, cost, effect, unlocked)
- [ ] LATER - Purchase upgrade logic
- [ ] LATER - Upgrade persistence
- [ ] LATER - Tests for upgrade system

### Move Counter Upgrade
- [ ] LATER - "Extend Game Length" upgrade definition
- [ ] LATER - Cost: 100 stones
- [ ] LATER - Effect: 5 moves → 10 moves per game
- [ ] LATER - Move counter enforcement in game loop
- [ ] LATER - "Patience reveals deeper patterns" message

### Enhanced Territory Visualization
- [ ] LATER - Improved territory gradient rendering
- [ ] LATER - Color intensity based on territory strength
- [ ] LATER - Territory animation on game end
- [ ] LATER - Territory explanation tooltips

### Board Size Progression (Future Upgrades)
- [ ] LATER - 7x7 board implementation
- [ ] LATER - 9x9 board implementation
- [ ] LATER - 13x13 board implementation
- [ ] LATER - 19x19 board implementation
- [ ] LATER - Board size unlock upgrades

### UI - Upgrade System
- [ ] LATER - Upgrades panel/menu
- [ ] LATER - Upgrade cards with cost/description
- [ ] LATER - Purchase button and affordability check
- [ ] LATER - Purchase confirmation feedback
- [ ] LATER - Locked vs unlocked upgrade states

---

## Milestone 4: School Specialization (4A - "Three Philosophies")

### School System Foundation
- [ ] LATER - School data model (Territory/Influence/Combat)
- [ ] LATER - School selection (one-time choice)
- [ ] LATER - School bonuses system
- [ ] LATER - School persistence
- [ ] LATER - Tests for school system

### School of Territory (Komi)
- [ ] LATER - +50% territory scoring rewards bonus
- [ ] LATER - Unlock joseki training modules
- [ ] LATER - Territory-focused upgrade tree
- [ ] LATER - Efficiency Points currency

### School of Influence (Moyo)
- [ ] LATER - +50% capture rewards bonus
- [ ] LATER - Unlock fuseki patterns library
- [ ] LATER - Influence-focused upgrade tree
- [ ] LATER - Moyo Potential currency

### School of Combat (Sabaki)
- [ ] LATER - +50% tsumego rewards bonus
- [ ] LATER - Unlock life/death puzzles
- [ ] LATER - Combat-focused upgrade tree
- [ ] LATER - Fighting Spirit currency

### UI - School Selection
- [ ] LATER - School introduction at 1000 stones
- [ ] LATER - Three school cards with descriptions
- [ ] LATER - School selection modal/screen
- [ ] LATER - Confirmation: "This choice is permanent"
- [ ] LATER - School badge/indicator in UI
- [ ] LATER - School-specific UI theming (optional)

---

## Milestone 5: Idle Mechanics (5A + 5C - Hybrid)

### 5A: Auto-Play Training Grounds

#### Background Game System
- [ ] LATER - Background game data model
- [ ] LATER - Run AI vs AI games in background
- [ ] LATER - Game speed multiplier for background games
- [ ] LATER - Background game result calculation
- [ ] LATER - Passive income from background games
- [ ] LATER - Tests for background game system

#### Training Grounds UI
- [ ] LATER - "Unlock Training Grounds" upgrade (5000 stones)
- [ ] LATER - Background game visualization (like OpenGo.com)
- [ ] LATER - Simultaneous games counter
- [ ] LATER - Stones/minute display
- [ ] LATER - Watch game feature
- [ ] LATER - Speed controls

#### Training Grounds Upgrades
- [ ] LATER - Increase simultaneous games (1 → 2 → 3 → 5 → 10)
- [ ] LATER - Increase AI strength (better rewards)
- [ ] LATER - Increase game speed
- [ ] LATER - Auto-restart games

### 5C: Dojo Students System

#### Student System
- [ ] LATER - Student data model
- [ ] LATER - Student assignment to puzzles/boards
- [ ] LATER - Student earning calculation (% of original reward)
- [ ] LATER - Student efficiency and AI level
- [ ] LATER - Tests for student system

#### Dojo UI
- [ ] LATER - "Recruit Students" unlock (5000 stones)
- [ ] LATER - Student roster/management UI
- [ ] LATER - Assign students to content
- [ ] LATER - Student progress visualization
- [ ] LATER - Student earnings display
- [ ] LATER - Managery idle game feel

#### Student Upgrades
- [ ] LATER - Recruit more students
- [ ] LATER - Increase student efficiency
- [ ] LATER - Increase student AI level
- [ ] LATER - Auto-assign students

---

## Milestone 6: Puzzle/Lesson Unlocks (6A + 6B + 6C - Hybrid)

### Content Loading Infrastructure
- [ ] LATER - SGF parser integration (sgf-ts)
- [ ] LATER - Content data models (Tsumego, Joseki, Game)
- [ ] LATER - Content loader utilities
- [ ] LATER - Content pack definitions
- [ ] LATER - Tests for content loading

### 6A: Achievement-Based Unlocks

#### Achievement System
- [ ] LATER - Achievement data model
- [ ] LATER - Achievement tracking (games won, stones captured, etc)
- [ ] LATER - Achievement unlock conditions
- [ ] LATER - Achievement rewards
- [ ] LATER - Tests for achievement system

#### Beginner Achievements
- [ ] LATER - "Win 10 games" → Unlock Beginner Tsumego Pack 1
- [ ] LATER - "Capture 100 stones" → Unlock "Basics of Capture" joseki
- [ ] LATER - "Play on 9x9" → Unlock "Opening Principles" lessons
- [ ] LATER - More achievement definitions

#### Achievement UI
- [ ] LATER - Achievement list/tracker
- [ ] LATER - Progress bars for achievements
- [ ] LATER - Achievement unlock notifications
- [ ] LATER - Reward claim flow

### 6B: Currency Purchase System

#### Content Shop
- [ ] LATER - Shop data model
- [ ] LATER - Content pack purchase logic
- [ ] LATER - Purchase validation (cost, prerequisites)
- [ ] LATER - Tests for shop system

#### Content Packs
- [ ] LATER - "100 Beginner Tsumego" pack (1000 stones)
- [ ] LATER - "Classic Joseki Library" pack (5000 stones)
- [ ] LATER - More content pack definitions
- [ ] LATER - Some packs locked behind achievements

#### Shop UI
- [ ] LATER - Content shop screen
- [ ] LATER - Pack cards with cost/description/preview
- [ ] LATER - Purchase button with affordability
- [ ] LATER - Owned/locked states
- [ ] LATER - Content library browser

### 6C: Contextual Discovery

#### Discovery System
- [ ] LATER - Game event detection (captured, played joseki, lost group)
- [ ] LATER - Event → content mapping
- [ ] LATER - Discovery unlock logic
- [ ] LATER - "You discovered X!" notifications
- [ ] LATER - Tests for discovery system

#### Discovery Events
- [ ] LATER - Get captured → Related tsumego unlocks
- [ ] LATER - Play joseki sequence → "You discovered Keima approach!"
- [ ] LATER - Lose group to atari → Life/death lesson unlocks
- [ ] LATER - More discovery patterns

### Educational Content Modes

#### Tsumego (Life/Death Puzzles)
- [ ] LATER - Tsumego game mode
- [ ] LATER - Load tsumego from SGF
- [ ] LATER - Solution validation
- [ ] LATER - Hint system
- [ ] LATER - Tsumego rewards
- [ ] LATER - Tsumego completion tracking

#### Joseki Training
- [ ] LATER - Joseki practice mode
- [ ] LATER - Load joseki from database
- [ ] LATER - Pattern recognition challenges
- [ ] LATER - Joseki variations
- [ ] LATER - Joseki rewards
- [ ] LATER - Joseki library progress

#### Famous Games Replay
- [ ] LATER - SGF replay mode
- [ ] LATER - Step through moves
- [ ] LATER - Commentary display (if available in SGF)
- [ ] LATER - Branch exploration
- [ ] LATER - Learning rewards for watching games

---

## Milestone 7: Late Game (7A - "AI Laboratory")

### AI Laboratory System
- [ ] LATER - AI personality data model
- [ ] LATER - AI hyperparameter configuration (MCTS iterations, exploration constant, heuristics)
- [ ] LATER - AI training system
- [ ] LATER - AI performance tracking
- [ ] LATER - Tests for AI lab system

### AI Breeding & Evolution
- [ ] LATER - Genetic algorithm framework
- [ ] LATER - AI crossover/mutation operations
- [ ] LATER - Evolutionary strategies
- [ ] LATER - Fitness evaluation
- [ ] LATER - Generation tracking

### Tournament System
- [ ] LATER - Tournament data model
- [ ] LATER - Tournament bracket generation
- [ ] LATER - AI vs AI tournament matches
- [ ] LATER - Tournament result calculation
- [ ] LATER - Tournament rewards (prestige, stones)
- [ ] LATER - Tests for tournament system

### AI Lab UI
- [ ] LATER - "Unlock AI Laboratory" at 100,000 stones
- [ ] LATER - AI roster/management screen
- [ ] LATER - AI hyperparameter editor
- [ ] LATER - AI personality presets
- [ ] LATER - Training interface
- [ ] LATER - Breeding/evolution interface
- [ ] LATER - Tournament scheduling UI
- [ ] LATER - Tournament results/leaderboards
- [ ] LATER - Meta-game statistics and analytics

### Manager Mode
- [ ] LATER - Shift from playing to managing
- [ ] LATER - Resource allocation for AI training
- [ ] LATER - Strategic decision-making
- [ ] LATER - Prestige system (reset for bonuses)

---

## Advanced Go Engine Features

### Rule Variants
- [ ] LATER - Japanese rules (territory scoring)
- [ ] LATER - Chinese rules (area scoring)
- [ ] LATER - Ko rule implementation
- [ ] LATER - Superko detection
- [ ] LATER - Komi adjustment
- [ ] LATER - Handicap stones

### Advanced Game Features
- [ ] LATER - Dead stone marking
- [ ] LATER - Seki detection
- [ ] LATER - Scoring with prisoners
- [ ] LATER - Pass moves
- [ ] LATER - Resignation
- [ ] LATER - Time controls (optional)

### MCTS AI Implementation
- [ ] LATER - Evaluate MCTS libraries (SethPipho, jsmcts, dsesclei)
- [ ] LATER - Integrate or implement MCTS
- [ ] LATER - UCT (Upper Confidence Bounds for Trees)
- [ ] LATER - Playout policies
- [ ] LATER - Tree search optimization
- [ ] LATER - Web Worker integration for MCTS
- [ ] LATER - AI difficulty scaling via iterations

### Advanced Heuristics
- [ ] LATER - Liberty maximization
- [ ] LATER - Capture urgency
- [ ] LATER - Connection heuristics
- [ ] LATER - Eye formation heuristics
- [ ] LATER - Ladder detection
- [ ] LATER - Territory estimation during play

---

## Game Modes & Variants (Post-MVP)

### Variant Modes
- [ ] LATER - "Dumb but Fast": AI plays 2 moves per player move
- [ ] LATER - "Capture Race": Score based on captures only
- [ ] LATER - "Territory Master": Pure territory scoring
- [ ] LATER - "Speed Go": Time pressure mechanics
- [ ] LATER - More creative variants

### Challenge Modes
- [ ] LATER - Daily challenges
- [ ] LATER - Weekly tournaments
- [ ] LATER - Leaderboards
- [ ] LATER - Achievement badges

---

## External Library Integration

### Go Libraries
- [ ] LATER - Evaluate Tenuki for rules engine
- [ ] LATER - Evaluate WGo.js for board rendering
- [ ] LATER - Evaluate reactive-goban
- [ ] LATER - Integrate chosen library or keep custom implementation
- [ ] LATER - sgf-ts integration for SGF parsing

### Idle Game Libraries
- [ ] LATER - Evaluate Incremental Game Template (IGT)
- [ ] LATER - break_eternity.js for big numbers
- [ ] LATER - beautiful-skill-tree for upgrade UI
- [ ] LATER - Integrate chosen libraries

### State Management (Already Chosen)
- [x] DONE - Research Zustand vs Jotai
- [ ] TODO - Implement Zustand for global state
- [ ] LATER - Implement Jotai for atomic state (if needed)

---

## Polish & Production

### Visual Design
- [ ] LATER - Design system / color palette
- [ ] LATER - Responsive layout for mobile
- [ ] LATER - Animations and transitions
- [ ] LATER - Board stone aesthetics
- [ ] LATER - Territory visualization aesthetics
- [ ] LATER - Loading states and skeletons

### Audio (Optional)
- [ ] DISCUSS - Stone placement sound
- [ ] DISCUSS - Capture sound
- [ ] DISCUSS - Victory/defeat sounds
- [ ] DISCUSS - Background music
- [ ] DISCUSS - Audio settings (mute, volume)

### Accessibility
- [ ] LATER - Keyboard navigation
- [ ] LATER - Screen reader support
- [ ] LATER - High contrast mode
- [ ] LATER - Reduced motion support
- [ ] LATER - Internationalization (i18n)

### Performance
- [ ] LATER - Code splitting
- [ ] LATER - Lazy loading for content
- [ ] LATER - Service worker for offline play
- [ ] LATER - Performance monitoring
- [ ] LATER - Bundle size optimization

### Analytics & Feedback
- [ ] LATER - Usage analytics (privacy-respecting)
- [ ] LATER - Error tracking
- [ ] LATER - User feedback system
- [ ] LATER - A/B testing framework

### Documentation
- [ ] LATER - User guide / tutorial
- [ ] LATER - Developer documentation
- [ ] LATER - API documentation
- [ ] LATER - Contributing guide

---

## CURRENT SPRINT: Neural Network Go AI Training

**Focus**: Train AlphaZero-style neural network using self-play on RTX 4080

**Hardware**: RTX 4080 12GB VRAM, CUDA 12.5

---

## 🚨 CURRENT STATE (2025-12-20)

**Status**: Instinct benchmark complete (1449 positions). Model baseline 1.3%. Implementing adaptive curriculum.

### Session Summary (2025-12-20)

**Instinct Benchmark Created:**
- 1449 positions across 8 instincts × 5 board sizes
- Categories: capture, escape, connect, cut, extend, block, atari, defend
- Model baseline: **1.3%** overall (0% on captures!)

**New Approach: Adaptive Instinct Curriculum**

The model can't see captures because **nothing tells it captures matter during training**.
Self-play with random init → no captures in data → model never learns.

**Solution: Wire instincts directly into training loss**

```
Loss = L_policy + L_value + λ(t) × L_instinct

Where:
  λ(t) = λ₀ × (1 - instinct_accuracy)

As model masters instincts → λ → 0 → pure RL takes over
```

**Implementation:**
1. During each training step, detect instinct opportunities in position
2. If model's top move != instinct move → add auxiliary loss
3. Weight decays as benchmark accuracy improves
4. Model "graduates" from instinct training naturally

**Files:**
- `training/instinct_benchmark.py` - Generates 1449 test positions
- `training/train_instincts.py` - Supervised training on instincts
- `training/benchmarks/instincts/*.json` - Position databases

---

## Previous State (2025-12-19)

### Session Summary

**What Works:**
| Component | Status | Notes |
|-----------|--------|-------|
| ONNX export pipeline | ✅ DONE | PyTorch → ONNX (7.3MB) |
| Browser inference | ✅ DONE | ONNX Runtime Web, Level 6 AI |
| Board size validation | ✅ DONE | Metadata-driven, 9x9 only |
| Playwright tests | ✅ DONE | 2 tests passing |
| Benchmark suite | ✅ DONE | 17 tactical positions |

**What's Implemented (Untested):**
| Technique | Code | Tests | Notes |
|-----------|------|-------|-------|
| A.4 Shaped Dirichlet | ✅ | ❌ | Concentrate noise on high-logit moves |
| A.5 Root Policy Temp | ✅ | ❌ | 1.25 early → 1.1 late |
| A.6 Soft Policy Head | ✅ | ❌ | Head exists, loss not wired |

**Model Quality (Current 9x9):**
```
Overall: 17.6% Top-1 accuracy
  Capture:  0.0% (8 positions) — FAILING
  Ladder:  20.0% (5 positions) — WEAK
  Snapback: 50.0% (4 positions) — OK
```

**Next Steps - REVISED (2025-12-19 Late Session):**

### The Problem
Model is STUPID at captures (0% accuracy). This is unacceptable - captures are DETERMINISTIC.
Throwing more compute at broken training won't help. Fix foundations first.

### New Strategy: Foundation → Signal → Scale

**Phase 1: Adaptive Instinct Curriculum** 🎯 HIGHEST PRIORITY
Why: Model can't learn captures if captures never appear in training data.
- [x] DONE - Instinct benchmark generator (1449 positions)
- [x] DONE - 8 instincts: Capture, Escape, Connect, Cut, Extend, Block, Atari, Defend
- [x] DONE - Model baseline: 1.3% overall, 0% captures
- [ ] TODO - Implement `InstinctLoss` class:
  - Detect instinct opportunities in training positions
  - Add auxiliary loss for missed instinct moves
  - Adaptive weight: λ(t) = λ₀ × (1 - benchmark_accuracy)
- [ ] TODO - Curriculum injection:
  - Early: 30% instinct positions in replay buffer
  - Mid: 10% instinct positions
  - Late: pure self-play (instincts mastered)
- [ ] TODO - Success criteria: Capture >80%, Escape >60%, Overall >50%

**Phase 2: Dense Training Signal** 🔥 HIGH PRIORITY
Why: Win/loss is 1 bit per game. Ownership is 81 floats per position.
- [ ] TODO - Wire ownership prediction loss (Board.ownership_map() exists)
- [ ] TODO - Wire soft policy loss (A.6 - head exists, loss not wired)
- [ ] TODO - Wire score prediction loss (head exists)
- [ ] TODO - Add symmetry augmentation (8× free data from rotations/reflections)

**Phase 3: Curriculum Learning** 📈 HIGH PRIORITY
Why: 5×5 games are 10× faster. Learn fundamentals, transfer up.
- [ ] TODO - Start curriculum at 5×5 board (capture-focused)
- [ ] TODO - Progress: 5×5 → 7×7 → 9×9 based on benchmark accuracy
- [ ] TODO - Transfer learning: load smaller model weights into larger

**Phase 4: Training Efficiency** ⚡ MEDIUM PRIORITY (after above work)
Why: Only optimize what works.
- [ ] TODO - Parallel self-play (multiprocessing)
- [ ] TODO - Increase MCTS batch size (8 → 32)
- [ ] TODO - Target: >50% GPU utilization (currently 6%)

### Success Criteria
Before Phase 4, we MUST achieve:
- [ ] Capture benchmark: >80% (currently 0%)
- [ ] Ladder benchmark: >60% (currently 20%)
- [ ] Overall benchmark: >50% (currently 17.6%)

### Key Insight
**Neurosymbolic > Pure Neural for tactics.**
The network learns PATTERNS. Tactics are CALCULATIONS.
Hybrid approach: Neural for strategy, Symbolic for tactics.

**Blog Posts:**
- 2025-12-19: "From Python to Browser" (Integration)
- 2025-12-20: "Footsteps of Giants" (Research)
- 2025-12-21: "Tester c'est Douter" (Reflection)

---

## Previous State (2025-12-12)

### KataGo Techniques Implemented ✅
- [x] DONE - Curriculum learning (tactical positions first)
- [x] DONE - Ownership head (1.65× speedup, 361× more signal)
- [x] DONE - Opponent move prediction (1.30× speedup)
- [x] DONE - Score distribution head (richer signal than win/loss)
- [x] DONE - BCE value loss (Cazenave 2020)
- [x] DONE - MobileNetV2 backbone (38% parameter reduction)
- [x] DONE - Architecture diagrams (ARCH.mermaid, DATA_FLOW.mermaid)

### Neurosymbolic Extension ✅ NEW
- [x] DONE - TacticalAnalyzer (ladders, snapbacks, capture verification)
- [x] DONE - HybridMCTS (neural + symbolic tactical verification)
- [x] DONE - watch_game.py for manual game validation
- [ ] IN PROGRESS - Training with all enhancements (25 epochs)

### What's Been Done:
- [x] DONE - Training infrastructure created
- [x] DONE - Pro games downloaded (93,164 games from CWI.nl)
- [x] DONE - SGF parser with caching + ownership + opponent moves
- [x] DONE - Supervised training script with all aux targets
- [x] DONE - Batched MCTS (15x speedup)
- [x] DONE - Defensive training practices added

### Current Test Command:
```bash
cd training && poetry run python train_supervised.py \
  --tactical-features --curriculum --ownership --opponent-move \
  --epochs 20 --max-positions 300000
```

---

## 🔬 POST-KATAGO SOTA IMPROVEMENTS (2019-2025 Research)

Based on comprehensive literature review. Sources in ARCH.md.

### Phase A: Quick Wins (1-2 days each) 🟢

**A.1: NN Evaluation Cache**
- [x] DONE - Zobrist hashing for Board (fast 64-bit position hash)
- [x] DONE - NNCache class with LRU eviction (100K entries default)
- [x] DONE - Integrated into MCTS search (cache hits skip batch eval)
- Source: [Speculative MCTS, NeurIPS 2024] - up to 5.8× speedup
- Complexity: Low (dict + hash function)

**A.2: WED (Weight by Episode Duration)**
- [x] DONE - Weight samples: `1 / game_length` so each game counts equally
- [x] DONE - WeightedRandomSampler for position sampling (--wed flag)
- Source: [Manipulating Distributions, 2020] - "single most effective technique"
- Complexity: Very low (one line in loss)

**A.3: BCE Value Loss**
- [x] DONE - Replace MSE with BCE: `BCE((tanh_output+1)/2, (target+1)/2)`
- [x] DONE - Disable autocast for BCE (not autocast-safe)
- Source: [Cazenave MobileNet, 2020] - more robust on small networks
- Complexity: Very low (change loss function)

### Phase A.4-A.6: More KataGo Quick Wins 🟢

**A.4: Shaped Dirichlet Noise**
- [ ] TODO - Concentrate exploration noise on higher-logit moves (not uniform)
- [ ] TODO - Split alpha: half uniform, half weighted by policy
- Source: [KataGo Methods](https://github.com/lightvector/KataGo/blob/master/docs/KataGoMethods.md)
- Complexity: Low (modify noise generation in MCTS)

**A.5: Root Policy Softmax Temperature**
- [ ] TODO - Apply temperature >1.0 to policy before search (1.25 early, decay to 1.1)
- [ ] TODO - Counteracts MCTS sharpening on already-preferred moves
- Source: KataGo Methods
- Complexity: Very low (one line change)

**A.6: Auxiliary Soft Policy Target**
- [ ] TODO - Additional policy head predicting softened policy (T=4)
- [ ] TODO - Weight 8× relative to main policy during training
- [ ] TODO - Forces discrimination of lower-probability moves
- Source: KataGo Methods - improves policy learning
- Complexity: Low (one extra head + loss term)

### Phase B: Medium Effort (1-2 weeks each) 🟡

**B.1: MobileNetV2 Backbone** ⭐ HIGH PRIORITY
- [x] DONE - MobileNetV2Block: expand → depthwise → project pattern
- [x] DONE - ReLU6 activation, linear bottleneck (no final ReLU)
- [x] DONE - CLI: `--backbone mobilenetv2 --mobilenet-expansion 4`
- [x] DONE - 38% parameter reduction (2.5M → 1.5M params)
- Source: [Cazenave MobileNet, 2020] - beats larger ResNets in playing strength
- Complexity: Medium (rewrite ResBlock class)

**B.2: Score Distribution Head**
- [x] DONE - Add head predicting P(final_score = k) for k in [-50, +50]
- [x] DONE - Cross-entropy loss against actual score (mask invalid scores from resigns)
- [x] DONE - CLI: `--score-dist --score-weight 0.02`
- Source: [KataGo 2019] - ~21% of games have numerical scores
- Complexity: Low (one conv head + softmax over 101 bins)

**B.3: LATE Simulation Scheduling**
- [ ] LATER - Early training: more playouts for late-game moves
- [ ] LATER - Gradually shift budget to earlier moves
- [ ] LATER - Weight loss by `w(generation, move_index)`
- Source: [LATE, ECAI 2023] - beats KataGo's RPC, 20-80% compute savings
- Complexity: Medium (modify self-play loop)

**B.4: Go-Exploit State Archive**
- [ ] LATER - Archive positions with high MCTS variance / policy disagreement
- [ ] LATER - Start 30-50% of games from archived states
- Source: [Go-Exploit, 2023] - more sample-efficient than standard AlphaZero
- Complexity: Medium (archive buffer + selection logic)

**B.5: Policy Surprise Weighting**
- [ ] LATER - Overweight samples where MCTS found moves policy underestimated
- [ ] LATER - Redistribute weights based on KL-divergence (prior vs target)
- [ ] LATER - Helps discover overlooked strong moves
- Source: [KataGo Methods](https://github.com/lightvector/KataGo/blob/master/docs/KataGoMethods.md)
- Complexity: Medium (KL calculation + reweighting)

**B.6: Short-term Value Targets**
- [ ] LATER - Auxiliary heads predicting exponentially-averaged future MCTS values
- [ ] LATER - Three time horizons (short/medium/long)
- [ ] LATER - Lower-variance feedback improves convergence
- Source: KataGo Methods
- Complexity: Medium (3 extra heads + target computation)

**B.7: Dynamic Variance-Scaled cPUCT**
- [ ] LATER - Scale exploration coefficient by empirical utility variance at each node
- [ ] LATER - High variance → more exploration; low variance → exploitation
- Source: KataGo Methods
- Complexity: Medium (online variance tracking)

### Phase B.5: Neurosymbolic Hybrid (MCTS + Neural) ⭐ NEW

**Motivation**: Pure neural networks learn patterns but can't calculate tactical sequences.
For situations like snapbacks, ladders, and capture races, we need symbolic verification.

**Implementation**:
- [x] DONE - TacticalAnalyzer class (`training/tactics.py`)
  - Ladder detection and tracing
  - Snapback detection
  - Capture sequence verification (alpha-beta)
  - Life/death evaluation (minimax)
- [x] DONE - HybridMCTS class (`training/hybrid_mcts.py`)
  - Policy adjustment: boost captures, penalize dead groups
  - Value refinement: blend neural + tactical for critical positions
  - Selective deepening for tactical positions
- [x] DONE - Updated selfplay.py with `--use-hybrid` flag
- [ ] TODO - Integration with training loop
- [ ] TODO - Validation: watch games for tactical improvements

**Key Insight**: Neural network learns *what kinds of positions* are tactical.
Symbolic component verifies *specific outcomes* through calculation.

**Expected Benefits**:
| Aspect | Pure Neural | Neurosymbolic |
|--------|-------------|---------------|
| Ladder reading | ~60% accuracy | ~99% accuracy |
| Snapback detection | Often missed | Deterministic |
| Capture races | Value network guesses | Verified outcomes |

**Documentation**: See `training/NEUROSYMBOLIC.md` for full architecture details.

### Phase C: Ambitious (weeks-months) 🔴

**C.1: ResTNet Hybrid Architecture**
- [ ] LATER - Interleave ResNet (R) and Transformer (T) blocks
- [ ] LATER - Pattern: RRRTRRTRRT (10 blocks)
- [ ] LATER - Relative positional attention, 2D↔1D conversion
- Source: [ResTNet, IJCAI 2025] - win rate 53.6% → 60.9% vs KataGo
- Complexity: High (transformer blocks)

**C.2: EfficientFormer Integration**
- [ ] LATER - Replace some conv blocks with EfficientFormer meta-blocks
- [ ] LATER - Keep 19×19 resolution (no patching)
- Source: [ViT for Go, 2023] - beats 20×256 ResNet
- Complexity: High (new architecture family)

**C.3: Lightweight Reanalyze**
- [ ] LATER - Periodically re-run MCTS on old buffer positions
- [ ] LATER - Update policy/value targets in place
- Source: [MuZero Unplugged, 2021] - improves data efficiency
- Complexity: Medium (background job)

**C.4: Full MuZero Dynamics**
- [ ] LATER - Learn transition model for planning
- [ ] LATER - Self-supervised latent consistency loss
- Source: [EfficientZero, NeurIPS 2021] - 500× sample efficiency
- Complexity: Very high (paradigm shift)

---

## 📋 NEXT STEPS

### Step 1: Reset GPU
```bash
# Just reboot the machine
sudo reboot
```

### Step 2: Verify GPU Works
```bash
cd ~/Work/Perso/Games/gogogo/training
poetry run python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}'); print(torch.cuda.get_device_name(0))"
```

### Step 3: Run Supervised Training (with defensive practices)
```bash
cd ~/Work/Perso/Games/gogogo/training

# Full training with mid-epoch checkpoints
poetry run python train_supervised.py --board-size 19 --max-games 5000 --epochs 10

# If it crashes again, resume from checkpoint:
poetry run python train_supervised.py --board-size 19 --max-games 5000 --epochs 10 \
    --resume checkpoints/supervised_epoch_1_batch_2000.pt
```

### Step 4: Self-Play Training (after supervised completes)
```bash
poetry run python train.py --board-size 19 --resume checkpoints/supervised_best.pt --iterations 50
```

### Step 5: Monitor with TensorBoard
```bash
poetry run tensorboard --logdir logs
# Open http://localhost:6006
```

---

## Defensive Training Features (Now Implemented)

- ✅ **Mid-epoch checkpoints**: Every 1000 batches
- ✅ **Gradient clipping**: Prevents exploding gradients
- ✅ **CUDA cache clearing**: Every 500 batches
- ✅ **Emergency save on crash**: Catches RuntimeError/KeyboardInterrupt
- ✅ **Resume support**: `--resume` flag continues from checkpoint
- ✅ **Batched MCTS**: 15x GPU speedup via batch inference

---

### Phase 0: Training Infrastructure
- [x] DONE - Create `training/` directory structure
- [x] DONE - Setup Poetry environment (PyTorch + CUDA 12.5)
- [x] DONE - Verify GPU detection and CUDA works

### Phase 1: Board Representation (Tensors)
- [x] DONE - Board state → tensor conversion (17 planes)
- [x] DONE - Feature planes:
  - [x] Current player stones (1)
  - [x] Opponent stones (1)
  - [x] Current player indicator (1)
  - [x] Move history planes (7 × 2 = 14)
- [x] DONE - Tests for tensor conversion

### Phase 2: Neural Network Architecture
- [x] DONE - ResNet backbone (configurable blocks/filters)
- [x] DONE - Policy head (outputs move probabilities)
- [x] DONE - Value head (outputs scalar -1 to +1)
- [x] DONE - GPU memory check (fits in 12GB)

### Phase 3: Neural MCTS
- [x] DONE - MCTS node with neural network evaluation
- [x] DONE - UCB formula with policy prior
- [x] DONE - Leaf evaluation via value head
- [x] DONE - Batched MCTS for GPU efficiency (15x speedup)
- [x] DONE - Temperature-based move selection

### Phase 4: Supervised Pre-Training (NEW)
- [x] DONE - Download pro games database (CWI.nl, 93K games)
- [x] DONE - SGF parser with tensor conversion
- [x] DONE - Dataset caching (fast reload)
- [x] DONE - Supervised training script
- [ ] **IN PROGRESS** - Train on pro games (crashed, needs reboot)

### Phase 5: Self-Play Game Generation
- [x] DONE - Play game using neural MCTS
- [x] DONE - Record (board_state, mcts_policy, game_result) tuples
- [x] DONE - Replay buffer implementation
- [x] DONE - Game stats display (stones, groups, score)
- [ ] TODO - Run self-play after supervised training

### Phase 6: Training Loop
- [x] DONE - Load replay buffer samples
- [x] DONE - Policy loss: cross-entropy
- [x] DONE - Value loss: MSE
- [x] DONE - Optimizer: Adam
- [x] DONE - Checkpointing (every N steps + mid-epoch)
- [x] DONE - TensorBoard logging
- [x] DONE - Emergency checkpoint on crash

### Phase 7: Evaluation & Iteration
- [x] DONE - Pit current model vs previous checkpoint
- [x] DONE - Win rate calculation
- [ ] TODO - Run full evaluation cycle

### Phase 8: Export to Browser
- [ ] LATER - Export PyTorch model to ONNX
- [ ] LATER - Convert ONNX to TensorFlow.js
- [ ] LATER - Test inference in browser

---

### Training Commands

```bash
# Setup (already done)
cd training
poetry install

# Verify GPU
poetry run python -c "import torch; print(torch.cuda.is_available())"

# Supervised training on pro games
poetry run python train_supervised.py --board-size 19 --max-games 5000 --epochs 10

# Self-play training (after supervised)
poetry run python train.py --board-size 19 --resume checkpoints/supervised_best.pt --iterations 50

# Quick test (smaller network)
poetry run python train_supervised.py --board-size 19 --max-games 100 --epochs 3 --quick
```

---

### File Structure (Current)

```
training/
├── pyproject.toml     # Poetry dependencies
├── config.py          # Hyperparameters (DEFAULT, QUICK configs)
├── board.py           # Board representation (tensors)
├── model.py           # Neural network (ResNet + heads)
├── mcts.py            # Neural MCTS (batched for GPU)
├── selfplay.py        # Game generation + replay buffer
├── train.py           # Self-play training loop
├── train_supervised.py # Supervised training on pro games
├── sgf_parser.py      # SGF file parser + dataset loader
├── TRAIN.md           # Quick training instructions
├── checkpoints/       # Saved models
├── dataset_cache/     # Cached parsed games (fast reload)
├── data/games/        # Raw SGF files (93K games)
└── logs/              # TensorBoard logs
```

---

## 🔮 FUTURE: Board Size Generalization Research

**Goal**: Train once, play on any board size (9x9, 13x13, 19x19+)

**Current State**: Model trained on 9x9, cannot play other sizes.

### Research Directions (Ranked by Promise)

**1. Multi-Scale Training** ⭐⭐⭐ PROVEN
- [ ] LATER - Train simultaneously on 7x7, 9x9, 13x13, 19x19
- [ ] LATER - Dynamic policy head (output size varies with board)
- [ ] LATER - Benchmark: does smaller-board training accelerate larger?
- Source: [KataGo](https://github.com/lightvector/KataGo/blob/master/docs/KataGoMethods.md) - trains 7x7→19x19, generalizes to 30x30+
- Evidence: "Mixed-size training is NOT slower—may be FASTER due to knowledge transfer"

**2. Transfer Learning / Curriculum** ⭐⭐⭐ PROVEN
- [ ] LATER - Train heavily on 9x9 (fast feedback)
- [ ] LATER - Fine-tune on 13x13, then 19x19
- [ ] LATER - Freeze early layers, retrain policy/value heads
- Source: [AlphaGateau, NeurIPS 2024](https://arxiv.org/abs/2410.23753) - 5x5→8x8 chess works
- Research Q: Can we measure transfer vs fresh training in small-scale experiment?

**3. Graph Neural Networks** ⭐⭐⭐ EMERGING (Separate Baseline)
- [ ] LATER - Treat board as graph (nodes=intersections, edges=adjacency)
- [ ] LATER - GNN architecture (message passing, attention)
- [ ] LATER - Compare parameter efficiency vs CNN
- [ ] LATER - Test browser inference speed (ONNX GNN support?)
- Source: [AlphaGateau, NeurIPS 2024](https://arxiv.org/abs/2410.23753) - 1.0M GNN beats 2.2M CNN
- Evidence: "Order of magnitude faster learning, better generalization"
- Risk: Browser inference may be slow, immature tooling

**4. Coordinate Embeddings** ⭐⭐ MAYBE
- [ ] LATER - Add explicit positional encoding (row/col as features)
- [ ] LATER - Relative position features (distance to walls, corners)
- [ ] LATER - Sinusoidal vs learned embeddings
- Source: CoordConv paper, Transformer positional encodings
- Note: KataGo succeeds without explicit coords—convolutions learn position implicitly

**5. Padding/Cropping** ❌ SKIP
- Not worth pursuing—loses global context, no SOTA uses it

### Research Questions to Answer
- [ ] DISCUSS - Can we run a small-scale experiment (1 day training) to compare multi-scale vs single-scale?
- [ ] DISCUSS - Is self-play sufficient for curriculum, or do we need pro games at each size?
- [ ] DISCUSS - GNN vs CNN: worth the implementation cost for generalization benefits?

### Sources
- [KataGo Methods](https://github.com/lightvector/KataGo/blob/master/docs/KataGoMethods.md)
- [AlphaGateau - NeurIPS 2024](https://arxiv.org/abs/2410.23753)
- [Curriculum Learning Survey](https://jmlr.org/papers/volume21/20-212/20-212.pdf)

---

## Previous Sprint: MVP (Milestones 1-2)

**Status**: Basic game working, now pivoting to neural AI

**Completed**:
- [x] Basic Go rules (capture, territory)
- [x] Simple heuristic AI (wall-building problem)
- [x] Game UI with board rendering
- [x] Currency system

**On Hold**:
- Heuristic AI improvements (superseded by neural approach)
- Policy tactical lookahead (features may inform neural network)

---

**Last Updated**: 2025-11-29
**Total Features Tracked**: 300+
**Status**: Neural network training infrastructure in progress
