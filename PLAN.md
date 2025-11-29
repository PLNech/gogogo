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

### Phase 0: Training Infrastructure
- [ ] TODO - Create `training/` directory structure
- [ ] TODO - Setup Python environment (PyTorch + CUDA)
- [ ] TODO - Verify GPU detection and CUDA works

### Phase 1: Board Representation (Tensors)
- [ ] TODO - Board state → tensor conversion (19×19 planes)
- [ ] TODO - Feature planes:
  - [ ] Black stones plane (1)
  - [ ] White stones plane (1)
  - [ ] Current player plane (1)
  - [ ] Move history planes (8 moves × 2 colors = 16)
  - [ ] Liberty count planes (1-4+ liberties = 4)
  - [ ] Ko point plane (1)
  - **Total: ~24 planes**
- [ ] TODO - Tests for tensor conversion

### Phase 2: Neural Network Architecture
- [ ] TODO - ResNet backbone (start simple: 6 blocks, 128 filters)
- [ ] TODO - Policy head (outputs 361 move probabilities for 19×19)
- [ ] TODO - Value head (outputs single scalar -1 to +1)
- [ ] TODO - Tests for network forward pass
- [ ] TODO - GPU memory check (fit in 12GB)

### Phase 3: Neural MCTS
- [ ] TODO - MCTS node with neural network evaluation
- [ ] TODO - UCB formula with policy prior: `Q + c * P * sqrt(N_parent) / (1 + N)`
- [ ] TODO - Leaf evaluation via value head (no random rollouts)
- [ ] TODO - MCTS search loop (configurable iterations)
- [ ] TODO - Temperature-based move selection
- [ ] TODO - Tests for MCTS correctness

### Phase 4: Self-Play Game Generation
- [ ] TODO - Play game using neural MCTS
- [ ] TODO - Record (board_state, mcts_policy, game_result) tuples
- [ ] TODO - Save games to replay buffer (disk-based for large datasets)
- [ ] TODO - Parallel game generation (multiple games simultaneously)
- [ ] TODO - Tests for self-play games

### Phase 5: Training Loop
- [ ] TODO - Load replay buffer samples
- [ ] TODO - Policy loss: cross-entropy vs MCTS policy
- [ ] TODO - Value loss: MSE vs game outcome
- [ ] TODO - Combined loss: `L = L_policy + L_value`
- [ ] TODO - Optimizer: Adam or SGD with momentum
- [ ] TODO - Learning rate schedule
- [ ] TODO - Checkpointing (save every N steps)
- [ ] TODO - Logging (TensorBoard or wandb)
- [ ] TODO - Tests for training step

### Phase 6: Evaluation & Iteration
- [ ] TODO - Pit current model vs previous checkpoint
- [ ] TODO - Win rate calculation (need >55% to promote)
- [ ] TODO - Elo estimation (optional)
- [ ] TODO - Automated training loop:
  ```
  while True:
    1. Generate N self-play games
    2. Add to replay buffer
    3. Train for K steps
    4. Checkpoint
    5. Evaluate vs previous
    6. If better → promote, else → continue
  ```

### Phase 7: Export to Browser
- [ ] TODO - Export PyTorch model to ONNX
- [ ] TODO - Convert ONNX to TensorFlow.js
- [ ] TODO - Test inference in browser
- [ ] TODO - Integrate with existing game UI

### Phase 8: Difficulty Scaling
- [ ] TODO - Weak AI: fewer MCTS iterations (8-16)
- [ ] TODO - Medium AI: moderate iterations (64-128)
- [ ] TODO - Strong AI: full iterations (400-800)
- [ ] TODO - Multiple model sizes for different levels

---

### Training Parameters (Starting Point)

```python
# Network
BOARD_SIZE = 9  # Start with 9x9, scale to 19x19 later
NUM_BLOCKS = 6  # Residual blocks
NUM_FILTERS = 128  # Channels per conv layer
INPUT_PLANES = 24  # Feature planes

# MCTS
MCTS_SIMULATIONS = 100  # Per move during self-play
C_PUCT = 1.5  # Exploration constant
TEMPERATURE = 1.0  # First 30 moves, then 0 (argmax)

# Training
BATCH_SIZE = 256
LEARNING_RATE = 0.001
REPLAY_BUFFER_SIZE = 100_000  # Position tuples
GAMES_PER_ITERATION = 100
TRAINING_STEPS_PER_ITERATION = 1000
CHECKPOINT_INTERVAL = 500  # Steps

# Hardware
NUM_PARALLEL_GAMES = 4  # Depends on RAM
```

---

### Training Commands (To Be Implemented)

```bash
# Setup
cd training
python -m venv venv
source venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install numpy tensorboard

# Verify GPU
python -c "import torch; print(torch.cuda.is_available())"

# Train
python train.py --board-size 9 --iterations 100

# Evaluate
python evaluate.py --model checkpoints/latest.pt --games 100

# Export
python export.py --model checkpoints/best.pt --output ../src/core/ai/models/
```

---

### Success Criteria

**9x9 Board (First Target)**:
- [ ] Network trains without crashing
- [ ] Self-play games are legal (no rule violations)
- [ ] Win rate vs random > 90%
- [ ] Win rate vs previous model > 55% (improvement)
- [ ] Training completes in < 24 hours for basic strength

**19x19 Board (Later)**:
- [ ] Scale network (more blocks, more filters)
- [ ] Longer training (days not hours)
- [ ] Competent play (doesn't build walls, understands territory)

---

### File Structure

```
training/
├── __init__.py
├── config.py          # Hyperparameters
├── board.py           # Board representation (tensors)
├── model.py           # Neural network (ResNet + heads)
├── mcts.py            # Neural MCTS
├── selfplay.py        # Game generation
├── train.py           # Training loop
├── evaluate.py        # Model evaluation
├── export.py          # ONNX/TFJS export
├── utils.py           # Helpers
├── checkpoints/       # Saved models
├── games/             # Self-play game records
└── logs/              # TensorBoard logs
```

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
