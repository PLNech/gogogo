# Sensei's Library Integration Guide for GoGoGo

## Overview

Sensei's Library (https://senseis.xmp.net) is a comprehensive wiki-based resource for the game of Go, containing over 26,490 pages covering everything from beginner tutorials to professional game analysis. This document maps how we can leverage this extensive resource for our idle game "GoGoGo".

**Site Statistics:**
- 26,490+ pages of content
- 25+ daily edits (active community)
- Collaborative wiki architecture
- Multi-language support (primarily English, with Japanese/Chinese terminology)

---

## Site Structure Overview

### Primary Navigation Sections

#### 1. For Newcomers
- **What Go Is** - Basic game introduction
- **Pages for Beginners** - Comprehensive beginner portal
- **Beginner Study Section** - 14-lesson guided curriculum

#### 2. Core Content Areas
- **Reference Section** - Encyclopedic top-down structure
- **Problems and Exercises** - 600+ graded problems
- **Starting Points** - Multiple themed entry points
- **Guided Tours** - Curated thematic pathways

#### 3. Search & Discovery
- Quick title search
- Full-text search
- Keyword search
- **Diagram position search** - Find positions by board patterns
- Random page browsing
- Recent changes tracking

### URL Pattern Structure

Base URL: `https://senseis.xmp.net/`

**Key URL Patterns:**
```
https://senseis.xmp.net/?[PageName]
https://senseis.xmp.net/?topic=[TopicID]
```

**Examples:**
- Main page: `https://senseis.xmp.net/`
- Beginner page: `https://senseis.xmp.net/?PagesForBeginners`
- Study section: `https://senseis.xmp.net/?BeginnerStudySection`
- Specific topic: `https://senseis.xmp.net/?Joseki`
- Problems: `https://senseis.xmp.net/?BeginnerExercises`

**CamelCase Convention:** All page names use CamelCase without spaces (e.g., `LifeAndDeath`, `BasicTactics`, `StrategicConcepts`)

---

## Beginner's Curriculum Mapping

### The 14-Lesson Beginner Study Section

This structured curriculum provides a complete beginner-to-intermediate progression, perfect for our milestone system:

#### **Phase 1: Foundations (Lessons 0-3)**

**Lesson 0: What is Go?**
- URL: `https://senseis.xmp.net/?BeginnerStudySection`
- Content: Game orientation, basic concepts
- Game Integration: Tutorial introduction, first-time player experience

**Lesson 1: The Rules (15-minute intro)**
- Content: Legal moves, captures, basic mechanics
- Game Integration: Interactive rule demonstrations, unlock basic capturing

**Lesson 2: Territory Concept**
- Content: End-game timing, scoring methods
- Game Integration: Territory school foundation, scoring mechanics unlock

**Lesson 3: Liberties and Capturing**
- Content: Double atari, ladders, nets
- Key Techniques:
  - **Double Atari** - Threatening two groups simultaneously
  - **Ladder** - Sequential capturing technique
  - **Net** - Surrounding opponent stones
- Game Integration: Combat school basics, tactical challenges

#### **Phase 2: Tactical Training (Lessons 4-5)**

**Lesson 4: Introduction to Life and Death**
- Content: Eyes, vulnerable groups, basic survival
- URL: `https://senseis.xmp.net/?LifeAndDeath`
- Key Concepts:
  - False vs. true eyes
  - Two-eye rule
  - Eye space configurations
- Game Integration: Life & death puzzles, survival challenges

**Lesson 5: Beginner Life and Death**
- Content: Predicting group outcomes
- Problems: 300+ beginner exercises available
- Game Integration: Progressive puzzle milestones, pattern recognition rewards

#### **Phase 3: Strategic Development (Lessons 6-9)**

**Lesson 6: The Opening/Fuseki**
- URL: `https://senseis.xmp.net/?Opening` or `?Fuseki`
- Content: Foundational positioning principles
- Three study areas:
  1. Whole board strategy
  2. Half-board patterns
  3. Corner sequences (joseki)
- Game Integration: Influence school introduction, opening theory unlocks

**Lesson 7: Balance of Power**
- Content: Strength, thickness, fighting purpose
- URL: `https://senseis.xmp.net/?Influence` & `?Thickness`
- Game Integration: Power dynamics, school balance mechanics

**Lesson 8: Local Efficiency**
- Content: Shape and tesuji
- URLs: `https://senseis.xmp.net/?Shape` & `?Tesuji`
- Key Principles:
  - Good shape = efficiency + flexibility
  - Maximizing liberties
  - Connection and eye formation
- Game Integration: Efficiency bonuses, pattern rewards

**Lesson 9: The Fundamentals**
- Content: Deep understanding through precision study
- Note: "Biggest return on investment from studying deeper, not wider"
- Game Integration: Mastery system, depth-over-breadth rewards

#### **Phase 4: Advanced Concepts (Lessons 10-14)**

**Lesson 10: Positional Judgment**
- URL: `https://senseis.xmp.net/?StrategicConcepts`
- Content: Counting and situation assessment
- Game Integration: AI evaluation display, position analysis tools

**Lesson 11: Strategic Concepts**
- Eight core tools including:
  - **Miai** - Alternative plays of equal value
  - **Aji** - Residual potential in positions
  - **Sente/Gote** - Initiative and response
  - **Thickness** - Strong position building
- Game Integration: Strategy card unlocks, concept-based bonuses

**Lesson 12: More Tesuji and Tsumego**
- Content: Daily tactical exercises
- URLs: `https://senseis.xmp.net/?Tesuji` & `?Tsumego`
- Game Integration: Daily challenges, streak rewards

**Lesson 13: The Endgame/Yose**
- URL: `https://senseis.xmp.net/?Endgame`
- Content: Move sequencing, value calculation
- Key Techniques:
  - Monkey jump
  - Hanetsugi (connection after diagonal)
  - Hane-descend
  - Tedomari (final important play)
- Game Integration: Endgame specialist path, scoring optimization

**Lesson 14: Professional Game Study**
- Content: Pattern recognition and planning
- Game Integration: Replay famous games, professional AI opponents

#### **Interludes (Between Lessons)**

Three break sections offer:
- Etiquette guidelines
- Humor and Go culture
- Terminology deepening

---

## Content Categorization by Three Schools

### TERRITORY SCHOOL (Amashi Style)

**Core Philosophy:** Control defined board areas, accumulate concrete points

**Key Senseis Library Resources:**

1. **Territory Fundamentals**
   - URL: `https://senseis.xmp.net/?Territory`
   - Concepts:
     - Definite territory (countable points)
     - Estimated territory (certain areas)
     - Potential territory (invasion-vulnerable)
   - Requirements: Living groups surrounding empty points

2. **Corner Patterns (Joseki)**
   - URL: `https://senseis.xmp.net/?Joseki`
   - Categories:
     - **3-3 (Sansan)** - Immediate corner territory
     - **3-4 (Komoku)** - Balanced approach
     - Corner joseki for secure points
   - Beginner Focus: Territory-oriented joseki outcomes

3. **Endgame Excellence**
   - URL: `https://senseis.xmp.net/?Endgame`
   - Territory optimization in final phase
   - Counting and calculation focus
   - Boundary play mastery

4. **Strategic Concepts for Territory**
   - Connection (securing groups)
   - Defense (protecting points)
   - Efficiency (maximum points per stone)
   - Timing (when to secure vs. expand)

**Game Integration Ideas:**
- Territory school bonuses for secure area control
- Joseki pattern library focused on corner territory
- Endgame specialists with calculation bonuses
- "Solid player" achievement path
- Defensive tesuji emphasis

---

### INFLUENCE SCHOOL (Power/Moyo Style)

**Core Philosophy:** Build strength and potential across the board, convert influence to territory later

**Key Senseis Library Resources:**

1. **Influence Fundamentals**
   - URL: `https://senseis.xmp.net/?Influence`
   - Definition: "Live stones' radiated light toward the outside"
   - Four amplifiers:
     - Group strength
     - Open space
     - Weak opposition
     - Long-range effect

2. **Opening Theory (Fuseki)**
   - URL: `https://senseis.xmp.net/?Fuseki`
   - Whole-board thinking emphasis
   - Notable patterns:
     - **Chinese Fuseki** - Large-scale influence
     - **Sanrensei** - Three star points (power-oriented)
     - **Shin Fuseki** - New opening theory
   - Focus on positioning over immediate territory

3. **Thickness and Power**
   - URL: `https://senseis.xmp.net/?Thickness`
   - Building strong positions
   - Converting strength to future advantage
   - Long-term strategic thinking

4. **Corner Patterns for Influence**
   - **4-4 (Hoshi)** - Star point for influence
   - **4-5 (Takamoku)** - High approach
   - Extended patterns (5-5, 6-3, 6-4)

5. **Strategic Concepts for Influence**
   - Mobility and flexibility
   - Reduction (limiting opponent territory)
   - Timing (knowing when to cash in influence)
   - Whole-board vision

**Game Integration Ideas:**
- Influence school bonuses for board control
- Moyo (framework) building rewards
- Fuseki pattern library focused on power positions
- "Cosmic player" achievement path
- Long-term planning bonuses
- Conversion mechanics (influence → territory)

---

### COMBAT SCHOOL (Fighting Style)

**Core Philosophy:** Direct confrontation, tactical superiority, aggressive play

**Key Senseis Library Resources:**

1. **Fighting Fundamentals**
   - URL: `https://senseis.xmp.net/?Fighting`
   - Three dimensions:
     - Local tactical battles (capturing races)
     - Strategic engagements (running fights)
     - Psychological dimension (fighting spirit)
   - Key topics:
     - Capturing races
     - Ko fights
     - Concurrent fights
     - Profit vs. fighting balance

2. **Life and Death Mastery**
   - URL: `https://senseis.xmp.net/?LifeAndDeath`
   - Killing opponent groups
   - Saving your own groups
   - Key patterns:
     - **L-formation**
     - **Tripod**
     - **Carpenter's Square**
     - Bent four, twisted four
     - Corner formations
   - 300+ beginner exercises available

3. **Tesuji (Tactical Techniques)**
   - URL: `https://senseis.xmp.net/?Tesuji`
   - Eight main categories:

     **A. Capturing Techniques:**
     - Net (primary beginner technique)
     - Ladder (secondary beginner technique)
     - Squeeze
     - Oiotoshi
     - Crane's nest

     **B. Attacking Techniques:**
     - Shortage of liberties
     - Eye stealing
     - Throw-in
     - Driving techniques

     **C. Defending Techniques:**
     - Racing wins
     - Descent moves
     - Eye-making tactics

     **D. Shortage of Liberties:**
     - Snapback
     - Belly attachment
     - Twirl
     - Elbow lock

     **E. Miai Tesuji:**
     - Grappling hook
     - Nose
     - Clamping
     - Double threat

     **F. Shape Tesuji:**
     - Ear-cleaning techniques

     **G. Yose Tesuji:**
     - Endgame-specific moves

     **H. Miscellaneous:**
     - Wedge
     - Connecting techniques
     - Sente-getting plays

4. **Tactical Problems**
   - URL: `https://senseis.xmp.net/?ProblemsAndExercises`
   - 300+ beginner exercises
   - 200+ kyu exercises
   - 100+ dan exercises
   - Organized by:
     - Difficulty level
     - Game stage (opening/middle/endgame)
     - Specific tactics
     - Essential concepts

5. **Strategic Concepts for Combat**
   - Attack (putting pressure)
   - Cut (separating opponent)
   - Capture and capturing races
   - Sacrifice (trading for advantage)
   - Aji (residual potential)

**Game Integration Ideas:**
- Combat school bonuses for captures and kills
- Tesuji challenge library with 8 categories
- Life & death puzzle progression (300+ problems)
- "Aggressive player" achievement path
- Capturing race mini-games
- Ko fight mechanics
- "Fighting spirit" momentum system
- Sacrifice-for-advantage calculations

---

## Cross-School Strategic Concepts

These 34 strategic concepts from Senseis Library apply across all schools:

**URL:** `https://senseis.xmp.net/?StrategicConcepts`

### Tier 1: Fundamental Concepts (Beginner Priority)
1. **Territory** - Securing board area
2. **Connection** - Linking stones
3. **Cut** - Separating opponent stones
4. **Attack** - Pressuring enemy groups
5. **Defense** - Protecting your stones
6. **Sente** - Maintaining initiative
7. **Gote** - Responding to opponent
8. **Capture** - Taking opponent stones
9. **Efficiency** - Maximizing value per move
10. **Timing** - Knowing when to act

### Tier 2: Intermediate Concepts
11. **Thickness** - Strong positions
12. **Influence** - Board control
13. **Capturing Race** - Liberty competition
14. **Aji** - Residual potential
15. **Mobility** - Stone flexibility
16. **Flexibility** - Maintaining options
17. **Reduction** - Limiting opponent territory
18. **Sacrifice** - Trading for advantage
19. **Shape** - Stone efficiency patterns

### Tier 3: Advanced Concepts
20. **Miai** - Alternative equal-value plays
21. **Forcing Move** - Compelling responses
22. **Kikashi** - Light forcing moves
23. **Sabaki** - Light, flexible play
24. **Aji Keshi** - Destroying potential
25. **Probe** - Testing opponent response
26. **Tenuki** - Playing elsewhere
27. **Urgent Points** - Critical timing moves
28. **Big Points** - High-value locations
29. **Balance** - Positional equilibrium
30. **Overconcentration** - Inefficient clustering
31. **Light Play** - Flexible, non-committal moves
32. **Heavy Play** - Committed, burdened stones
33. **Haengma** - Intuitive efficiency
34. **Reading** - Mental calculation

**Game Integration:**
- Unlock concepts as skill tree nodes
- Each concept provides specific bonuses
- Cross-school synergies when combining concepts
- Achievement system for mastering concept combinations

---

## Problems and Exercises Database

### Beginner Exercises Collection

**URL:** `https://senseis.xmp.net/?BeginnerExercises`

**Scope:** 346+ beginner problems (with room for expansion to 350)

**Organization:**
- Sequential numbering (Exercise 1 through 346)
- Thumbnail galleries in groups of 50
- Covers: life and death, capturing races, cutting/connecting

**Progressive Structure:**
```
Exercises 1-50:    Absolute beginner fundamentals
Exercises 51-100:  Basic pattern recognition
Exercises 101-150: Elementary tactics
Exercises 151-200: Intermediate beginner concepts
Exercises 201-250: Advanced beginner challenges
Exercises 251-300: Transition to kyu level
Exercises 301-346: Bridge to intermediate play
```

**Note:** Contributors acknowledge difficulty variation; not all problems maintain true beginner level throughout.

### Kyu Exercises Collection

**URL:** `https://senseis.xmp.net/?KyuExercises`

**Scope:** 200+ problems for kyu-level players (30k to 1k)

### Dan Exercises Collection

**URL:** `https://senseis.xmp.net/?DanExercises`

**Scope:** ~100 problems for dan-level players (1d+)

### Problem Organization Systems

**By Difficulty Level:**
- Beginner (300+)
- Kyu (200+)
- Dan (100+)
- Unsolved problems

**By Game Stage:**
- Opening exercises
- Middle game exercises
- Endgame problems

**By Specific Tactics:**
- Ladders
- Capturing races
- Snapback techniques
- Corner patterns
- Stone escape maneuvers
- Eye formation
- Connection problems
- Cutting exercises

**By Essential Concepts:**
- Joseki problems
- Shape recognition
- Miai strategy
- Thickness development
- Positional judgment

**Classical Collections:**
- Historical problem sets from China and Japan
- 14th-20th century collections
- 140-500+ problems per collection
- Cultural and historical context

**Real Game Problems:**
- 170+ tsumego from actual matches
- Professional game positions
- Tournament situations

### Search and Discovery Tools

1. **Random Tsumego Page** - Serendipitous problem discovery
2. **Advanced Search** - Filter by difficulty and keywords
3. **Diagram Position Search** - Find problems by board pattern
4. **Related Pages** - Connected problem sets

**Game Integration Strategy:**

```
Daily Challenge System:
- Random beginner problem (Exercises 1-346)
- Progressive problem sets (unlock sequentially)
- Difficulty scaling based on player performance
- Streak rewards for daily completion

Milestone Unlocks:
- Complete 20 exercises → Unlock Kyu problems
- Complete 30 exercises → Unlock specific tesuji category
- Complete 50 exercises → Unlock classical collection

Problem Categories by School:
- Territory: Endgame problems, counting exercises
- Influence: Opening exercises, whole-board problems
- Combat: Life & death, capturing races, tesuji challenges

Achievement System:
- "Problem Solver" - Complete 100 problems
- "Tactician" - Complete all beginner tesuji problems
- "Life & Death Master" - Complete all L&D exercises
- "Classical Scholar" - Complete a historical collection
```

---

## Skill Tree Mapping to Senseis Library

### Territory School Skill Tree

```
Level 1: Territory Basics
├─ Understanding Territory (Lesson 2)
│  └─ URL: senseis.xmp.net/?Territory
├─ Counting Fundamentals
│  └─ URL: senseis.xmp.net/?Counting
└─ Basic Endgame
   └─ URL: senseis.xmp.net/?Endgame

Level 2: Corner Mastery
├─ 3-3 Joseki (immediate territory)
│  └─ URL: senseis.xmp.net/?3-3Point
├─ 3-4 Joseki (balanced)
│  └─ URL: senseis.xmp.net/?3-4Point
└─ Corner Life & Death
   └─ Exercises 1-50

Level 3: Endgame Excellence
├─ Yose Theory
│  └─ URL: senseis.xmp.net/?Yose
├─ Endgame Tesuji
│  └─ Monkey jump, hanetsugi
└─ Counting and Calculation
   └─ Positional judgment (Lesson 10)

Level 4: Secure Play
├─ Defensive Tesuji
├─ Territory Protection
└─ Efficient Shape
   └─ URL: senseis.xmp.net/?Shape

Capstone: Master of Territory
└─ Complete 100 endgame problems
└─ Master all basic joseki
└─ Perfect counting accuracy
```

### Influence School Skill Tree

```
Level 1: Opening Theory
├─ Fuseki Basics (Lesson 6)
│  └─ URL: senseis.xmp.net/?Fuseki
├─ Whole Board Thinking
│  └─ URL: senseis.xmp.net/?WholeBoardThinking
└─ 4-4 Point Fundamentals
   └─ URL: senseis.xmp.net/?4-4Point

Level 2: Building Power
├─ Thickness Concepts (Lesson 7)
│  └─ URL: senseis.xmp.net/?Thickness
├─ Influence Fundamentals
│  └─ URL: senseis.xmp.net/?Influence
└─ Moyo (Framework) Building
   └─ URL: senseis.xmp.net/?Moyo

Level 3: Pattern Mastery
├─ Chinese Fuseki
│  └─ URL: senseis.xmp.net/?ChineseFuseki
├─ Sanrensei (Three Stars)
│  └─ URL: senseis.xmp.net/?Sanrensei
└─ Opening Exercises
   └─ 50+ opening problems

Level 4: Strategic Depth
├─ Reduction Techniques
│  └─ URL: senseis.xmp.net/?Reduction
├─ Flexibility and Mobility
└─ Converting Influence
   └─ Strategic concepts (Lesson 11)

Capstone: Cosmic Player
└─ Master 5 major fuseki patterns
└─ Complete 100 opening problems
└─ Perfect whole-board vision
```

### Combat School Skill Tree

```
Level 1: Basic Tactics
├─ Liberties & Capturing (Lesson 3)
│  └─ Double atari, ladders, nets
├─ Basic Life & Death (Lessons 4-5)
│  └─ URL: senseis.xmp.net/?LifeAndDeath
└─ Capturing Fundamentals
   └─ Exercises 1-100

Level 2: Tactical Weapons
├─ Tesuji Fundamentals
│  └─ URL: senseis.xmp.net/?Tesuji
│  ├─ Capturing: Net, ladder, squeeze
│  ├─ Attacking: Eye stealing, throw-in
│  └─ Defending: Racing, descent, eyes
├─ Life & Death Patterns
│  └─ L-formation, tripod, carpenter's square
└─ Capturing Races
   └─ URL: senseis.xmp.net/?CapturingRace

Level 3: Advanced Combat
├─ Complex L&D
│  └─ Bent four, twisted four, corner L&D
├─ Ko Fights
│  └─ URL: senseis.xmp.net/?Ko
├─ Shortage of Liberties Tesuji
│  └─ Snapback, belly attachment, twirl
└─ Miai Tesuji
   └─ Grappling hook, nose, clamping

Level 4: Fighting Mastery
├─ Concurrent Fights
│  └─ URL: senseis.xmp.net/?Fighting
├─ Sabaki (Light Play)
│  └─ URL: senseis.xmp.net/?Sabaki
├─ Attack and Defense
└─ Fighting Spirit
   └─ Psychological dimension

Capstone: Combat Master
└─ Complete all 300 beginner L&D exercises
└─ Master all 8 tesuji categories
└─ Perfect reading ability
```

---

## Deep-Linking Strategies

### URL Construction Guidelines

**Standard Page Access:**
```
Pattern: https://senseis.xmp.net/?[PageName]
Example: https://senseis.xmp.net/?Joseki
```

**Topic/Forum Access:**
```
Pattern: https://senseis.xmp.net/?topic=[TopicID]
Example: https://senseis.xmp.net/?topic=832
```

**Specific Exercise Access:**
```
Pattern: https://senseis.xmp.net/?BeginnerExercise[Number]
Example: https://senseis.xmp.net/?BeginnerExercise42
```

### Key Page Reference URLs

**Core Learning Paths:**
- Front page: `https://senseis.xmp.net/`
- Beginners: `https://senseis.xmp.net/?PagesForBeginners`
- Study section: `https://senseis.xmp.net/?BeginnerStudySection`
- Reference: `https://senseis.xmp.net/?ReferenceSection`
- Starting points: `https://senseis.xmp.net/?StartingPoints`

**Territory School URLs:**
- Territory: `https://senseis.xmp.net/?Territory`
- Counting: `https://senseis.xmp.net/?Counting`
- Endgame: `https://senseis.xmp.net/?Endgame`
- Yose: `https://senseis.xmp.net/?Yose`
- 3-3 Point: `https://senseis.xmp.net/?3-3Point`
- 3-4 Point: `https://senseis.xmp.net/?3-4Point`

**Influence School URLs:**
- Influence: `https://senseis.xmp.net/?Influence`
- Fuseki: `https://senseis.xmp.net/?Fuseki`
- Thickness: `https://senseis.xmp.net/?Thickness`
- Moyo: `https://senseis.xmp.net/?Moyo`
- 4-4 Point: `https://senseis.xmp.net/?4-4Point`
- Chinese Fuseki: `https://senseis.xmp.net/?ChineseFuseki`
- Sanrensei: `https://senseis.xmp.net/?Sanrensei`
- Whole Board Thinking: `https://senseis.xmp.net/?WholeBoardThinking`

**Combat School URLs:**
- Fighting: `https://senseis.xmp.net/?Fighting`
- Life and Death: `https://senseis.xmp.net/?LifeAndDeath`
- Tesuji: `https://senseis.xmp.net/?Tesuji`
- Capturing Race: `https://senseis.xmp.net/?CapturingRace`
- Ko: `https://senseis.xmp.net/?Ko`
- Sabaki: `https://senseis.xmp.net/?Sabaki`

**Strategic Concepts:**
- Strategic Concepts: `https://senseis.xmp.net/?StrategicConcepts`
- Shape: `https://senseis.xmp.net/?Shape`
- Playing Style: `https://senseis.xmp.net/?PlayingStyle`
- Opening: `https://senseis.xmp.net/?Opening`

**Problems and Exercises:**
- Problems Index: `https://senseis.xmp.net/?ProblemsAndExercises`
- Beginner Exercises: `https://senseis.xmp.net/?BeginnerExercises`
- Kyu Exercises: `https://senseis.xmp.net/?KyuExercises`
- Dan Exercises: `https://senseis.xmp.net/?DanExercises`
- Tsumego: `https://senseis.xmp.net/?Tsumego`

### Implementation Recommendations

**1. Lesson Unlocks with Deep Links**
```javascript
// Example structure for lesson unlock system
const lessons = [
  {
    id: 1,
    title: "The Rules",
    senseisUrl: "https://senseis.xmp.net/?RulesOfGo",
    description: "Learn the fundamental rules of Go",
    school: "all",
    unlockLevel: 1
  },
  {
    id: 2,
    title: "Territory Concept",
    senseisUrl: "https://senseis.xmp.net/?Territory",
    description: "Understanding how to build and count territory",
    school: "territory",
    unlockLevel: 5
  },
  // ... more lessons
];
```

**2. Problem Integration**
```javascript
// Link to specific problem ranges
const problemSets = {
  beginnerEarly: {
    range: "1-50",
    baseUrl: "https://senseis.xmp.net/?BeginnerExercise",
    description: "Absolute beginner fundamentals"
  },
  beginnerMid: {
    range: "51-150",
    baseUrl: "https://senseis.xmp.net/?BeginnerExercise",
    description: "Basic pattern recognition"
  },
  // ... more sets
};
```

**3. Context-Aware Help System**
```javascript
// Show relevant Senseis Library link based on game context
function getContextualHelp(gameState) {
  if (gameState.phase === "opening") {
    return "https://senseis.xmp.net/?Fuseki";
  } else if (gameState.phase === "endgame") {
    return "https://senseis.xmp.net/?Endgame";
  } else if (gameState.lastMove === "lifeThreat") {
    return "https://senseis.xmp.net/?LifeAndDeath";
  }
  // ... more contexts
}
```

**4. Achievement Links**
```javascript
// Link achievements to relevant Senseis Library pages
const achievements = {
  territoryMaster: {
    title: "Master of Territory",
    learnMoreUrl: "https://senseis.xmp.net/?Territory",
    relatedConcepts: [
      "https://senseis.xmp.net/?Endgame",
      "https://senseis.xmp.net/?Counting"
    ]
  },
  // ... more achievements
};
```

---

## Integration Recommendations for GoGoGo

### 1. Tutorial System Integration

**Implement the 14-Lesson Beginner Study Section as core progression:**

```
Level 1-5 (Foundations):
- Integrate Lessons 0-3 as mandatory tutorials
- Interactive rule demonstrations
- Territory concept through mini-games
- Liberties & capturing practice

Level 6-10 (Tactics):
- Lessons 4-5: Life & death challenges
- Progressive problem solving (Exercises 1-100)
- Pattern recognition rewards
- Tactical achievement unlocks

Level 11-20 (Strategy):
- Lessons 6-9: Opening, balance, shape, fundamentals
- School specialization choice
- Strategic concept unlocks
- Depth-over-breadth rewards

Level 21+ (Mastery):
- Lessons 10-14: Advanced concepts
- Cross-school synergies
- Professional game study
- Master-level challenges
```

### 2. Daily Challenge System

**Leverage Senseis Library's 600+ problems:**

- **Daily Problem:** Random selection from appropriate difficulty tier
- **Weekly Theme:** Focus on specific concepts (e.g., "Capturing Week", "Shape Week")
- **Monthly Classic:** Historical problem from classical collections
- **Streak Rewards:** Bonus for consecutive daily completions
- **Problem Discovery:** Unlock new problem categories as rewards

### 3. School-Specific Content

**Territory School:**
- Joseki library focused on territorial outcomes
- Endgame challenge mode
- Counting mini-game with accuracy scoring
- "Secure Player" achievement path
- Defensive tesuji emphasis

**Influence School:**
- Fuseki pattern library (Chinese, Sanrensei, etc.)
- Moyo-building challenges
- Whole-board strategic puzzles
- "Cosmic Player" achievement path
- Long-term planning bonuses

**Combat School:**
- Life & death puzzle progression (300+ exercises)
- Tesuji challenge by category (8 types)
- Capturing race mini-games
- Ko fight mechanics
- "Fighting Master" achievement path

### 4. Learn More / Help System

**Context-Sensitive Help:**
- Every game screen has "Learn More" button
- Links to relevant Senseis Library pages
- In-game browser or external link option
- Bookmark favorite pages for quick reference

**Concept Encyclopedia:**
- In-game glossary of 34 strategic concepts
- Each entry links to Senseis Library
- Progressive unlock as player learns
- Cross-reference related concepts

### 5. Achievement and Milestone System

**Learning Milestones:**
```
Beginner Path:
- Complete Lesson 1 → "Rules Master"
- Complete 20 problems → "Tactician"
- Complete Lesson 6 → "Opening Student"
- Complete 50 problems → "Problem Solver"
- Complete all 14 lessons → "Graduate"

School Specialization:
- Territory: Complete 100 endgame problems
- Influence: Master 5 fuseki patterns
- Combat: Complete all L&D exercises

Master Path:
- Complete 300+ beginner exercises
- Master all 8 tesuji categories
- Complete classical problem collection
- Study professional games
```

### 6. Problem Library Interface

**Organization:**
```
By Difficulty:
├─ Beginner (1-346)
│  ├─ Early (1-100)
│  ├─ Mid (101-200)
│  └─ Advanced (201-346)
├─ Kyu (200+ problems)
└─ Dan (100+ problems)

By Category:
├─ Life & Death
├─ Capturing Races
├─ Tesuji (8 subcategories)
├─ Opening
├─ Middle Game
└─ Endgame

By School:
├─ Territory Focus
├─ Influence Focus
└─ Combat Focus

Special Collections:
├─ Daily Challenges
├─ Classical Problems
├─ Real Game Situations
└─ Themed Sets
```

### 7. Idle Game Mechanics

**Passive Learning:**
- AI assistants "study" Senseis Library content while offline
- Generate passive currency based on studied topics
- Unlock new concepts through study time
- "Research" mechanic: invest time to unlock new skills

**Active Learning:**
- Solve problems for immediate rewards
- Complete lessons for permanent bonuses
- Master concepts for multiplier effects
- Chain-solving problems for combo bonuses

**School Progression:**
- Each school has dedicated skill tree
- Unlock branches by completing relevant Senseis content
- Cross-school synergies for varied learning
- Prestige system: reset with bonuses after mastery

### 8. Social Features

**Share Progress:**
- Share interesting problems from Senseis Library
- Link to specific lessons in chat
- Create custom problem collections
- Challenge friends with specific exercises

**Community Learning:**
- Recommended problems from experienced players
- Curated learning paths
- Discussion forums linked to Senseis pages
- Study groups focused on specific concepts

### 9. Attribution Display

**Proper Credit Throughout:**
- "Powered by Senseis Library" badge
- Link to source page for every lesson/problem
- In-game credit section
- "Explore More at Senseis Library" call-to-action

---

## Licensing and Attribution Requirements

### Open Content License

**Key Points:**
- Senseis Library content uses the **Open Content License**
- Contributors grant rights to copy, distribute, and modify material
- Must respect original authors' wishes regarding quotations
- Images and specific contributions have individual copyright holders

### Attribution Best Practices

**1. Every Lesson/Problem Page:**
```
Content source: Sensei's Library
URL: [specific page URL]
License: Open Content License
Visit senseis.xmp.net for more Go resources
```

**2. Credits Screen:**
```
GoGoGo uses content from Sensei's Library (senseis.xmp.net),
a collaborative wiki dedicated to the game of Go.

Sensei's Library is maintained by:
- Arno Hollosi
- Morten Pahle
- The global Go community

Images of stones © Andreas Fecke, used by permission

All Sensei's Library content is available under the
Open Content License.

Special thanks to the thousands of contributors who
have made Sensei's Library an invaluable resource
for Go players worldwide.
```

**3. In-Game Attribution:**
- Footer text: "Learning content from Sensei's Library"
- Clickable logo linking to senseis.xmp.net
- "Source" button on every lesson page
- Problem pages show original Senseis Library link

**4. External Marketing:**
- Mention Senseis Library in app description
- Credit in promotional materials
- Link in social media posts about learning features
- Acknowledge in press releases

### Recommended Attribution Text

**Short Form (in-game footer):**
```
Content from Sensei's Library (senseis.xmp.net) • Open Content License
```

**Medium Form (lesson/problem pages):**
```
This lesson is based on content from Sensei's Library,
a comprehensive wiki dedicated to the game of Go.
Learn more at senseis.xmp.net
```

**Long Form (credits screen):**
```
Educational Content Attribution

GoGoGo's learning system is built upon the extensive
resources of Sensei's Library (senseis.xmp.net), one
of the most comprehensive Go resources on the web.

Sensei's Library is a collaborative wiki containing
over 26,000 pages of Go knowledge, maintained by
Arno Hollosi, Morten Pahle, and thousands of
contributors from the global Go community.

We deeply appreciate the open content philosophy that
makes resources like Sensei's Library available for
projects like ours. We encourage all players to
explore senseis.xmp.net for deeper Go knowledge.

All Sensei's Library content is licensed under the
Open Content License, allowing copy, distribution,
and modification with proper attribution.

Special thanks to Andreas Fecke for stone images
used on Sensei's Library.
```

### Contact Information

**For licensing questions:**
- Contact Sensei's Library maintainers
- Email information available on site
- Forum for copyright/license issues: `senseis.xmp.net/?topic=2819`

### Compliance Checklist

- [ ] Every lesson page links back to source
- [ ] Every problem links back to Senseis Library
- [ ] Credits screen includes full attribution
- [ ] Footer includes short attribution
- [ ] App store description mentions Senseis Library
- [ ] Marketing materials credit the source
- [ ] No claim of original authorship for Senseis content
- [ ] Respect original authors' wishes
- [ ] Maintain Open Content License spirit
- [ ] Regular review of attribution completeness

---

## Practical Implementation Roadmap

### Phase 1: Foundation (MVP)

**Week 1-2: Core Tutorial**
- [ ] Implement Lessons 0-3 as interactive tutorials
- [ ] Link each lesson to Senseis Library page
- [ ] Add "Learn More" buttons to all tutorial screens
- [ ] Implement basic attribution footer

**Week 3-4: Problem System**
- [ ] Import beginner exercises 1-100 (with links)
- [ ] Build problem display interface
- [ ] Implement solution checking
- [ ] Add daily challenge system

**Week 5-6: Credits and Attribution**
- [ ] Create comprehensive credits screen
- [ ] Ensure all content has source links
- [ ] Test all Senseis Library URLs
- [ ] Add proper attribution to all screens

### Phase 2: School System (Version 1.0)

**Week 7-8: School Foundations**
- [ ] Import school-specific content from Senseis
- [ ] Build skill trees mapped to Senseis pages
- [ ] Create school-specific problem sets
- [ ] Implement school selection system

**Week 9-10: Advanced Lessons**
- [ ] Implement Lessons 4-9 (tactics & strategy)
- [ ] Add joseki library with Senseis links
- [ ] Create opening pattern collection
- [ ] Build life & death challenge mode

**Week 11-12: Content Expansion**
- [ ] Import exercises 101-200
- [ ] Add tesuji category system (8 types)
- [ ] Implement strategic concept encyclopedia
- [ ] Create themed problem collections

### Phase 3: Mastery System (Version 2.0)

**Week 13-14: Advanced Content**
- [ ] Implement Lessons 10-14 (mastery)
- [ ] Import kyu-level exercises
- [ ] Add professional game study feature
- [ ] Create advanced achievement system

**Week 15-16: Polish and Expansion**
- [ ] Import complete beginner exercise set (1-346)
- [ ] Add classical problem collections
- [ ] Implement real game situation problems
- [ ] Create custom problem collection feature

**Week 17-18: Social Features**
- [ ] Add problem sharing functionality
- [ ] Create study group system
- [ ] Implement recommended learning paths
- [ ] Add discussion forum integration

---

## Content Mapping Quick Reference

### By Game Level

| Player Level | Senseis Library Content | URL Pattern |
|-------------|------------------------|-------------|
| 1-5 (Tutorial) | Lessons 0-3, Exercises 1-50 | `?BeginnerStudySection`, `?BeginnerExercise[1-50]` |
| 6-10 (Beginner) | Lessons 4-5, Exercises 51-150 | `?LifeAndDeath`, `?BeginnerExercise[51-150]` |
| 11-20 (Advanced Beginner) | Lessons 6-9, Exercises 151-250 | `?Fuseki`, `?Shape`, `?BeginnerExercise[151-250]` |
| 21-30 (Intermediate) | Lessons 10-14, Exercises 251-346 | `?StrategicConcepts`, `?BeginnerExercise[251-346]` |
| 31+ (Advanced) | Kyu/Dan Exercises, Classical Collections | `?KyuExercises`, `?DanExercises` |

### By School Focus

| School | Primary URLs | Key Concepts | Exercise Focus |
|--------|-------------|--------------|----------------|
| Territory | `?Territory`, `?Endgame`, `?Yose`, `?Counting` | Secure points, efficiency, calculation | Endgame problems, counting exercises |
| Influence | `?Influence`, `?Fuseki`, `?Thickness`, `?Moyo` | Power, potential, whole-board | Opening problems, strategic puzzles |
| Combat | `?Fighting`, `?LifeAndDeath`, `?Tesuji` | Tactics, killing, survival | L&D exercises, tesuji problems, capturing races |

### By Learning Goal

| Goal | Senseis Library Resources | Recommended Path |
|------|--------------------------|------------------|
| Learn Rules | `?RulesOfGo`, `?PagesForBeginners` | Lesson 1 → Basic exercises 1-20 |
| Master Life & Death | `?LifeAndDeath`, `?BeginnerExercises` | Lessons 4-5 → All 300 beginner L&D problems |
| Opening Mastery | `?Fuseki`, `?Joseki`, `?Opening` | Lesson 6 → Opening exercises → Fuseki patterns |
| Tactical Excellence | `?Tesuji`, `?Tsumego` | Lesson 3 → All 8 tesuji categories → 300+ problems |
| Strategic Depth | `?StrategicConcepts` | Lesson 11 → All 34 concepts → Integration practice |
| Endgame Expertise | `?Endgame`, `?Yose` | Lesson 13 → Endgame problems → Counting mastery |

---

## Technical Implementation Notes

### URL Encoding

All Senseis Library URLs use CamelCase without spaces:
```javascript
// Convert display name to Senseis URL
function toSenseisUrl(displayName) {
  const basePage = displayName.replace(/\s+/g, '');
  return `https://senseis.xmp.net/?${basePage}`;
}

// Examples:
// "Life and Death" → "https://senseis.xmp.net/?LifeAndDeath"
// "Strategic Concepts" → "https://senseis.xmp.net/?StrategicConcepts"
```

### Link Validation

Some pages use alternate naming:
```javascript
const pageAliases = {
  "Endgame": ["Endgame", "Yose"],
  "Opening": ["Opening", "Fuseki", "Joban"],
  "Problems": ["ProblemsAndExercises", "Tsumego"],
  // ... more aliases
};
```

### Content Caching Strategy

**Don't scrape/cache Senseis Library content:**
- Always link to live pages
- Maintain attribution through direct links
- Respect Open Content License spirit
- Keep content fresh and updated

**Do cache:**
- List of available exercises
- Problem metadata (difficulty, category)
- Lesson structure and progression
- URL mappings and aliases

### Browser Integration

```javascript
// Open Senseis Library in appropriate context
function openSenseisPage(pageName, inAppBrowser = true) {
  const url = toSenseisUrl(pageName);

  if (inAppBrowser) {
    // Open in-game browser with back button
    openInAppBrowser(url, {
      showBackButton: true,
      attribution: "Content from Sensei's Library"
    });
  } else {
    // Open in external browser
    window.open(url, '_blank');
  }
}
```

---

## Conclusion

Sensei's Library provides an extraordinary foundation for GoGoGo's educational content. With over 26,000 pages, structured lessons, 600+ problems, and comprehensive coverage of all Go aspects, it offers everything needed to create a deep, engaging learning experience.

**Key Takeaways:**

1. **Structured Curriculum**: The 14-lesson Beginner Study Section provides a complete progression path that maps perfectly to our milestone system.

2. **Massive Problem Database**: 346+ beginner exercises, plus 200+ kyu and 100+ dan problems offer endless content for daily challenges and skill progression.

3. **School Alignment**: Territory, Influence, and Combat schools map naturally to Senseis Library's content organization.

4. **Strategic Depth**: 34 strategic concepts provide a rich framework for skill trees and cross-school synergies.

5. **Open Content License**: Proper attribution allows us to leverage this incredible resource while respecting the community's contributions.

**Next Steps:**

1. Implement Phase 1 (Foundation) as outlined in the roadmap
2. Create attribution system across all content
3. Build problem library interface
4. Develop school-specific skill trees
5. Integrate daily challenges and achievements

By properly crediting and linking to Sensei's Library, we can create an idle game that's not just entertaining, but genuinely educational—helping players become better Go players while they play.

---

**Document Version:** 1.0
**Last Updated:** 2025-11-23
**Maintained By:** GoGoGo Development Team
**Primary Source:** Sensei's Library (https://senseis.xmp.net)
