---
layout: default
title: Glossary
permalink: /glossary/
---

# Glossary

> 知彼知己、百戦不殆
> *Know the enemy and know yourself; in a hundred battles you will never be in peril*

Terms from Go, AI, and the spaces between.

---

## Go Terms

### The 8 Basic Instincts
{: #basic-instincts}
Fundamental responses that strong players execute without thinking. Patterns drilled until they become reflex.
[Sensei's](https://senseis.xmp.net/?BasicInstinct)

### 1. Extend from Atari
{: #extend-from-atari}
アタリから伸びよ — When your stone faces capture (one liberty), extend to gain liberties. The most fundamental survival instinct.
[Sensei's](https://senseis.xmp.net/?ExtendFromAtari)

### 2. Hane vs Tsuke
{: #hane-vs-tsuke}
ツケにはハネ — When opponent plays an unsupported adjacent stone (tsuke), respond with a hane to block their development and reduce their liberties.
[Sensei's](https://senseis.xmp.net/?HaneResponseToTsuke)

### 3. Hane at Head of Two
{: #hane-at-head-of-two}
二子の頭にハネ — Play above two consecutive opponent stones to create weakness and reduce their liberties.
[Sensei's](https://senseis.xmp.net/?HaneAtTheHeadOfTwoStones)

### 4. Stretch from Kosumi
{: #stretch-from-kosumi}
コスミから伸びよ — When opponent plays a diagonal attachment (kosumi-tsuke), stretch away rather than hane to deny them a powerful tiger mouth shape.
[Sensei's](https://senseis.xmp.net/?StretchFromAKosumiTsuke)

### 5. Block the Angle
{: #block-the-angle}
カケにはオサエ — Respond to diagonal threats by blocking diagonally, increasing your liberties while decreasing opponent's.
[Sensei's](https://senseis.xmp.net/?BlockTheAnglePlay)

### 6. Connect vs Peep
{: #connect-vs-peep}
ノゾキにはツギ — "Even a moron connects against a peep." When opponent peeps between your stones, connect immediately.
[Sensei's](https://senseis.xmp.net/?ConnectAgainstAPeep)

### 7. Block the Thrust
{: #block-the-thrust}
ツキアタリには — When opponent thrusts between your stones, block to let them choose which side to cut.
[Sensei's](https://senseis.xmp.net/?BlockTheThrust)

### 8. Stretch from Bump
{: #stretch-from-bump}
ブツカリから伸びよ — When opponent bumps (supported attachment), stretch rather than hane since their attachment is reinforced.
[Sensei's](https://senseis.xmp.net/?StretchFromABump)

---

### Atari
{: #atari}
A group with only one liberty remaining. One move from capture.
[Sensei's](https://senseis.xmp.net/?Atari)

### Block
{: #block}
Preventing the opponent from extending or connecting. Denying their path.
[Sensei's](https://senseis.xmp.net/?Block)

### Capture
{: #capture}
Taking opponent stones by filling their last liberty. The most basic tactical goal.
[Sensei's](https://senseis.xmp.net/?Capture)

### Connect
{: #connect}
Joining two friendly groups together. Strength through unity.
[Sensei's](https://senseis.xmp.net/?Connect)

### Cut
{: #cut}
Separating opponent stones into disconnected groups. Divide and conquer.
[Sensei's](https://senseis.xmp.net/?Cut)

### Dame
{: #dame}
Neutral points that belong to neither player. Empty intersections surrounded by both colors.
[Sensei's](https://senseis.xmp.net/?Dame)

### Defend
{: #defend}
Protecting a weak point or cutting point. Reinforcing before the opponent exploits.
[Sensei's](https://senseis.xmp.net/?Defend)

### Escape
{: #escape}
Saving stones in atari by extending or connecting. Running to safety.
[Sensei's](https://senseis.xmp.net/?Escape)

### Extend
{: #extend}
Growing outward from existing stones. Gaining liberties and influence.
[Sensei's](https://senseis.xmp.net/?Extend)

### Hane
{: #hane}
A move that goes around the opponent's stone diagonally. Often used to block development.
[Sensei's](https://senseis.xmp.net/?Hane)

### Kosumi
{: #kosumi}
A diagonal move one point away from your own stone. Creates a flexible connection.
[Sensei's](https://senseis.xmp.net/?Kosumi)

### Peep
{: #peep}
A move threatening to cut between two enemy stones. Forces a response.
[Sensei's](https://senseis.xmp.net/?Peep)

### Tsuke
{: #tsuke}
Playing directly adjacent to an opponent's stone. An attachment that creates contact fighting.
[Sensei's](https://senseis.xmp.net/?Tsuke)

### Joseki
{: #joseki}
Established sequences of moves, usually in corners. Local equality through studied patterns.
[Sensei's](https://senseis.xmp.net/?Joseki)

### Ko
{: #ko}
A position where capture would recreate the previous board state. Forbidden to retake immediately.
[Sensei's](https://senseis.xmp.net/?Ko)

### Ladder
{: #ladder}
A capturing sequence where a group is chased diagonally across the board.
[Sensei's](https://senseis.xmp.net/?Ladder)

### Liberty
{: #liberty}
An empty point adjacent to a stone or group. Stones live by their liberties.
[Sensei's](https://senseis.xmp.net/?Liberty)

### Seki
{: #seki}
Mutual life. Two groups that cannot capture each other without dying first.
[Sensei's](https://senseis.xmp.net/?Seki)

### Snapback
{: #snapback}
A sacrifice that enables immediate recapture of a larger group.
[Sensei's](https://senseis.xmp.net/?Snapback)

### Tenuki
{: #tenuki}
Playing elsewhere. Ignoring the local situation for a bigger point.
[Sensei's](https://senseis.xmp.net/?Tenuki)

### Tsumego
{: #tsumego}
Life and death problems. Reading exercises for tactical strength.
[Sensei's](https://senseis.xmp.net/?Tsumego)

---

## AI Terms

### AlphaGo / AlphaZero
{: #alphago--alphazero}
DeepMind's Go-playing AI. AlphaZero learned purely from self-play, no human games.
[Wikipedia](https://en.wikipedia.org/wiki/AlphaGo)

### Backpropagation
{: #backpropagation}
The algorithm that teaches neural networks. Gradients flow backward, adjusting weights.
[Wikipedia](https://en.wikipedia.org/wiki/Backpropagation)

### Credit Assignment
{: #credit-assignment}
The problem of determining which decisions caused success or failure. Hundreds of moves, one outcome.
[Wikipedia](https://en.wikipedia.org/wiki/Credit_assignment_problem)

### Curriculum Learning
{: #curriculum-learning}
Training on easy examples first, gradually increasing difficulty. Learn to crawl before you run.
[Wikipedia](https://en.wikipedia.org/wiki/Curriculum_learning)

### KataGo
{: #katago}
Open-source Go AI that trains 50× faster than AlphaZero through architectural innovations.
[KataGo Paper](https://arxiv.org/abs/1902.10565)

### Loss Function
{: #loss-function}
Measures how wrong predictions are. Training minimizes this. Lower loss, better model.
[Wikipedia](https://en.wikipedia.org/wiki/Loss_function)

### MCTS
{: #mcts}
Monte Carlo Tree Search. Build a tree through random sampling, then choose the most-visited path.
[Wikipedia](https://en.wikipedia.org/wiki/Monte_Carlo_tree_search)

### Policy Network
{: #policy-network}
Predicts move probabilities. "Where would a strong player look?"
[AlphaGo Paper](https://www.nature.com/articles/nature16961)

### ResNet
{: #resnet}
Residual Network. Skip connections that let gradients flow through deep networks.
[Wikipedia](https://en.wikipedia.org/wiki/Residual_neural_network)

### Self-Play
{: #self-play}
Training by playing against yourself. No human data needed. Pure bootstrap learning.
[Wikipedia](https://en.wikipedia.org/wiki/Self-play)

### Sparse Features
{: #sparse-features}
Input planes that are mostly zeros. The network sees nothing, learns nothing from them.

### UCB
{: #ucb}
Upper Confidence Bound. Balances exploitation (known good moves) with exploration (untried moves).
[Wikipedia](https://en.wikipedia.org/wiki/Monte_Carlo_tree_search#Exploration_and_exploitation)

### Value Network
{: #value-network}
Predicts win probability from a position. "Who is winning here?"
[AlphaGo Paper](https://www.nature.com/articles/nature16961)

---

## Computer Science

### Big-O Notation
{: #big-o-notation}
Describes algorithm efficiency. O(n) scales linearly, O(n²) scales quadratically. Smaller is faster.
[Wikipedia](https://en.wikipedia.org/wiki/Big_O_notation)

### Complexity
{: #complexity}
How resource usage (time, memory) grows with input size. The difference between possible and impossible.
[Wikipedia](https://en.wikipedia.org/wiki/Computational_complexity)

---

## Project Terms

### Ownership Head
{: #ownership-head}
Auxiliary network output predicting final ownership of each board point.
[KataGo Paper](https://arxiv.org/abs/1902.10565)

### Neurosymbolic
{: #neurosymbolic}
Combining neural networks (pattern recognition) with symbolic reasoning (exact calculation).

### Feature Plane
{: #feature-plane}
One channel of the neural network input. Each plane encodes one aspect of the board state.

---

> 碁の上手は碁を教えない
> *A Go master does not teach Go*

They show. You see.
