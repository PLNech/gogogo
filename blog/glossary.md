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

### Atari
{: #atari}
A group with only one liberty remaining. One move from capture.
[Sensei's](https://senseis.xmp.net/?Atari)

### Dame
{: #dame}
Neutral points that belong to neither player. Empty intersections surrounded by both colors.
[Sensei's](https://senseis.xmp.net/?Dame)

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
