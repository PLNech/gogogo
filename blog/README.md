# GoGoGo - Chronicle

> 一石万局
> *One stone, ten thousand games*

---

A journey through AI and Go. Building intelligence through play, making oneself stronger by teaching machines to see, discovering patterns where none seemed to exist.

## Philosophy

**Primary Goal**: Become stronger at Go through targeted practice and study.

**Secondary Goal**: Build a capable amateur AI (10-5 kyu) using modern techniques, documenting everything for understanding and reflection.

**Approach**: Test-Driven Development, clean architecture, SOTA board representation, and rigorous experimentation. Every feature earns its place. Every deliverable proves itself through code.

## Chronicle

### Phase 0: Foundation

- [**First Stone**](posts/2025-11-28-first-stone.md) (Nov 28, 2025) - The empty board awaits. A project begins.
- [**Foundation**](posts/2025-11-30-foundation.md) (Nov 30, 2025) - Tests first. Simplicity through subtraction. All tests green.
- [**Test Fixtures**](posts/2025-12-01-test-fixtures.md) (Dec 1, 2025) - Building test infrastructure with patterns and clarity.

### Coming Soon

- **Phase 1**: Board Representation - SOTA feature extraction
- **Phase 2**: Heuristic Policy - Move priors, evaluation, rollouts
- **Phase 3**: Enhanced MCTS - Tree search with policy integration
- **Phase 4**: Analysis Tools - CLI visualization, game records
- **Phase 5**: Strength Building - Tuning to 10-5 kyu
- **Phase 6**: Educational Games - Tsumego, joseki, reading practice

## Architecture

```
┌─────────────────────────────────┐
│   MCTS Tree Search Engine       │
│   guided by                     │
├─────────────────────────────────┤
│   Heuristic Policy Network      │
│   operating on                  │
├─────────────────────────────────┤
│   Multi-plane Board Features    │
│   (liberty planes, captures,    │
│    ko status, group analysis)   │
└─────────────────────────────────┘
```

MCTS provides tactical reading.
Heuristics provide strategic intuition.
Together: a capable opponent, a patient teacher.

---

> 碁は対話
> *Go is conversation*

Built with simplicity. Served with intention.
