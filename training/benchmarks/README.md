# Go Model Benchmark Suite

This directory contains test positions for evaluating Go model quality.

## Directory Structure

```
benchmarks/
├── capture/       # Basic capture problems
├── ladder/        # Ladder recognition
├── snapback/      # Snapback detection
├── life_death/    # Group survival problems
└── joseki/        # Opening patterns
```

## Position Format

Each subdirectory contains JSON files with test positions:

```json
{
  "name": "Simple corner capture",
  "board_size": 9,
  "black_stones": [[0, 0], [0, 1]],
  "white_stones": [[1, 0]],
  "to_play": "black",
  "expected_moves": [[1, 1]],
  "category": "capture",
  "difficulty": "easy"
}
```

## Running Benchmarks

```bash
poetry run python benchmark.py --checkpoint checkpoints/model.pt
```

## Metrics

- **Top-1 Accuracy**: Expected move is the model's #1 choice
- **Top-5 Accuracy**: Expected move is in model's top 5
- **Category Breakdown**: Performance by problem type
