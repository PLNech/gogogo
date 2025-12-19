---
layout: post
title: "From Python to Browser"
date: 2025-12-19
phase: "Integration"
excerpt: "Exporting neural networks to ONNX, building benchmarks, and the joy of seeing your model play in the browser."
---

> 温故知新 — *Review the old to understand the new*

The model exists. It trains. It learns (slowly). But it lives in Python, trapped in PyTorch tensors. Today we set it free.

![From Python to Browser](/images/python-to-browser.png)

---

## The Export Pipeline

PyTorch to ONNX. A single script.

```python
# training/export_model.py
class GoNetExport(nn.Module):
    """Simplified wrapper for ONNX export."""

    def forward(self, x):
        # Policy: softmax probabilities
        # Value: tanh [-1, 1]
        return policy, value
```

The model shrinks: 22MB PyTorch → 7.3MB ONNX.

We verify with `onnxruntime`:

```
Top 5 policy moves:
  C8: 0.0146
  D2: 0.0146
  G2: 0.0145
```

Uniform distribution. The model knows nothing yet. But it runs.

---

## Measuring Truth

A model without measurement is faith.

We build a benchmark suite. Categories:
- **Capture**: Can it see one move ahead?
- **Ladder**: Can it read the diagonal?
- **Snapback**: Can it see the trap?

```
training/benchmarks/
├── capture/basic_captures.json   # 8 positions
├── ladder/ladders.json           # 5 positions
└── snapback/snapbacks.json       # 4 positions
```

Results on our 9x9 self-play model:

```
Overall (17 positions):
  Top-1 Accuracy: 17.6%
  Top-5 Accuracy: 17.6%

By Category:
  Capture    8 pos | Top-1:  0.0%
  Ladder     5 pos | Top-1: 20.0%
  Snapback   4 pos | Top-1: 50.0%
```

Zero percent on captures. The most basic skill. Humbling.

But now we can measure. And what gets measured gets improved.

---

## Into the Browser

ONNX Runtime Web. The bridge between worlds.

```typescript
// src/core/ai/neural.ts
import * as ort from 'onnxruntime-web'

export async function loadNeuralModel() {
  ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/'

  return await ort.InferenceSession.create(MODEL_PATH, {
    executionProviders: ['wasm'],
    graphOptimizationLevel: 'all'
  })
}
```

The WASM binary: 23MB. Lazy-loaded. Gzipped to 5.6MB over the wire.

First load takes seconds. Subsequent inference: milliseconds.

---

## Level 6: Neural

Five levels of heuristic AI. Now a sixth.

```typescript
// src/core/ai/types.ts
6: {
  level: 6,
  searchDepth: 0,      // Policy directly
  randomness: 0.1,     // Small temperature
  useNeural: true,
}
```

> *frost on the GPU*
> *use neural — centaurs build dreams*
> *paperclips can wait*

The async dance:

```typescript
export async function getAIDecisionAsync(board, player, config) {
  if (config.useNeural) {
    if (!isNeuralModelLoaded()) {
      await loadNeuralModel()  // First call loads
    }
    return await getNeuralAIDecision(board, player)
  }
  return getAIDecision(board, player, config)  // Fallback
}
```

Select Level 6. Click Play. Watch the console:

```
[AI] Loading neural model...
[Neural] Loading model from /gogogo/play/models/go_9x9/model.onnx
[Neural] Model loaded successfully
[Neural] Inputs: [board]
[Neural] Outputs: [policy, value]
```

It works. The model plays. Badly, but it plays.

---

## The 9x9 Problem

Our model is 9x9. Fixed input shape. It cannot play 19x19.

The solution: metadata.

```json
// public/models/go_9x9/metadata.json
{
  "name": "GoGoGo Neural v0.1",
  "board_size": 9,
  "input_planes": 17,
  "training": {
    "method": "self-play",
    "games": 100
  },
  "notes": "Early model, weak but functional."
}
```

The UI must check. Neural AI only available when board size matches.

Future work: tricks to generalize. Padding? Multiple models? The research continues.

---

## Testing with Playwright

Automated browser testing. Headful mode—we watch.

```typescript
// tests/neural-ai.spec.ts
test('Level 6 vs Level 6 neural game', async ({ page }) => {
  await page.goto(BASE_URL)
  await page.getByRole('button', { name: 'Watch' }).click()

  // Select Level 6 for both AIs
  await selects.first().selectOption('6')
  await selects.nth(1).selectOption('6')

  // Click Play
  await playBtn.click()

  // Watch the game unfold...
})
```

The browser opens. Clicks happen. The model loads. Stones appear.

Magic.

---

## Current State

**What works:**
- ONNX export pipeline
- Benchmark suite (17 positions)
- Browser inference via ONNX Runtime Web
- Level 6 neural AI in Watch page
- Playwright test infrastructure

**What's weak:**
- Model accuracy: 17.6%
- Only 9x9 supported
- No board size validation yet

**Files created:**
```
training/export_model.py      # PyTorch → ONNX
training/benchmark.py         # Evaluation suite
training/benchmarks/          # Test positions
src/core/ai/neural.ts         # Browser inference
public/models/go_9x9/         # Model + metadata
tests/neural-ai.spec.ts       # Playwright tests
```

---

## Next

Train more. Export again. Measure. Repeat.

The feedback loop is closed. Python trains, browser plays, benchmarks measure.

Now we iterate.

> 千里の道も一歩から — *A journey of a thousand miles begins with a single step*

We have taken that step. The model plays in the browser. It plays poorly.

But it plays.
