#!/usr/bin/env python3
"""Create a shareable training summary image."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import json
import numpy as np

# Load metrics
with open('training_plots/metrics_20251201_195245.json', 'r') as f:
    metrics = json.load(f)

# Setup figure with custom layout
fig = plt.figure(figsize=(16, 12), facecolor='#1a1a2e')
gs = GridSpec(3, 3, figure=fig, height_ratios=[0.8, 1, 1], hspace=0.35, wspace=0.3)

# Color scheme
COLORS = {
    'bg': '#1a1a2e',
    'card': '#16213e',
    'accent': '#e94560',
    'text': '#eaeaea',
    'green': '#4ecca3',
    'blue': '#00b4d8',
    'orange': '#ff9f1c',
    'purple': '#9d4edd'
}

# Title
fig.suptitle('GoGoGo Neural Network - Training Summary',
             fontsize=22, fontweight='bold', color=COLORS['text'], y=0.98)

# ===== TOP ROW: Architecture Overview =====
ax_arch = fig.add_subplot(gs[0, :])
ax_arch.set_facecolor(COLORS['card'])
ax_arch.axis('off')

arch_text = """
ALPHAZERO-STYLE NETWORK WITH KATAGO IMPROVEMENTS

INPUT (27 planes)              BACKBONE                              OUTPUT HEADS
+---------------+        +------------------+        +----------------------------------------+
| 8 stone hist  |        |                  |        |  Policy Head    -> 362 moves (softmax) |
| 8 liberties   |   ->   |  6 ResBlocks     |   ->   |  Value Head     -> win prob (tanh)     |
| 1 color       |        |  128 filters     |        |  Ownership Head -> 19x19 territory     |
| 10 tactical   |        |  + GlobalPool    |        |  Opponent Head  -> 362 moves           |
+---------------+        +------------------+        +----------------------------------------+

TRAINING TECHNIQUES: Curriculum Learning | Ownership Prediction | Opponent Move | Mixed Precision
"""
ax_arch.text(0.5, 0.5, arch_text, fontsize=11, fontfamily='monospace',
             color=COLORS['text'], ha='center', va='center',
             transform=ax_arch.transAxes)

# ===== MIDDLE LEFT: Training Loss =====
ax_loss = fig.add_subplot(gs[1, 0])
ax_loss.set_facecolor(COLORS['card'])
epochs = range(1, len(metrics['train_losses']) + 1)
ax_loss.plot(epochs, metrics['train_losses'], 'o-', color=COLORS['blue'], linewidth=2.5, markersize=7)
ax_loss.fill_between(epochs, metrics['train_losses'], alpha=0.3, color=COLORS['blue'])
ax_loss.set_xlabel('Epoch', color=COLORS['text'], fontsize=12)
ax_loss.set_ylabel('Loss', color=COLORS['text'], fontsize=12)
ax_loss.set_title('Training Loss', color=COLORS['text'], fontsize=14, fontweight='bold', pad=10)
ax_loss.tick_params(colors=COLORS['text'])
ax_loss.grid(True, alpha=0.2, color=COLORS['text'])
for spine in ax_loss.spines.values():
    spine.set_color(COLORS['text'])
    spine.set_alpha(0.3)

# Annotate
ax_loss.annotate(f'Start: {metrics["train_losses"][0]:.1f}',
                 xy=(1, metrics['train_losses'][0]), xytext=(4, 5.7),
                 color=COLORS['orange'], fontsize=10, fontweight='bold',
                 arrowprops=dict(arrowstyle='->', color=COLORS['orange'], lw=1.5))
ax_loss.annotate(f'Final: {metrics["train_losses"][-1]:.1f}',
                 xy=(20, metrics['train_losses'][-1]), xytext=(14, 4.4),
                 color=COLORS['green'], fontsize=10, fontweight='bold',
                 arrowprops=dict(arrowstyle='->', color=COLORS['green'], lw=1.5))

# ===== MIDDLE CENTER: Validation Accuracy =====
ax_acc = fig.add_subplot(gs[1, 1])
ax_acc.set_facecolor(COLORS['card'])
accuracies = [a * 100 for a in metrics['val_accuracies']]
ax_acc.plot(epochs, accuracies, 'o-', color=COLORS['green'], linewidth=2.5, markersize=7)
ax_acc.fill_between(epochs, accuracies, alpha=0.3, color=COLORS['green'])
ax_acc.set_xlabel('Epoch', color=COLORS['text'], fontsize=12)
ax_acc.set_ylabel('Accuracy (%)', color=COLORS['text'], fontsize=12)
ax_acc.set_title('Move Prediction Accuracy', color=COLORS['text'], fontsize=14, fontweight='bold', pad=10)
ax_acc.tick_params(colors=COLORS['text'])
ax_acc.grid(True, alpha=0.2, color=COLORS['text'])
for spine in ax_acc.spines.values():
    spine.set_color(COLORS['text'])
    spine.set_alpha(0.3)

# Mark best and curriculum transitions
best_epoch = accuracies.index(max(accuracies)) + 1
ax_acc.axhline(y=max(accuracies), color=COLORS['accent'], linestyle='--', alpha=0.7, lw=2)
ax_acc.annotate(f'BEST: {max(accuracies):.1f}%',
                xy=(best_epoch, max(accuracies)), xytext=(best_epoch-6, max(accuracies)+1.2),
                color=COLORS['accent'], fontsize=11, fontweight='bold',
                arrowprops=dict(arrowstyle='->', color=COLORS['accent'], lw=1.5))

# Curriculum transitions
ax_acc.axvline(x=4, color=COLORS['orange'], linestyle=':', alpha=0.8, lw=2)
ax_acc.axvline(x=7, color=COLORS['orange'], linestyle=':', alpha=0.8, lw=2)
ax_acc.text(2.5, 16, '100%\ntactical', color=COLORS['orange'], fontsize=9, ha='center', fontweight='bold')
ax_acc.text(5.5, 16, '70%', color=COLORS['orange'], fontsize=9, ha='center', fontweight='bold')
ax_acc.text(13, 16, '50% tactical', color=COLORS['orange'], fontsize=9, ha='center', fontweight='bold')

# ===== MIDDLE RIGHT: Stats Box =====
ax_stats = fig.add_subplot(gs[1, 2])
ax_stats.set_facecolor(COLORS['card'])
ax_stats.axis('off')

stats_text = """TRAINING STATS
--------------
Parameters:  2.5M
Dataset:     300K positions
Duration:    27 minutes
Epochs:      20

RESULTS
-------
Loss:     6.1 -> 4.1
Accuracy: 22.7%
Task:     1 of 361 moves

BENCHMARKS
----------
Random:   0.3%
Us:       22.7% <--
Amateur:  ~35%
Pro:      ~55%"""
ax_stats.text(0.05, 0.5, stats_text, fontsize=11, fontfamily='monospace',
              color=COLORS['text'], ha='left', va='center',
              transform=ax_stats.transAxes, linespacing=1.2)

# ===== BOTTOM LEFT+CENTER: Loss Components =====
ax_components = fig.add_subplot(gs[2, :2])
ax_components.set_facecolor(COLORS['card'])

# Loss components from training logs
policy_loss = [5.97, 3.6, 3.4, 3.3, 3.2, 3.1, 3.0, 2.95, 2.9, 2.8, 2.75, 2.7, 2.65, 2.6, 2.55, 2.5, 2.5, 2.5, 2.5, 2.5]
ownership_loss = [1.4, 0.65, 0.63, 0.70, 0.70, 0.70, 0.74, 0.74, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75]
opponent_loss = [6.0, 5.3, 5.1, 4.95, 4.9, 4.85, 4.75, 4.75, 4.7, 4.7, 4.7, 4.7, 4.65, 4.65, 4.6, 4.6, 4.55, 4.55, 4.55, 4.55]

ax_components.plot(epochs, policy_loss, 'o-', color=COLORS['blue'], linewidth=2.5, label='Policy (main task)', markersize=6)
ax_components.plot(epochs, ownership_loss, 's-', color=COLORS['purple'], linewidth=2.5, label='Ownership (x1.5/b²)', markersize=6)
ax_components.plot(epochs, opponent_loss, '^-', color=COLORS['orange'], linewidth=2.5, label='Opponent (x0.15)', markersize=6)
ax_components.set_xlabel('Epoch', color=COLORS['text'], fontsize=12)
ax_components.set_ylabel('Loss Component', color=COLORS['text'], fontsize=12)
ax_components.set_title('Multi-Head Loss Breakdown (KataGo-style auxiliary tasks)',
                        color=COLORS['text'], fontsize=14, fontweight='bold', pad=10)
ax_components.legend(loc='upper right', facecolor=COLORS['card'], edgecolor=COLORS['text'],
                     labelcolor=COLORS['text'], fontsize=11)
ax_components.tick_params(colors=COLORS['text'])
ax_components.grid(True, alpha=0.2, color=COLORS['text'])
for spine in ax_components.spines.values():
    spine.set_color(COLORS['text'])
    spine.set_alpha(0.3)

# ===== BOTTOM RIGHT: Key Insights =====
ax_insights = fig.add_subplot(gs[2, 2])
ax_insights.set_facecolor(COLORS['card'])
ax_insights.axis('off')

insights_text = """KEY INSIGHTS
------------
[+] Curriculum works!
    Jump at epoch 4

[+] Multi-head helps
    361x more signal

[+] Not overfitting
    Can train longer

NEXT STEPS
----------
> BCE value loss
> MobileNetV2
> 500K+ positions
> 10+ blocks

TARGET: 35%+"""
ax_insights.text(0.05, 0.5, insights_text, fontsize=11, fontfamily='monospace',
                 color=COLORS['text'], ha='left', va='center',
                 transform=ax_insights.transAxes, linespacing=1.2)

# Footer
fig.text(0.5, 0.01,
         'GoGoGo Project | AlphaZero-style Go AI | Training on 93K pro games from CWI.nl | github.com/...',
         ha='center', fontsize=10, color=COLORS['text'], style='italic', alpha=0.7)

plt.savefig('training_plots/training_summary.png',
            facecolor=COLORS['bg'],
            edgecolor='none',
            dpi=150,
            bbox_inches='tight',
            pad_inches=0.3)
print('Saved to training_plots/training_summary.png')
