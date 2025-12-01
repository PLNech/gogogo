"""Training visualization utilities."""
import os
import json
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend (no tkinter needed)
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from datetime import datetime


class TrainingVisualizer:
    """Tracks and visualizes training metrics."""

    def __init__(self, output_dir: str = 'training_plots'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Metrics storage
        self.train_losses = []
        self.val_accuracies = []
        self.learning_rates = []
        self.batch_losses = []  # Per-batch for detailed view
        self.epoch_times = []

        # Config info
        self.config_info = {}
        self.start_time = datetime.now()

    def set_config(self, config, extra_info: dict = None):
        """Store config for visualization labels."""
        self.config_info = {
            'board_size': config.board_size,
            'num_blocks': config.num_blocks,
            'num_filters': config.num_filters,
            'input_planes': config.input_planes,
            'batch_size': config.batch_size,
            'learning_rate': config.learning_rate,
        }
        if extra_info:
            self.config_info.update(extra_info)

    def log_batch(self, loss: float, lr: float):
        """Log per-batch metrics."""
        self.batch_losses.append(loss)
        self.learning_rates.append(lr)

    def log_epoch(self, epoch: int, train_loss: float, val_accuracy: float, epoch_time: float = None):
        """Log end-of-epoch metrics."""
        self.train_losses.append(train_loss)
        self.val_accuracies.append(val_accuracy)
        if epoch_time:
            self.epoch_times.append(epoch_time)

    def plot_training_curves(self, save: bool = True, show: bool = False):
        """Generate training loss and accuracy curves."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(self._get_title(), fontsize=14, fontweight='bold')

        # 1. Training Loss (per epoch)
        ax1 = axes[0, 0]
        if self.train_losses:
            epochs = range(1, len(self.train_losses) + 1)
            ax1.plot(epochs, self.train_losses, 'b-o', linewidth=2, markersize=6)
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss')
            ax1.set_title('Training Loss')
            ax1.grid(True, alpha=0.3)
            ax1.set_xticks(list(epochs))

        # 2. Validation Accuracy
        ax2 = axes[0, 1]
        if self.val_accuracies:
            epochs = range(1, len(self.val_accuracies) + 1)
            ax2.plot(epochs, [a * 100 for a in self.val_accuracies], 'g-o', linewidth=2, markersize=6)
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Accuracy (%)')
            ax2.set_title('Validation Accuracy')
            ax2.grid(True, alpha=0.3)
            ax2.set_xticks(list(epochs))
            # Add best accuracy annotation
            best_acc = max(self.val_accuracies) * 100
            best_epoch = self.val_accuracies.index(max(self.val_accuracies)) + 1
            ax2.annotate(f'Best: {best_acc:.1f}%', xy=(best_epoch, best_acc),
                        xytext=(best_epoch + 0.5, best_acc - 5),
                        arrowprops=dict(arrowstyle='->', color='green'),
                        fontsize=10, color='green')

        # 3. Batch Loss (smoothed)
        ax3 = axes[1, 0]
        if self.batch_losses:
            # Smooth with moving average
            window = min(50, len(self.batch_losses) // 10 + 1)
            smoothed = np.convolve(self.batch_losses, np.ones(window)/window, mode='valid')
            ax3.plot(smoothed, 'b-', alpha=0.8, linewidth=1)
            ax3.plot(self.batch_losses, 'b-', alpha=0.2, linewidth=0.5)
            ax3.set_xlabel('Batch')
            ax3.set_ylabel('Loss')
            ax3.set_title(f'Batch Loss (smoothed, window={window})')
            ax3.grid(True, alpha=0.3)

        # 4. Learning Rate Schedule
        ax4 = axes[1, 1]
        if self.learning_rates:
            ax4.plot(self.learning_rates, 'r-', linewidth=1)
            ax4.set_xlabel('Batch')
            ax4.set_ylabel('Learning Rate')
            ax4.set_title('Learning Rate Schedule')
            ax4.grid(True, alpha=0.3)
            ax4.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))

        plt.tight_layout()

        if save:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            path = self.output_dir / f'training_curves_{timestamp}.png'
            plt.savefig(path, dpi=150, bbox_inches='tight')
            print(f"Saved training curves to {path}")

        if show:
            plt.show()
        else:
            plt.close()

        return fig

    def plot_loss_distribution(self, save: bool = True):
        """Plot loss distribution across batches."""
        if not self.batch_losses:
            return

        fig, ax = plt.subplots(figsize=(10, 6))

        # Histogram of losses
        ax.hist(self.batch_losses, bins=50, alpha=0.7, color='blue', edgecolor='black')
        ax.axvline(np.mean(self.batch_losses), color='red', linestyle='--',
                   label=f'Mean: {np.mean(self.batch_losses):.3f}')
        ax.axvline(np.median(self.batch_losses), color='green', linestyle='--',
                   label=f'Median: {np.median(self.batch_losses):.3f}')

        ax.set_xlabel('Loss')
        ax.set_ylabel('Frequency')
        ax.set_title('Loss Distribution Across Batches')
        ax.legend()
        ax.grid(True, alpha=0.3)

        if save:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            path = self.output_dir / f'loss_distribution_{timestamp}.png'
            plt.savefig(path, dpi=150, bbox_inches='tight')
            print(f"Saved loss distribution to {path}")

        plt.close()

    def plot_convergence_analysis(self, save: bool = True):
        """Analyze convergence rate."""
        if len(self.val_accuracies) < 2:
            return

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # 1. Accuracy improvement per epoch
        ax1 = axes[0]
        improvements = [self.val_accuracies[i] - self.val_accuracies[i-1]
                       for i in range(1, len(self.val_accuracies))]
        epochs = range(2, len(self.val_accuracies) + 1)
        colors = ['green' if x > 0 else 'red' for x in improvements]
        ax1.bar(epochs, [x * 100 for x in improvements], color=colors, alpha=0.7)
        ax1.axhline(0, color='black', linewidth=0.5)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy Change (%)')
        ax1.set_title('Per-Epoch Improvement')
        ax1.grid(True, alpha=0.3)

        # 2. Loss vs Accuracy scatter
        ax2 = axes[1]
        if len(self.train_losses) == len(self.val_accuracies):
            ax2.scatter(self.train_losses, [a * 100 for a in self.val_accuracies],
                       c=range(len(self.train_losses)), cmap='viridis', s=100)
            ax2.set_xlabel('Training Loss')
            ax2.set_ylabel('Validation Accuracy (%)')
            ax2.set_title('Loss vs Accuracy (color = epoch)')
            ax2.grid(True, alpha=0.3)
            # Add colorbar
            sm = plt.cm.ScalarMappable(cmap='viridis',
                                        norm=plt.Normalize(1, len(self.train_losses)))
            plt.colorbar(sm, ax=ax2, label='Epoch')

        plt.tight_layout()

        if save:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            path = self.output_dir / f'convergence_{timestamp}.png'
            plt.savefig(path, dpi=150, bbox_inches='tight')
            print(f"Saved convergence analysis to {path}")

        plt.close()

    def save_metrics(self):
        """Save raw metrics to JSON for later analysis."""
        metrics = {
            'config': self.config_info,
            'train_losses': self.train_losses,
            'val_accuracies': self.val_accuracies,
            'learning_rates': self.learning_rates[-1] if self.learning_rates else None,
            'total_batches': len(self.batch_losses),
            'best_accuracy': max(self.val_accuracies) if self.val_accuracies else 0,
            'final_loss': self.train_losses[-1] if self.train_losses else None,
            'training_duration': str(datetime.now() - self.start_time),
        }

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        path = self.output_dir / f'metrics_{timestamp}.json'
        with open(path, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"Saved metrics to {path}")

    def generate_all_plots(self):
        """Generate all visualization plots."""
        self.plot_training_curves(save=True)
        self.plot_loss_distribution(save=True)
        self.plot_convergence_analysis(save=True)
        self.save_metrics()

    def _get_title(self) -> str:
        """Generate title from config."""
        c = self.config_info
        if c:
            return (f"GoNet Training: {c.get('num_blocks', '?')} blocks × "
                   f"{c.get('num_filters', '?')} filters, "
                   f"{c.get('input_planes', '?')} input planes, "
                   f"batch={c.get('batch_size', '?')}")
        return "GoNet Training"


def plot_network_architecture(config, save_path: str = None):
    """Generate a visual representation of the network architecture."""
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis('off')

    # Colors
    colors = {
        'input': '#E3F2FD',
        'conv': '#BBDEFB',
        'resblock': '#90CAF9',
        'policy': '#C8E6C9',
        'value': '#FFCDD2',
        'output': '#FFF9C4'
    }

    def draw_box(x, y, w, h, text, color, fontsize=9):
        rect = plt.Rectangle((x, y), w, h, facecolor=color, edgecolor='black', linewidth=1.5)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=fontsize, wrap=True)

    def draw_arrow(x1, y1, x2, y2):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                   arrowprops=dict(arrowstyle='->', color='black', lw=1.5))

    # Input
    draw_box(5, 75, 15, 15, f'Input\n{config.input_planes}×{config.board_size}×{config.board_size}',
             colors['input'], 10)

    # Initial Conv
    draw_box(25, 75, 15, 15, f'Conv 3×3\n→{config.num_filters}\n+BN+ReLU', colors['conv'], 9)
    draw_arrow(20, 82.5, 25, 82.5)

    # Residual Tower
    tower_y = 75
    for i in range(min(config.num_blocks, 4)):  # Show max 4 blocks
        draw_box(45, tower_y - i*12, 20, 10, f'ResBlock {i+1}\n{config.num_filters}ch',
                colors['resblock'], 9)
        if i > 0:
            draw_arrow(55, tower_y - (i-1)*12 - 1, 55, tower_y - i*12 + 10)

    if config.num_blocks > 4:
        ax.text(55, tower_y - 4*12 + 5, f'... +{config.num_blocks - 4} more',
               ha='center', fontsize=8, style='italic')

    draw_arrow(40, 82.5, 45, 82.5)

    # Split to heads
    split_y = tower_y - (min(config.num_blocks, 4) - 1) * 12 - 5

    # Policy Head
    draw_box(72, 70, 12, 12, 'Policy\nConv 1×1\n→2ch', colors['policy'], 8)
    draw_box(72, 55, 12, 12, 'FC\n722→362', colors['policy'], 8)
    draw_box(72, 40, 12, 10, 'LogSoftmax', colors['policy'], 8)
    draw_box(72, 25, 12, 12, f'π\n{config.board_size**2+1}\nmoves', colors['output'], 9)

    draw_arrow(65, split_y, 72, 76)
    draw_arrow(78, 70, 78, 67)
    draw_arrow(78, 55, 78, 52)
    draw_arrow(78, 40, 78, 37)

    # Value Head
    draw_box(88, 70, 10, 12, 'Value\nConv 1×1\n→1ch', colors['value'], 8)
    draw_box(88, 55, 10, 12, 'FC\n361→256', colors['value'], 8)
    draw_box(88, 40, 10, 10, 'FC\n256→1', colors['value'], 8)
    draw_box(88, 25, 10, 12, 'v\n[-1,1]', colors['output'], 9)

    draw_arrow(65, split_y, 88, 76)
    draw_arrow(93, 70, 93, 67)
    draw_arrow(93, 55, 93, 52)
    draw_arrow(93, 40, 93, 37)

    # Title
    params = sum(p.numel() for p in []) if False else "~2.1M"  # Placeholder
    ax.text(50, 95, f'GoNet Architecture ({config.num_blocks} blocks, {config.num_filters} filters, {params} params)',
           ha='center', fontsize=14, fontweight='bold')

    # Legend for input planes
    if config.input_planes == 27:
        legend_text = "Input: 17 basic (stones+history) + 10 tactical (liberties, captures, eyes)"
    else:
        legend_text = "Input: 17 basic planes (stones + 8 history)"
    ax.text(50, 5, legend_text, ha='center', fontsize=10, style='italic')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved architecture diagram to {save_path}")

    plt.close()
    return fig


def plot_feature_planes(board_tensor: np.ndarray, save_path: str = None):
    """Visualize input feature planes."""
    n_planes = board_tensor.shape[0]

    # Determine grid size
    cols = 6
    rows = (n_planes + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.5, rows * 2.5))
    axes = axes.flatten()

    plane_names = [
        'Current stones', 'Opponent stones',
        'History 1', 'History 2', 'History 3', 'History 4',
        'History 5', 'History 6', 'History 7', 'History 8',
        'Opp Hist 1', 'Opp Hist 2', 'Opp Hist 3', 'Opp Hist 4',
        'Opp Hist 5', 'Opp Hist 6', 'Opp Hist 7',
        'Own 1-lib', 'Own 2-lib', 'Own 3+-lib',
        'Opp 1-lib', 'Opp 2-lib', 'Opp 3+-lib',
        'Capture', 'Self-atari', 'Eye-like', 'Edge dist'
    ]

    for i in range(len(axes)):
        ax = axes[i]
        if i < n_planes:
            im = ax.imshow(board_tensor[i], cmap='Blues', vmin=0, vmax=1)
            name = plane_names[i] if i < len(plane_names) else f'Plane {i}'
            ax.set_title(name, fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])
        else:
            ax.axis('off')

    plt.suptitle(f'Input Feature Planes ({n_planes} total)', fontsize=12, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved feature planes to {save_path}")

    plt.close()
    return fig


if __name__ == '__main__':
    # Demo
    from config import DEFAULT

    # Generate architecture diagram
    plot_network_architecture(DEFAULT, 'training_plots/architecture.png')

    # Demo training curves
    viz = TrainingVisualizer()
    viz.set_config(DEFAULT)

    # Simulate some data
    for epoch in range(10):
        for batch in range(100):
            viz.log_batch(5.0 - epoch * 0.3 - batch * 0.01 + np.random.randn() * 0.2,
                         0.001 * (1 - (epoch * 100 + batch) / 1000))
        viz.log_epoch(epoch + 1, 5.0 - epoch * 0.3, 0.1 + epoch * 0.03)

    viz.generate_all_plots()
    print("Demo visualizations generated!")
