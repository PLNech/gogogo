# Training Guide

## Quick Start (19x19 Pro Games)

### 1. Test Run (5 minutes)
```bash
poetry run python train_supervised.py --board-size 19 --max-games 100 --epochs 3 --quick
```

### 2. Medium Training (30-60 minutes)
```bash
poetry run python train_supervised.py --board-size 19 --max-games 5000 --epochs 10
```

### 3. Full Training (3-5 hours)
```bash
poetry run python train_supervised.py --board-size 19 --max-games 50000 --epochs 20
```

### 4. Use Pre-trained Model for Self-Play
```bash
# Use same board size as supervised training!
poetry run python train.py --board-size 19 --resume checkpoints/supervised_best.pt --iterations 50
```

---

## Tips

- **First run**: Slow (loads SGF). **Second run**: Fast (uses cache).
- **Accuracy target**: 5-10% after 3 epochs is normal, 20-40% after 20 epochs on full dataset.
- **GPU**: RTX 4080 handles 19x19 fine.
- **Cache**: Stored in `dataset_cache/`, safe to delete to rebuild.

---

## If Training Crashes

Resume from checkpoint:
```bash
poetry run python train.py --board-size 19 --resume checkpoints/supervised_epoch_5.pt --iterations 100
```
