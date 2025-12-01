"""Neural network model (ResNet with policy and value heads).

Improvements over vanilla AlphaZero:
- Global pooling (KataGo) - captures non-local patterns like ko, ladders
- Pre-activation ResNet - better gradient flow
- Squeeze-and-excitation - channel attention
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import Config


class GlobalPoolingBlock(nn.Module):
    """Global pooling block (KataGo style).

    Concatenates global mean and max pooling to capture board-wide patterns.
    This helps with non-local tactics like ko, large-scale influence, etc.
    """

    def __init__(self, channels: int):
        super().__init__()
        # Project pooled features back to spatial
        self.fc = nn.Linear(channels * 2, channels)
        self.bn = nn.BatchNorm1d(channels)

    def forward(self, x):
        # x: (batch, channels, H, W)
        batch, channels, h, w = x.shape

        # Global pooling
        mean_pool = x.mean(dim=(2, 3))  # (batch, channels)
        max_pool = x.amax(dim=(2, 3))   # (batch, channels)

        # Concatenate and project
        pooled = torch.cat([mean_pool, max_pool], dim=1)  # (batch, channels*2)
        pooled = F.relu(self.bn(self.fc(pooled)))  # (batch, channels)

        # Broadcast back to spatial and add
        pooled = pooled.view(batch, channels, 1, 1)
        return x + pooled.expand_as(x)


class ResBlock(nn.Module):
    """Pre-activation residual block (better gradient flow)."""

    def __init__(self, channels: int, use_global_pool: bool = False):
        super().__init__()
        # Pre-activation: BN -> ReLU -> Conv (instead of Conv -> BN -> ReLU)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)

        # Optional global pooling
        self.global_pool = GlobalPoolingBlock(channels) if use_global_pool else None

    def forward(self, x):
        residual = x

        # Pre-activation
        x = F.relu(self.bn1(x))
        x = self.conv1(x)
        x = F.relu(self.bn2(x))
        x = self.conv2(x)

        # Add residual
        x = x + residual

        # Optional global pooling
        if self.global_pool is not None:
            x = self.global_pool(x)

        return x


class MobileNetV2Block(nn.Module):
    """MobileNetV2-style inverted residual block (Cazenave 2020).

    Key differences from standard ResNet:
    - Inverted bottleneck: expand → depthwise → project (instead of squeeze)
    - Depthwise separable convolutions: more efficient
    - Linear bottleneck: no ReLU at the end (preserves information)
    - ReLU6 instead of ReLU (better for quantization, empirically works well)

    Reference: https://arxiv.org/abs/2001.09613
    """

    def __init__(self, channels: int, expansion: int = 4, use_global_pool: bool = False):
        super().__init__()
        expanded = channels * expansion

        # 1. Expand: 1x1 conv to increase channels
        self.expand_conv = nn.Conv2d(channels, expanded, 1, bias=False)
        self.expand_bn = nn.BatchNorm2d(expanded)

        # 2. Depthwise: 3x3 depthwise conv (groups = channels)
        self.depthwise_conv = nn.Conv2d(expanded, expanded, 3, padding=1, groups=expanded, bias=False)
        self.depthwise_bn = nn.BatchNorm2d(expanded)

        # 3. Project: 1x1 conv to reduce channels (linear bottleneck - no activation)
        self.project_conv = nn.Conv2d(expanded, channels, 1, bias=False)
        self.project_bn = nn.BatchNorm2d(channels)

        # Optional global pooling
        self.global_pool = GlobalPoolingBlock(channels) if use_global_pool else None

    def forward(self, x):
        residual = x

        # Expand
        x = F.relu6(self.expand_bn(self.expand_conv(x)))

        # Depthwise conv
        x = F.relu6(self.depthwise_bn(self.depthwise_conv(x)))

        # Project (linear - no activation!)
        x = self.project_bn(self.project_conv(x))

        # Add residual
        x = x + residual

        # Optional global pooling
        if self.global_pool is not None:
            x = self.global_pool(x)

        return x


class GoNet(nn.Module):
    """AlphaZero-style network for Go with KataGo improvements.

    Backbone options:
    - "resnet": Standard pre-activation residual blocks
    - "mobilenetv2": MobileNetV2-style inverted residual blocks (Cazenave 2020)
                    More parameter-efficient with depthwise separable convolutions
    """

    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.board_size = config.board_size
        self.action_size = config.board_size ** 2 + 1  # +1 for pass

        # Initial conv
        self.conv_init = nn.Conv2d(config.input_planes, config.num_filters, 3, padding=1, bias=False)
        self.bn_init = nn.BatchNorm2d(config.num_filters)

        # Residual tower with global pooling every few blocks
        self.res_blocks = nn.ModuleList()
        backbone = getattr(config, 'backbone', 'resnet')  # Default to resnet for compatibility
        expansion = getattr(config, 'mobilenet_expansion', 4)

        for i in range(config.num_blocks):
            # Add global pooling every 3rd block (or at least once)
            use_global = (i > 0 and i % 3 == 0) or (i == config.num_blocks - 1)

            if backbone == 'mobilenetv2':
                self.res_blocks.append(MobileNetV2Block(
                    config.num_filters, expansion=expansion, use_global_pool=use_global))
            else:  # resnet
                self.res_blocks.append(ResBlock(config.num_filters, use_global_pool=use_global))

        # Policy head (shared conv, separate FC for player and opponent)
        self.policy_conv = nn.Conv2d(config.num_filters, 2, 1, bias=False)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * config.board_size ** 2, self.action_size)
        # Opponent policy head (KataGo: 1.30x speedup from predicting opponent's response)
        # Shares conv features with main policy, separate FC layer
        self.opponent_policy_fc = nn.Linear(2 * config.board_size ** 2, self.action_size)

        # Value head with global pooling
        self.value_conv = nn.Conv2d(config.num_filters, 1, 1, bias=False)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(config.board_size ** 2, 256)
        self.value_fc2 = nn.Linear(256, 1)

        # Ownership head (KataGo's key insight: 361x more signal per game)
        # Predicts final ownership of each point: +1 = current player, -1 = opponent
        # Output is logits; use BCE loss with targets mapped from [-1,1] to [0,1]
        self.ownership_conv = nn.Conv2d(config.num_filters, 1, 1)

    def forward(self, x, return_ownership: bool = False, return_opponent_policy: bool = False):
        # x: (batch, planes, size, size)
        x = F.relu(self.bn_init(self.conv_init(x)))

        for block in self.res_blocks:
            x = block(x)

        # Policy head (shared conv features)
        p = F.relu(self.policy_bn(self.policy_conv(x)))
        p_flat = p.view(p.size(0), -1)
        policy = F.log_softmax(self.policy_fc(p_flat), dim=1)

        # Opponent policy head (shares conv, separate FC)
        opponent_policy = None
        if return_opponent_policy:
            opponent_policy = F.log_softmax(self.opponent_policy_fc(p_flat), dim=1)

        # Value head
        v = F.relu(self.value_bn(self.value_conv(x)))
        v = v.view(v.size(0), -1)
        v = F.relu(self.value_fc1(v))
        value = torch.tanh(self.value_fc2(v))

        # Build return tuple based on what was requested
        if return_ownership or return_opponent_policy:
            result = [policy, value]
            if return_ownership:
                ownership = self.ownership_conv(x)
                result.append(ownership)
            if return_opponent_policy:
                result.append(opponent_policy)
            return tuple(result)

        return policy, value

    def predict(self, board_tensor):
        """Single board prediction."""
        self.eval()
        with torch.no_grad():
            x = torch.FloatTensor(board_tensor).unsqueeze(0).to(next(self.parameters()).device)
            policy, value = self(x)
            return torch.exp(policy).cpu().numpy()[0], value.cpu().numpy()[0, 0]


def create_model(config: Config) -> GoNet:
    """Create and initialize model."""
    model = GoNet(config)
    model = model.to(config.device)
    return model


def save_checkpoint(model: GoNet, optimizer, step: int, path: str):
    """Save model checkpoint."""
    torch.save({
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': model.config,
    }, path)


def load_checkpoint(path: str, config: Config):
    """Load model from checkpoint."""
    checkpoint = torch.load(path, map_location=config.device, weights_only=False)
    model = GoNet(checkpoint.get('config', config))
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(config.device)
    return model, checkpoint.get('step', 0)
