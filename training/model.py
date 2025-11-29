"""Neural network model (ResNet with policy and value heads)."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import Config

class ResBlock(nn.Module):
    """Residual block with two conv layers."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x = F.relu(x + residual)
        return x


class GoNet(nn.Module):
    """AlphaZero-style network for Go."""

    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.board_size = config.board_size
        self.action_size = config.board_size ** 2 + 1  # +1 for pass

        # Initial conv
        self.conv_init = nn.Conv2d(config.input_planes, config.num_filters, 3, padding=1, bias=False)
        self.bn_init = nn.BatchNorm2d(config.num_filters)

        # Residual tower
        self.res_blocks = nn.ModuleList([
            ResBlock(config.num_filters) for _ in range(config.num_blocks)
        ])

        # Policy head
        self.policy_conv = nn.Conv2d(config.num_filters, 2, 1, bias=False)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * config.board_size ** 2, self.action_size)

        # Value head
        self.value_conv = nn.Conv2d(config.num_filters, 1, 1, bias=False)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(config.board_size ** 2, 256)
        self.value_fc2 = nn.Linear(256, 1)

    def forward(self, x):
        # x: (batch, planes, size, size)
        x = F.relu(self.bn_init(self.conv_init(x)))

        for block in self.res_blocks:
            x = block(x)

        # Policy head
        p = F.relu(self.policy_bn(self.policy_conv(x)))
        p = p.view(p.size(0), -1)
        p = self.policy_fc(p)
        policy = F.log_softmax(p, dim=1)

        # Value head
        v = F.relu(self.value_bn(self.value_conv(x)))
        v = v.view(v.size(0), -1)
        v = F.relu(self.value_fc1(v))
        value = torch.tanh(self.value_fc2(v))

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
    checkpoint = torch.load(path, map_location=config.device)
    model = GoNet(checkpoint.get('config', config))
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(config.device)
    return model, checkpoint.get('step', 0)
