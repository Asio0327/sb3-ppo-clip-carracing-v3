
import torch
import torch.nn as nn

from train_bc import BCPolicy


class CriticNetwork(nn.Module):
    def __init__(self, action_dim=3):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(12, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 256),
            nn.ReLU(),
        )

        self.q_head = nn.Sequential(
            nn.Linear(256 + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, state, action):
        features = self.encoder(state)
        x = torch.cat([features, action], dim=1)
        return self.q_head(x)


class TwinCritic(nn.Module):
    def __init__(self, action_dim=3):
        super().__init__()

        self.q1 = CriticNetwork(action_dim)
        self.q2 = CriticNetwork(action_dim)

    def forward(self, state, action):
        q1 = self.q1(state, action)
        q2 = self.q2(state, action)
        return q1, q2


def make_actor():
    return BCPolicy()


def clip_action(action):
    steering = action[:, 0:1].clamp(-1.0, 1.0)
    gas = action[:, 1:2].clamp(0.0, 1.0)
    brake = action[:, 2:3].clamp(0.0, 1.0)

    return torch.cat(
        [steering, gas, brake],
        dim=1,
    )


def soft_update(source, target, tau):
    with torch.no_grad():
        for source_param, target_param in zip(
            source.parameters(),
            target.parameters(),
        ):
            target_param.data.mul_(1.0 - tau)
            target_param.data.add_(
                tau * source_param.data
            )
