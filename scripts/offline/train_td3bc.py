
import argparse
import copy
import os
import random

import numpy as np
import torch
import torch.nn.functional as F

from torch.utils.data import DataLoader

from td3bc_dataset import TD3BCDataset
from td3bc_model import (
    TwinCritic,
    clip_action,
    make_actor,
    soft_update,
)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data",
        default="data/td3bc/ppo_expert_100k.h5",
    )

    parser.add_argument(
        "--save_dir",
        default="experiments/td3bc",
    )
    parser.add_argument(
        "--updates",
        type=int,
        default=10000,
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
    )
    parser.add_argument(
        "--save_interval",
        type=int,
        default=1000,
    )
    parser.add_argument(
        "--resume",
        default=None,
    )

    args = parser.parse_args()

    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    gamma = 0.99
    tau = 0.005
    alpha = 2.5
    policy_noise = 0.2
    noise_clip = 0.5
    policy_freq = 2
    learning_rate = 3e-4

    os.makedirs(args.save_dir, exist_ok=True)

    dataset = TD3BCDataset(args.data)

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=device.type == "cuda",
        drop_last=True,
    )

    data_iterator = iter(loader)

    actor = make_actor().to(device)
    critic = TwinCritic().to(device)

    actor_target = copy.deepcopy(actor).to(device)
    critic_target = copy.deepcopy(critic).to(device)

    actor_optimizer = torch.optim.Adam(
        actor.parameters(),
        lr=learning_rate,
    )

    critic_optimizer = torch.optim.Adam(
        critic.parameters(),
        lr=learning_rate,
    )

    start_step = 0

    if args.resume is not None:
        checkpoint = torch.load(
            args.resume,
            map_location=device,
            weights_only=True,
        )

        actor.load_state_dict(checkpoint["actor"])
        critic.load_state_dict(checkpoint["critic"])

        actor_target.load_state_dict(
            checkpoint["actor_target"]
        )
        critic_target.load_state_dict(
            checkpoint["critic_target"]
        )

        actor_optimizer.load_state_dict(
            checkpoint["actor_optimizer"]
        )
        critic_optimizer.load_state_dict(
            checkpoint["critic_optimizer"]
        )

        start_step = int(checkpoint["step"])
        print("Resumed from step:", start_step)

    print("Device:", device)
    print("Transitions:", len(dataset))
    print("Target updates:", args.updates)

    actor.train()
    critic.train()

    last_actor_loss = float("nan")

    for step in range(
        start_step + 1,
        args.updates + 1,
    ):
        try:
            batch = next(data_iterator)
        except StopIteration:
            data_iterator = iter(loader)
            batch = next(data_iterator)

        state, action, reward, next_state, terminated = batch

        state = state.to(
            device,
            non_blocking=True,
        ).float() / 255.0

        next_state = next_state.to(
            device,
            non_blocking=True,
        ).float() / 255.0

        action = action.to(
            device,
            non_blocking=True,
        )

        reward = reward.to(
            device,
            non_blocking=True,
        ).unsqueeze(1)

        terminated = terminated.to(
            device,
            non_blocking=True,
        ).unsqueeze(1)

        # Compute the TD3 Bellman target
        with torch.no_grad():
            noise = (
                torch.randn_like(action) * policy_noise
            ).clamp(-noise_clip, noise_clip)

            next_action = actor_target(next_state)
            next_action = clip_action(
                next_action + noise
            )

            target_q1, target_q2 = critic_target(
                next_state,
                next_action,
            )

            target_q = torch.min(target_q1, target_q2)

            target_q = (
                reward
                + (1.0 - terminated) * gamma * target_q
            )

        # Update both critics
        current_q1, current_q2 = critic(
            state,
            action,
        )

        critic_loss = (
            F.mse_loss(current_q1, target_q)
            + F.mse_loss(current_q2, target_q)
        )

        critic_optimizer.zero_grad()
        critic_loss.backward()
        critic_optimizer.step()

        # Update actor less frequently than critics
        if step % policy_freq == 0:
            for parameter in critic.parameters():
                parameter.requires_grad_(False)

            predicted_action = actor(state)

            q1 = critic.q1(
                state,
                predicted_action,
            )

            bc_loss = F.mse_loss(
                predicted_action,
                action,
            )

            # Normalize Q contribution using dataset-action Q
            with torch.no_grad():
                dataset_q = critic.q1(
                    state,
                    action,
                )
                q_scale = dataset_q.abs().mean().clamp_min(
                    1e-6
                )
                lambda_value = alpha / q_scale

            actor_loss = (
                -lambda_value * q1.mean()
                + bc_loss
            )

            actor_optimizer.zero_grad()
            actor_loss.backward()
            actor_optimizer.step()

            last_actor_loss = actor_loss.item()

            for parameter in critic.parameters():
                parameter.requires_grad_(True)

            soft_update(
                actor,
                actor_target,
                tau,
            )
            soft_update(
                critic,
                critic_target,
                tau,
            )

        if step % 100 == 0:
            print(
                f"Step {step}/{args.updates} | "
                f"Critic loss: {critic_loss.item():.6f} | "
                f"Actor loss: {last_actor_loss:.6f}"
            )

        if step % args.save_interval == 0 or step == args.updates:
            checkpoint = {
                "step": step,
                "actor": actor.state_dict(),
                "critic": critic.state_dict(),
                "actor_target": actor_target.state_dict(),
                "critic_target": critic_target.state_dict(),
                "actor_optimizer": actor_optimizer.state_dict(),
                "critic_optimizer": critic_optimizer.state_dict(),
            }

            path = os.path.join(
                args.save_dir,
                f"checkpoint_{step}.pt",
            )

            torch.save(checkpoint, path)

            torch.save(
                actor.state_dict(),
                os.path.join(
                    args.save_dir,
                    "actor_latest.pt",
                ),
            )

            print("Checkpoint saved:", path)

    print("TD3+BC training completed.")


if __name__ == "__main__":
    main()
