import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "src")
    )
)

from carracing_ppo.envs.carracing import make_carracing_vec_env
from carracing_ppo.utils.io import load_config

from train_bc import BCPolicy


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained BC model."
    )

    parser.add_argument(
        "--config",
        default="configs/ppo_rgb.yaml"
    )

    parser.add_argument(
        "--checkpoint",
        default="experiments/bc_baseline/bc_best.pt"
    )

    parser.add_argument(
        "--episodes",
        type=int,
        default=20
    )

    args = parser.parse_args()

    cfg = load_config(args.config)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    env = make_carracing_vec_env(
        env_id=cfg["env_name"],
        n_envs=1,
        seed=int(cfg.get("seed", 42)) + 123,
        grayscale=False,
        resize=84,
        frame_stack=4,
        render_mode=None,
    )

    model = BCPolicy().to(device)

    model.load_state_dict(
        torch.load(
            args.checkpoint,
            map_location=device
        )
    )

    model.eval()

    obs = env.reset()

    episode_rewards = []
    episode_reward = 0.0

    print("Evaluation device:", device)

    while len(episode_rewards) < args.episodes:

        obs_tensor = torch.as_tensor(
            obs,
            dtype=torch.float32,
            device=device
        ) / 255.0

        with torch.no_grad():
            action = model(obs_tensor).cpu().numpy()

        action = np.clip(
            action,
            env.action_space.low,
            env.action_space.high
        )

        obs, rewards, dones, infos = env.step(action)

        episode_reward += float(rewards[0])

        if dones[0]:
            episode_rewards.append(episode_reward)

            print(
                f"Episode {len(episode_rewards)}: "
                f"Reward = {episode_reward:.2f}"
            )

            episode_reward = 0.0

    env.close()

    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)

    print("\nEvaluation Results")
    print("------------------")
    print(f"Mean reward: {mean_reward:.3f}")
    print(f"Std reward: {std_reward:.3f}")


if __name__ == "__main__":
    main()
