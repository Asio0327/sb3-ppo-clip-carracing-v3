
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
from td3bc_model import make_actor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/ppo_rgb.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--episodes", type=int, default=20)
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
    )

    actor = make_actor().to(device)

    checkpoint = torch.load(
        args.checkpoint,
        map_location=device,
        weights_only=True,
    )

    actor.load_state_dict(checkpoint["actor"])
    actor.eval()

    obs = env.reset()

    rewards_per_episode = []
    episode_reward = 0.0

    while len(rewards_per_episode) < args.episodes:
        state = torch.as_tensor(
            obs,
            dtype=torch.float32,
            device=device,
        ) / 255.0

        with torch.no_grad():
            action = actor(state).cpu().numpy()

        action = np.clip(
            action,
            env.action_space.low,
            env.action_space.high,
        )

        obs, rewards, dones, infos = env.step(action)
        episode_reward += float(rewards[0])

        if dones[0]:
            rewards_per_episode.append(episode_reward)

            print(
                f"Episode {len(rewards_per_episode)}: "
                f"Reward = {episode_reward:.2f}"
            )

            episode_reward = 0.0

    env.close()

    print("\nEvaluation Results")
    print("Mean reward:", np.mean(rewards_per_episode))
    print("Std reward:", np.std(rewards_per_episode))


if __name__ == "__main__":
    main()
