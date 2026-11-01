import os
import sys

import h5py
import numpy as np
from stable_baselines3 import PPO

sys.path.insert(0, os.path.abspath("src"))

from carracing_ppo.envs.carracing import make_carracing_vec_env
from carracing_ppo.utils.io import load_config


CONFIG = "configs/ppo_rgb.yaml"

MODEL_PATH = (
    "experiments/carracing-ppo_20261029_062035/"
    "models/best_model.zip"
)

OUTPUT = "data/expert/ppo_expert.h5"
TOTAL_STEPS = 100000


def main():
    cfg = load_config(CONFIG)

    env = make_carracing_vec_env(
        env_id=cfg["env_name"],
        n_envs=1,
        seed=int(cfg.get("seed", 42)),
        grayscale=False,
        resize=84,
        frame_stack=4,
    )

    model = PPO.load(
        MODEL_PATH,
        env=env,
        device="auto",
    )

    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)

    obs = env.reset()

    episode_id = 0
    total_steps = 0

    with h5py.File(OUTPUT, "w") as dataset:
        dataset.attrs["source"] = "PPO Best"
        dataset.attrs["grayscale"] = False
        dataset.attrs["frame_stack"] = 4

        while total_steps < TOTAL_STEPS:
            frames = []
            actions = []
            rewards = []

            episode_reward = 0.0
            done = False

            while not done and total_steps < TOTAL_STEPS:
                action, _ = model.predict(
                    obs,
                    deterministic=True,
                )

                # Extract the latest RGB frame from the stacked observation
                current_frame = (
                    obs[0, -3:, :, :]
                    .transpose(1, 2, 0)
                    .copy()
                )

                next_obs, reward, dones, infos = env.step(action)

                frames.append(current_frame)
                actions.append(action[0].copy())
                rewards.append(float(reward[0]))

                episode_reward += float(reward[0])
                total_steps += 1

                obs = next_obs
                done = bool(dones[0])

            group = dataset.create_group(
                f"episode_{episode_id:04d}"
            )

            group.create_dataset(
                "images",
                data=np.asarray(frames, dtype=np.uint8),
                compression="gzip",
            )

            group.create_dataset(
                "actions",
                data=np.asarray(actions, dtype=np.float32),
            )

            group.create_dataset(
                "rewards",
                data=np.asarray(rewards, dtype=np.float32),
            )

            group.attrs["episode_finished"] = done
            group.attrs["episode_reward"] = episode_reward

            print(
                f"Episode {episode_id}: "
                f"{len(frames)} steps, "
                f"reward={episode_reward:.1f}, "
                f"finished={done}"
            )

            episode_id += 1

    env.close()

    print(f"Collection completed: {total_steps} steps")
    print(f"Dataset saved to: {OUTPUT}")


if __name__ == "__main__":
    main()
