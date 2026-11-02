
import argparse
import os
import sys

import h5py
import numpy as np

from stable_baselines3 import PPO

sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "src")
    )
)

from carracing_ppo.envs.carracing import make_carracing_vec_env
from carracing_ppo.utils.io import load_config


def to_chw(observation):
    observation = np.asarray(observation)

    if observation.shape == (12, 84, 84):
        return observation

    if observation.shape == (84, 84, 12):
        return observation.transpose(2, 0, 1)

    raise ValueError(
        f"Unexpected observation shape: {observation.shape}"
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        default="configs/ppo_rgb.yaml"
    )

    parser.add_argument(
        "--checkpoint",
        default=(
            "experiments/carracing-ppo_20261029_062035/"
            "models/best_model.zip"
        )
    )

    parser.add_argument(
        "--output",
        default="data/td3bc/ppo_expert_100k.h5"
    )

    parser.add_argument(
        "--total_steps",
        type=int,
        default=100000
    )

    args = parser.parse_args()

    if os.path.exists(args.output):
        raise FileExistsError(
            f"Output already exists: {args.output}"
        )

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    cfg = load_config(args.config)

    env = make_carracing_vec_env(
        env_id=cfg["env_name"],
        n_envs=1,
        seed=int(cfg.get("seed", 42)),
        grayscale=False,
        resize=84,
        frame_stack=4,
        render_mode=None,
    )

    model = PPO.load(
        args.checkpoint,
        device="auto",
    )

    obs = env.reset()

    total_steps = 0
    episode_id = 0

    print("Starting TD3+BC dataset collection.")
    print("Target steps:", args.total_steps)

    try:
        with h5py.File(args.output, "w") as dataset:

            dataset.attrs["source"] = "PPO expert"
            dataset.attrs["grayscale"] = False
            dataset.attrs["frame_stack"] = 4
            dataset.attrs["image_size"] = 84
            dataset.attrs["target_steps"] = args.total_steps

            while total_steps < args.total_steps:

                initial_observation = to_chw(obs[0]).copy()

                frames = []
                actions = []
                rewards = []
                terminated_flags = []
                truncated_flags = []

                episode_reward = 0.0
                done = False
                terminal_observation = None

                while not done:

                    current_obs = to_chw(obs[0])

                    current_frame = (
                        current_obs[-3:, :, :]
                        .transpose(1, 2, 0)
                        .copy()
                    )

                    action, _ = model.predict(
                        obs,
                        deterministic=True,
                    )

                    next_obs, reward, dones, infos = env.step(action)

                    done = bool(dones[0])
                    info = infos[0]

                    frames.append(current_frame)
                    actions.append(
                        np.asarray(action[0], dtype=np.float32)
                    )
                    rewards.append(float(reward[0]))

                    episode_reward += float(reward[0])

                    truncated = bool(
                        info.get("TimeLimit.truncated", False)
                    )

                    terminated_flags.append(
                        done and not truncated
                    )

                    truncated_flags.append(
                        done and truncated
                    )

                    if done:
                        terminal = info.get("terminal_observation")

                        if terminal is None:
                            raise RuntimeError(
                                "Missing terminal_observation."
                            )

                        terminal_observation = to_chw(terminal).copy()

                    obs = next_obs
                    total_steps += 1

                group = dataset.create_group(
                    f"episode_{episode_id:05d}"
                )

                group.create_dataset(
                    "initial_observation",
                    data=initial_observation,
                )

                group.create_dataset(
                    "frames",
                    data=np.asarray(frames, dtype=np.uint8),
                    compression="lzf",
                    shuffle=True,
                )

                group.create_dataset(
                    "actions",
                    data=np.asarray(actions, dtype=np.float32),
                )

                group.create_dataset(
                    "rewards",
                    data=np.asarray(rewards, dtype=np.float32),
                )

                group.create_dataset(
                    "terminated",
                    data=np.asarray(terminated_flags, dtype=np.bool_),
                )

                group.create_dataset(
                    "truncated",
                    data=np.asarray(truncated_flags, dtype=np.bool_),
                )

                group.create_dataset(
                    "terminal_observation",
                    data=terminal_observation,
                    compression="lzf",
                )

                group.attrs["episode_reward"] = episode_reward
                group.attrs["episode_steps"] = len(actions)

                dataset.attrs["total_steps"] = total_steps
                dataset.attrs["episodes"] = episode_id + 1

                dataset.flush()

                print(
                    f"Episode {episode_id}: "
                    f"steps={len(actions)}, "
                    f"reward={episode_reward:.2f}, "
                    f"total={total_steps}"
                )

                episode_id += 1

    finally:
        env.close()

    print("Dataset collection completed.")
    print("Total transitions:", total_steps)
    print("Episodes:", episode_id)
    print("Output:", args.output)


if __name__ == "__main__":
    main()
