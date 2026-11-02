
import argparse
import os
import sys

import numpy as np
import torch

from stable_baselines3.common.vec_env import VecVideoRecorder

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
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/ppo_rgb.yaml")
    parser.add_argument(
        "--checkpoint",
        default="experiments/bc_baseline/bc_best.pt"
    )
    parser.add_argument("--out_dir", default="assets/videos")
    parser.add_argument("--video_length", type=int, default=1000)
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    os.makedirs(args.out_dir, exist_ok=True)

    env = make_carracing_vec_env(
        env_id=cfg["env_name"],
        n_envs=1,
        seed=int(cfg.get("seed", 42)) + 123,
        grayscale=False,
        resize=84,
        frame_stack=4,
        render_mode="rgb_array",
    )

    env = VecVideoRecorder(
        env,
        video_folder=args.out_dir,
        record_video_trigger=lambda step: step == 0,
        video_length=args.video_length,
        name_prefix="bc_rgb_100k",
    )

    model = BCPolicy().to(device)
    model.load_state_dict(
        torch.load(args.checkpoint, map_location=device)
    )
    model.eval()

    obs = env.reset()

    for step in range(args.video_length):
        obs_tensor = torch.as_tensor(
            obs,
            dtype=torch.float32,
            device=device,
        ) / 255.0

        with torch.no_grad():
            action = model(obs_tensor).cpu().numpy()

        action = np.clip(
            action,
            env.action_space.low,
            env.action_space.high,
        )

        obs, rewards, dones, infos = env.step(action)

    env.close()
    print("BC video recording completed.")


if __name__ == "__main__":
    main()
