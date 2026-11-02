
import bisect

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class TD3BCDataset(Dataset):
    def __init__(self, path):
        self.episodes = []
        self.lengths = []

        with h5py.File(path, "r") as f:
            for name in sorted(f.keys()):
                group = f[name]

                episode = {
                    "frames": group["frames"][:],
                    "actions": group["actions"][:],
                    "rewards": group["rewards"][:],
                    "terminated": group["terminated"][:],
                    "truncated": group["truncated"][:],
                    "initial_observation": group[
                        "initial_observation"
                    ][:],
                    "terminal_observation": group[
                        "terminal_observation"
                    ][:],
                }

                n = len(episode["actions"])

                assert episode["frames"].shape == (
                    n, 84, 84, 3
                )
                assert episode["initial_observation"].shape == (
                    12, 84, 84
                )
                assert episode["terminal_observation"].shape == (
                    12, 84, 84
                )
                assert len(episode["rewards"]) == n
                assert len(episode["terminated"]) == n
                assert len(episode["truncated"]) == n
                assert bool(
                    episode["terminated"][-1]
                    or episode["truncated"][-1]
                )

                self.episodes.append(episode)
                self.lengths.append(n)

        self.offsets = np.cumsum(
            [0] + self.lengths
        ).tolist()

        print("Dataset loaded.")
        print("Episodes:", len(self.episodes))
        print("Transitions:", len(self))

    def __len__(self):
        return self.offsets[-1]

    @staticmethod
    def build_observation(frames, step):
        stacked = np.zeros(
            (4, 84, 84, 3),
            dtype=np.uint8,
        )

        start = max(0, step - 3)
        selected = frames[start:step + 1]

        stacked[-len(selected):] = selected

        return (
            stacked.transpose(0, 3, 1, 2)
            .reshape(12, 84, 84)
        )

    def __getitem__(self, index):
        episode_idx = (
            bisect.bisect_right(self.offsets, index) - 1
        )

        step = index - self.offsets[episode_idx]
        episode = self.episodes[episode_idx]

        if step == 0:
            state = episode["initial_observation"]
        else:
            state = self.build_observation(
                episode["frames"],
                step,
            )

        terminated = bool(
            episode["terminated"][step]
        )
        truncated = bool(
            episode["truncated"][step]
        )

        if terminated or truncated:
            next_state = episode["terminal_observation"]
        else:
            next_state = self.build_observation(
                episode["frames"],
                step + 1,
            )

        action = episode["actions"][step]
        reward = episode["rewards"][step]

        return (
            torch.from_numpy(state.copy()),
            torch.as_tensor(
                action,
                dtype=torch.float32,
            ),
            torch.tensor(
                reward,
                dtype=torch.float32,
            ),
            torch.from_numpy(next_state.copy()),
            torch.tensor(
                float(terminated),
                dtype=torch.float32,
            ),
        )
