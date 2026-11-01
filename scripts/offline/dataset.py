
import bisect

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class ExpertDataset(Dataset):
    def __init__(self, path, episode_names):
        self.path = path
        self.episode_names = list(episode_names)
        self.file = None
        self.lengths = []

        with h5py.File(path, "r") as f:
            for name in self.episode_names:
                self.lengths.append(len(f[name]["actions"]))

        self.offsets = np.cumsum([0] + self.lengths).tolist()

    def __len__(self):
        return self.offsets[-1]

    def __getitem__(self, index):
        if self.file is None:
            self.file = h5py.File(self.path, "r")

        episode_idx = bisect.bisect_right(self.offsets, index) - 1
        step = index - self.offsets[episode_idx]

        group = self.file[self.episode_names[episode_idx]]

        start = max(0, step - 3)
        frames = group["images"][start:step + 1]

        # Match the zero-padding behavior of VecFrameStack
        stacked = np.zeros((4, 84, 84, 3), dtype=np.uint8)
        stacked[-len(frames):] = frames

        # Convert T, H, W, C to T*C, H, W
        stacked = stacked.transpose(0, 3, 1, 2)
        stacked = stacked.reshape(12, 84, 84)

        action = group["actions"][step].astype(np.float32)

        observation = torch.from_numpy(stacked.copy()).float() / 255.0
        target = torch.from_numpy(action.copy())

        return observation, target
