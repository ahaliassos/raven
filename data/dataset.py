import os

import cv2
import numpy as np
import torch
import torchaudio
from torch.utils.data import Dataset


def cut_or_pad(data, size, dim=0):
    # Pad with zeros on the right if data is too short
    if data.size(dim) < size:
        # assert False
        padding = size - data.size(dim)
        data = torch.from_numpy(np.pad(data, (0, padding), "constant"))
    # Cut from the right if data is too long
    elif data.size(dim) > size:
        data = data[:size]
    # Keep if data is exactly right
    assert data.size(dim) == size
    return data


class AVDataset(Dataset):
    def __init__(
            self, 
            data_path,
            video_path_prefix_lrs2,
            audio_path_prefix_lrs2,
            video_path_prefix_lrs3, 
            audio_path_prefix_lrs3, 
            video_path_prefix_vox2=None, 
            audio_path_prefix_vox2=None, 
            transforms=None,
            modality="audiovisual",
        ):

        self.data_path = data_path
        self.video_path_prefix_lrs3 = video_path_prefix_lrs3
        self.audio_path_prefix_lrs3 = audio_path_prefix_lrs3
        self.video_path_prefix_vox2 = video_path_prefix_vox2
        self.audio_path_prefix_vox2 = audio_path_prefix_vox2
        self.video_path_prefix_lrs2 = video_path_prefix_lrs2
        self.audio_path_prefix_lrs2 = audio_path_prefix_lrs2
        self.transforms = transforms
        self.modality = modality

        self.paths_counts_labels = self.configure_files()
        self.num_fails = 0
    
    def configure_files(self):
        # from https://github.com/facebookresearch/pytorchvideo/blob/874d27cb55b9d7e9df6cd0881e2d7fe9f262532b/pytorchvideo/data/labeled_video_paths.py#L37
        paths_counts_labels = []
        with open(self.data_path, "r") as f:
            for path_count_label in f.read().splitlines():
                tag, file_path, count, label = path_count_label.split(",")
                paths_counts_labels.append((tag, file_path, int(count), [int(lab) for lab in label.split()]))
        return paths_counts_labels

    def load_video(self, path):
        cap = cv2.VideoCapture(path)
        frames = []
        while True:
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
            else:
                break
        cap.release()
        if not frames:
            print(path)
            return None
        frames = torch.from_numpy(np.stack(frames))
        frames = frames.permute((3, 0, 1, 2))  # TxHxWxC -> # CxTxHxW
        return frames
    
    def load_audio(self, path):
        audio = torchaudio.load(path, normalize=True)[0]
        return audio
        
    def __len__(self):
        return len(self.paths_counts_labels)

    def __getitem__(self, index):
        tag, file_path, _, label = self.paths_counts_labels[index]
        self.video_path_prefix = getattr(self, f"video_path_prefix_{tag}", "")
        self.audio_path_prefix = getattr(self, f"audio_path_prefix_{tag}", "")

        if self.modality == "video":
            data = self.load_video(os.path.join(self.video_path_prefix, file_path))
            if data is None:
                self.num_fails += 1
                if self.num_fails == 200:
                    raise ValueError("Too many file errors.")
                return {'data': None, 'label': None}
            data = self.transforms['video'](data).permute((1, 2, 3, 0))
        elif self.modality == "audio":
            data = self.load_audio(os.path.join(self.audio_path_prefix, file_path[:-4] + ".wav"))
            data = self.transforms['audio'](data).squeeze(0)
        elif self.modality == "audiovisual":
            video = self.load_video(os.path.join(self.video_path_prefix, file_path))
            if video is None:
                self.num_fails += 1
                if self.num_fails == 200:
                    raise ValueError("Too many file errors.")
                return {'video': None, 'audio': None, 'label': None}
            audio = self.load_audio(os.path.join(self.audio_path_prefix, file_path[:-4] + ".wav"))
            audio = cut_or_pad(audio.squeeze(0), video.size(1) * 640)
            video = self.transforms['video'](video).permute((1, 2, 3, 0))
            audio = self.transforms['audio'](audio.unsqueeze(0)).squeeze(0)
            return {'video': video, 'audio': audio, 'label': torch.tensor(label)}

        return {'data': data, 'label': torch.tensor(label)}
