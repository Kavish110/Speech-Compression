import json, random, torch, torchaudio
from torch.utils.data import Dataset

class WavChunks(Dataset):
    def __init__(self, manifest_json, sample_rate=16000, seg_sec=2.0):
        self.items = json.load(open(manifest_json))
        self.sr = sample_rate
        self.n = int(seg_sec * sample_rate)

    def __len__(self): return len(self.items)

    def __getitem__(self, idx):
        path = self.items[idx]['audio']
        wav, sr = torchaudio.load(path)
        if sr != self.sr:
            wav = torchaudio.functional.resample(wav, sr, self.sr)
        wav = wav.mean(0, keepdim=True)  # mono
        if wav.size(-1) < self.n:
            pad = self.n - wav.size(-1)
            wav = torch.nn.functional.pad(wav, (0,pad))
        start = random.randint(0, wav.size(-1)-self.n)
        seg = wav[:, start:start+self.n]
        return seg
