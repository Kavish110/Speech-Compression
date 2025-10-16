import torch
import torch.nn.functional as F
import torchaudio

def lsgan_d(real_logits, fake_logits):
    loss = 0.
    for r in real_logits:
        loss += ((r - 1)**2).mean()
    for f in fake_logits:
        loss += (f**2).mean()
    return loss

def lsgan_g(fake_logits):
    loss = 0.
    for f in fake_logits:
        loss += ((f - 1)**2).mean()
    return loss

def feature_matching_loss(real_feats, fake_feats):
    loss = 0.
    for r_scale, f_scale in zip(real_feats, fake_feats):
        for r, f in zip(r_scale, f_scale):
            loss += F.l1_loss(f, r)
    return loss

def mel_spectrogram(x, sr, n_fft, hop_length, win_length, n_mels):
    mel = torchaudio.transforms.MelSpectrogram(
        sample_rate=sr, n_fft=n_fft, hop_length=hop_length,
        win_length=win_length, n_mels=n_mels, center=True, power=1.0)
    return mel(x)

def mel_loss(x, x_hat, sr, cfg):
    mel_x = mel_spectrogram(x, sr, **cfg)
    mel_y = mel_spectrogram(x_hat, sr, **cfg)
    return F.l1_loss(mel_y, mel_x)

def cc_repr_loss(xi_func, x_real, x_fake):
    # Eq. (3): L1 distance in CC latent spaces (short and long)
    z_r = xi_func(x_real)
    z_f = xi_func(x_fake)
    return F.l1_loss(z_f, z_r)
