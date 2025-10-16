# tests/sanity_check.py
import os
import sys
import argparse
import yaml
import math
import torch
import torch.nn.functional as F

# Make project importable when run from repo root or tests/
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.models.cc_encoder import CCEncoder
from src.models.decoder import CognitiveDecoder
from src.models.discriminators import MSD, MPD
from src.models.feature_extractors import XiHead
from src.losses import (lsgan_d, lsgan_g, feature_matching_loss, mel_loss, cc_repr_loss)


def bitrate_from_bits(bits_tensor, sr, hop_ms):
    """
    bits_tensor: [B, T_frames, C] with entries in {0,1}
    hop_ms: frame hop in milliseconds (int or float)
    returns kbps (float)
    """
    B, T, C = bits_tensor.shape
    frames_per_sec = 1000.0 / float(hop_ms)
    # bits per second (per stream) = C * frames_per_sec
    bps = C * frames_per_sec
    return bps / 1000.0  # kbps


def check_shapes(x, x_hat, Cs, Cl, name=""):
    assert x.dim() == 3 and x.shape[1] == 1, f"{name} wav must be [B,1,T]"
    assert x_hat.shape[0] == x.shape[0] and x_hat.shape[1] == 1, "decoder output channel mismatch"
    assert Cs.dim() == 3 and Cs.shape[-1] == 64, "Cs should be [B,T_s,64]"
    assert Cl.dim() == 3 and Cl.shape[-1] == 64, "Cl should be [B,T_l,64]"


def main(cfg_path="configs/default.yaml", device=None, seconds=1.0, seed=1234):
    torch.manual_seed(seed)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    cfg = yaml.safe_load(open(cfg_path))
    sr = int(cfg["data"]["sample_rate"])
    seg_T = int(seconds * sr)

    # === 1) Synthesize a simple test waveform (1 kHz tone with small noise) ===
    t = torch.arange(seg_T, dtype=torch.float32) / sr
    wav = 0.15 * torch.sin(2 * math.pi * 1000 * t)  # 1 kHz
    wav += 0.01 * torch.randn_like(wav)
    wav = wav.unsqueeze(0).unsqueeze(0).to(device)  # [1,1,T]

    # === 2) Instantiate models ===
    enc = CCEncoder(
        sr=sr,
        short_cfg=cfg["model"]["enc"]["short"],
        long_cfg=cfg["model"]["enc"]["long"],
    ).to(device).eval()  # encoder is not trained in G step but we need it to produce Cs/Cl

    dec = CognitiveDecoder(cfg["model"]["dec"]["top"],
                           cfg["model"]["dec"]["low"]).to(device).train()

    msd, mpd = MSD().to(device).train(), MPD().to(device).train()

    xi_s = XiHead(sr=sr, kind="short").to(device).train()
    xi_l = XiHead(sr=sr, kind="long").to(device).train()

    # === 3) Forward encoder (no grad needed to simulate codec quantization) ===
    with torch.no_grad():
        Cs = enc.short(wav)   # [B, T_s, 64]
        Cl = enc.long(wav)    # [B, T_l, 64]
        bs, _ = enc.q_short(Cs)  # 1-bit deltas (short)
        bl, _ = enc.q_long(Cl)   # 1-bit deltas (long)

    # === 4) Estimate kbps from produced streams ===
    kbps_s = bitrate_from_bits(bs, sr, cfg["model"]["enc"]["short"]["hop_ms"])
    kbps_l = bitrate_from_bits(bl, sr, cfg["model"]["enc"]["long"]["hop_ms"])
    kbps_total = kbps_s + kbps_l

    # === 5) Decode to waveform ===
    x_hat = dec(Cs, Cl)  # [B,1,T]
    check_shapes(wav, x_hat, Cs, Cl, "sanity")

    # === 6) Discriminate real/fake and compute losses ===
    # Discriminator forward (detach fake for D as usual)
    real_logits_msd, real_feats_msd = msd(wav)
    fake_logits_msd, fake_feats_msd = msd(x_hat.detach())
    real_logits_mpd, real_feats_mpd = mpd(wav)
    fake_logits_mpd, fake_feats_mpd = mpd(x_hat.detach())

    d_loss = lsgan_d(real_logits_msd + real_logits_mpd,
                     fake_logits_msd + fake_logits_mpd)

    # Generator forward through D (not detached)
    g_fake_logits_msd, g_fake_feats_msd = msd(x_hat)
    g_fake_logits_mpd, g_fake_feats_mpd = mpd(x_hat)

    adv = lsgan_g(g_fake_logits_msd + g_fake_logits_mpd)
    fm  = feature_matching_loss(real_feats_msd, g_fake_feats_msd)
    fm += feature_matching_loss(real_feats_mpd, g_fake_feats_mpd)

    mel = mel_loss(wav, x_hat, sr, cfg["train"]["mel"])
    cc_s = cc_repr_loss(xi_s, wav, x_hat)
    cc_l = cc_repr_loss(xi_l, wav, x_hat)

    lam = cfg["loss"]["lambda"]
    g_loss = (lam["adv"] * adv +
              lam["fm"] * fm +
              lam["mel"] * mel +
              lam["cc_short"] * cc_s +
              lam["cc_long"] * cc_l)

    # === 7) Backprop only generator side to ensure grads flow end-to-end ===
    g_params = list(dec.parameters()) + list(xi_s.parameters()) + list(xi_l.parameters())
    for p in g_params:
        if p.grad is not None:
            p.grad.zero_()
    g_loss.backward()
    torch.nn.utils.clip_grad_norm_(g_params, 5.0)

    # (Optional) Take a tiny optimizer step to ensure no errors
    opt_g = torch.optim.AdamW(g_params, lr=1e-4)
    opt_g.step()

    # === 8) Print summary ===
    print("\n=== Sanity Check Report ===")
    print(f"Device: {device}")
    print(f"Input wav: {wav.shape} | Recon: {x_hat.shape}")
    print(f"Cs (short) shape: {Cs.shape} | Cl (long) shape: {Cl.shape}")
    print(f"Δ-bits short: {bs.shape} -> ~{kbps_s:.2f} kbps")
    print(f"Δ-bits long : {bl.shape} -> ~{kbps_l:.2f} kbps")
    print(f"Estimated total bitrate: ~{kbps_total:.2f} kbps")
    print("\nLosses:")
    print(f" D loss: {d_loss.item():.4f}")
    print(f" G loss: {g_loss.item():.4f} | adv {adv.item():.4f} | fm {fm.item():.4f} | mel {mel.item():.4f} | cc_s {cc_s.item():.4f} | cc_l {cc_l.item():.4f}")
    print("Backprop through generator: OK (one AdamW step taken).")
    print("All core modules (Encoder, Decoder, MSD, MPD, Xi-heads) compiled and ran end-to-end.\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="configs/default.yaml", help="Path to YAML config")
    ap.add_argument("--device", default=None, help="'cuda' or 'cpu' (auto if None)")
    ap.add_argument("--seconds", type=float, default=1.0, help="Length of synthetic test tone")
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()
    main(args.cfg, args.device, args.seconds, args.seed)
