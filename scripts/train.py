import os, math, yaml, torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.datasets import WavChunks
from src.models.cc_encoder import CCEncoder
from src.models.decoder import CognitiveDecoder
from src.models.discriminators import MSD, MPD
from src.models.feature_extractors import XiHead
from src.losses import (lsgan_d, lsgan_g, feature_matching_loss, mel_loss, cc_repr_loss)

def main(cfg_path="configs/default.yaml"):
    cfg = yaml.safe_load(open(cfg_path))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    ds_tr = WavChunks(cfg['data']['train_manifest'], cfg['data']['sample_rate'], cfg['data']['segment_seconds'])
    dl = DataLoader(ds_tr, batch_size=cfg['train']['batch_size'], shuffle=True, num_workers=4, drop_last=True)

    enc = CCEncoder(sr=cfg['data']['sample_rate'],
                    short_cfg=cfg['model']['enc']['short'],
                    long_cfg=cfg['model']['enc']['long']).to(device)
    dec = CognitiveDecoder(cfg['model']['dec']['top'], cfg['model']['dec']['low']).to(device)
    msd, mpd = MSD().to(device), MPD().to(device)

    # ξ heads for CC-representation distances
    xi_s = XiHead(sr=cfg['data']['sample_rate'], kind='short').to(device)
    xi_l = XiHead(sr=cfg['data']['sample_rate'], kind='long').to(device)

    g_params = list(dec.parameters()) + list(xi_s.parameters()) + list(xi_l.parameters())
    d_params = list(msd.parameters()) + list(mpd.parameters())
    opt_g = torch.optim.AdamW(g_params, lr=cfg['train']['lr'], betas=(0.8, 0.99))
    opt_d = torch.optim.AdamW(d_params, lr=cfg['train']['lr'], betas=(0.8, 0.99))

    step, total = 0, cfg['train']['total_steps']
    while step < total:
        for wav in dl:
            step += 1
            wav = wav.to(device)

            # ===== ENCODE & QUANTIZE =====
            with torch.no_grad():
                Cs = enc.short(wav)   # [B, T_s, 64]
                Cl = enc.long(wav)    # [B, T_l, 64]
                # naive upsample long to short grid via repeat/interp for training convenience
                # (decoder does deconv-based upsampling internally)
                bs, _ = enc.q_short(Cs)
                bl, _ = enc.q_long(Cl)

            # ===== DECODE =====
            x_hat = dec(Cs, Cl)  # [B, 1, T]

            # ===== DISCRIMINATE =====
            # Real/fake
            real_logits_msd, real_feats_msd = msd(wav)
            fake_logits_msd, fake_feats_msd = msd(x_hat.detach())
            real_logits_mpd, real_feats_mpd = mpd(wav)
            fake_logits_mpd, fake_feats_mpd = mpd(x_hat.detach())

            # ----- Train D (LSGAN) -----
            opt_d.zero_grad(set_to_none=True)
            d_loss = lsgan_d(real_logits_msd + real_logits_mpd,
                             fake_logits_msd + fake_logits_mpd)
            d_loss.backward()
            torch.nn.utils.clip_grad_norm_(d_params, cfg['train']['grad_clip'])
            opt_d.step()

            # ----- Train G -----
            opt_g.zero_grad(set_to_none=True)
            g_fake_logits_msd, g_fake_feats_msd = msd(x_hat)
            g_fake_logits_mpd, g_fake_feats_mpd = mpd(x_hat)
            adv = lsgan_g(g_fake_logits_msd + g_fake_logits_mpd)

            # Feature matching (sum over all scales/periods)
            fm  = feature_matching_loss(real_feats_msd, g_fake_feats_msd)
            fm += feature_matching_loss(real_feats_mpd, g_fake_feats_mpd)

            # Mel L1
            mel = mel_loss(wav, x_hat, cfg['data']['sample_rate'], cfg['train']['mel'])

            # CC representation distances (Eq. 3) on short/long heads
            cc_s = cc_repr_loss(xi_s, wav, x_hat)
            cc_l = cc_repr_loss(xi_l, wav, x_hat)

            lam = cfg['loss']['lambda']
            g_loss = (lam['adv']*adv + lam['fm']*fm + lam['mel']*mel +
                      lam['cc_short']*cc_s + lam['cc_long']*cc_l)
            g_loss.backward()
            torch.nn.utils.clip_grad_norm_(g_params, cfg['train']['grad_clip'])
            opt_g.step()

            if step % cfg['train']['log_every'] == 0:
                print(f"step {step}/{total} | D {d_loss:.3f} | G {g_loss:.3f} (adv {adv:.3f} fm {fm:.3f} mel {mel:.3f} cc_s {cc_s:.3f} cc_l {cc_l:.3f})")

            if step % cfg['train']['ckpt_every'] == 0:
                os.makedirs("checkpoints", exist_ok=True)
                torch.save({'dec': dec.state_dict(), 'xi_s': xi_s.state_dict(),
                            'xi_l': xi_l.state_dict()},
                           f"checkpoints/codec_step{step}.pt")

            if step >= total:
                break

if __name__ == "__main__":
    main()
