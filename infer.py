import torch, argparse, torchaudio, yaml
from src.models.cc_encoder import CCEncoder
from src.models.decoder import CognitiveDecoder

@torch.no_grad()
def run(wav_path, ckpt, cfg_path="configs/default.yaml", out_path="out.wav"):
    cfg = yaml.safe_load(open(cfg_path))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    enc = CCEncoder(sr=cfg['data']['sample_rate'],
                    short_cfg=cfg['model']['enc']['short'],
                    long_cfg=cfg['model']['enc']['long']).to(device).eval()
    dec = CognitiveDecoder(cfg['model']['dec']['top'], cfg['model']['dec']['low']).to(device).eval()

    # load decoder/xihad checkpoint
    state = torch.load(ckpt, map_location=device)
    dec.load_state_dict(state['dec'])

    wav, sr = torchaudio.load(wav_path)
    if sr != cfg['data']['sample_rate']:
        wav = torchaudio.functional.resample(wav, sr, cfg['data']['sample_rate'])
    wav = wav.mean(0, keepdim=True).unsqueeze(0).to(device)  # [1,1,T]

    Cs = enc.short(wav); Cl = enc.long(wav)
    out = dec(Cs, Cl).cpu().squeeze(0)
    torchaudio.save(out_path, out, cfg['data']['sample_rate'])
    print("Wrote", out_path)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--cfg", default="configs/default.yaml")
    ap.add_argument("--out", default="recon.wav")
    args = ap.parse_args()
    run(args.wav, args.ckpt, args.cfg, args.out)
