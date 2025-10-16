import argparse, json, glob, os

def main(root, out_json):
    wavs = glob.glob(os.path.join(root, "**/*.flac"), recursive=True)
    items = [{'audio': w} for w in wavs]
    with open(out_json, "w") as f:
        json.dump(items, f, indent=2)
    print("wrote", out_json, len(items), "files")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="LibriSpeech subset root")
    ap.add_argument("--out", required=True)
    main(ap.parse_args().root, ap.parse_args().out)
