import h5py
import os
import argparse
import torch
import tqdm
import torchaudio.transforms as T
import numpy as np
import torchaudio
import glob 

from diora.data.utils import get_upstream

def main(args):

    device = 'cuda' if args.cuda else 'cpu'

    h5_files = []
    for layer in args.upstream_layers:
        h5_files.append(h5py.File(f'-{layer}.'.join(args.out_fname.rsplit('.',1)), "w"))

    upstream = get_upstream(args.upstream, cuda=args.cuda)
    print("Upstream is on:", next(upstream.parameters()).device)
    for fname in tqdm.tqdm((glob.glob(os.path.join(args.audio_root, "**", "*."+args.ext), recursive=True))):
        fname = fname.rsplit(".", 1)[0]
        wav, sr = torchaudio.load(fname+'.'+args.ext)
        wav = wav.squeeze()
        if sr != 16000:
            sampler = T.Resample(sr, 16000, dtype=wav.dtype)
            wav = sampler(wav)
        wav = wav.to(device)
        with torch.no_grad():
            reps = upstream([wav])
        for layer, h5_file in zip(args.upstream_layers, h5_files):
            rep = reps["hidden_states"][layer].cpu().numpy().squeeze()
            h5_file.create_dataset(fname, data=rep)
    for f in h5_files:
        f.close()

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--audio_root', type=str)
    parser.add_argument('--out_fname', type=str)
    parser.add_argument('--upstream', type=str)
    parser.add_argument('--ext', type=str)
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--upstream_layers', nargs='+', type=int)
    args = parser.parse_args()
    main(args)