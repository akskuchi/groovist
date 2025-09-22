import argparse
import clip
import configparser
import json
import numpy as np
import pandas as pd
from statistics import fmean
import sys
import torch
from torch.nn.functional import normalize
import utils


config = configparser.ConfigParser()
config.read('config.ini')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'running on {device}\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='VIST',
                        choices=['VIST', 'AESOP', 'VWP', 'custom'], help='dataset to score')
    parser.add_argument('--input_file', default='data/sample_nphrases.json',
                        help='path to file with NPs (?.json)')
    parser.add_argument('--output_file', default='data/sample_scores.json',
                        help='path to file with GROOViST scores (?.json)')
    parser.add_argument('--average_across_samples', action='store_true',
                        help='compute and print average score across all samples')
    args = parser.parse_args()

    try:
        with open(args.input_file, 'r') as fh:
            nphrases = json.load(fh)
        fh.close()
        print(f'loaded NPs for {len(nphrases)} stories in {args.input_file}\n')
    except Exception as e:
        print(f'unable to read {args.input_file}', e)
        sys.exit(1)

    try:
        theta = config[args.dataset].getfloat('theta')
        sid_2_iids = config[args.dataset]['sid_2_iids_file']
        sid_2_iids = None if sid_2_iids == 'None' else json.load(open(sid_2_iids, 'r'))
        IRs_df = pd.read_csv(config[args.dataset]['image_regions_info_file'])
        IRs_path = config[args.dataset]['image_regions']
        print(f'loaded {args.dataset} dataset configuration\n')
    except Exception as e:
        print(f'unable to read configuration for {args.dataset}', e)
        sys.exit(1)

    try:
        print('bootstrapping the CLIP (ViT-B/32) model...', end='')
        model, preprocess = clip.load("ViT-B/32", device=device, jit=False)
        model.eval()
        model = model.to(device)
        print('complete\n')
    except Exception as e:
        print('unable to load the CLIP model', e)
        sys.exit(1)

    sids, scores = list(nphrases.keys()), []
    for sid in sids:
        print(f'evaluating {sid}...')
        NPs, cr_weights = utils.get_concreteness_ratings(nphrases[sid])
        NPs_tokenized = clip.tokenize(NPs).to(device)
        with torch.no_grad():
            NPs_embs = normalize(model.encode_text(NPs_tokenized), p=2, dim=-1)
        iids = utils.get_image_ids(args.dataset, sid, sid_2_iids)

        try:
            NPs_max_alignment = utils.get_max_alignment_scores(NPs_embs, iids, preprocess, model, IRs_df, IRs_path)
            scores.append(utils.penalize_concretize_normalize(NPs_max_alignment, torch.tensor(cr_weights), theta, NPs))
            print(f'GROOViST score for {sid}: {scores[-1]['groovist']}, in range [-1, 1]: {np.tanh(scores[-1]['groovist'])}\n')
        except Exception as e:
            print(f'{sid} not used for computing GROOViST', e)
            scores.append({'groovist': 0.000, 'NP_scores': None})  # this default score is use-case dependent

    if args.average_across_samples:
        print(f'\noverall GROOViST score: {fmean([s['groovist'] for s in scores])}\n')

    with open(args.output_file, 'w') as fh:
        json.dump(dict(zip(sids, scores)), fh, indent=4)
    fh.close()
    print(f'saved scores to {args.output_file}\n')
