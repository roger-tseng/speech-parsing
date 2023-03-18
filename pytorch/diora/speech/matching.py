# Modified from: 
# https://github.com/jefflai108/VGNSL/blob/master/data/networkx_tutorial.py

import argparse
import glob
import json
import os

from praatio import textgrid
from tqdm import tqdm

import networkx as nx
import numpy as np
from networkx.algorithms.bipartite.matrix import from_biadjacency_matrix
from scipy.sparse import csr_matrix

def generate_sim_mat_via_duration(gt_word_list, wrd_seg):
    n = len(gt_word_list)
    m = len(wrd_seg)
    duration_overlap_mat = np.zeros((n, m))

    for i in range(n): 
        for j in range(m):
            gt_s, gt_e = gt_word_list[i][:2]
            pred_s, pred_e = wrd_seg[j][:2]
            # calculate max overlap
            duration_overlap_mat[i,j] = max(0, min(gt_e, pred_e) - max(gt_s, pred_s))

    return duration_overlap_mat

def _permute(edge, sim_mat):
    # Edge not in l,r order. Fix it
    if edge[0] < sim_mat.shape[0]:
        return edge[0], edge[1] - sim_mat.shape[0]
    else:
        return edge[1], edge[0] - sim_mat.shape[0]

def run(gt_word_list, wrd_seg): 
    # return max weight matching nodes from a bipartite graph. 
    # distance + min-match == -distance + max-match 
    # 
    # reference https://github.com/cisnlp/simalign/blob/05332bf2f6ccde075c3aba94248d6105d9f95a00/simalign/simalign.py#L96-L103

    duration_mat = generate_sim_mat_via_duration(gt_word_list, wrd_seg)
    sim_mat = duration_mat

    G = from_biadjacency_matrix(csr_matrix(sim_mat))
    matching = nx.max_weight_matching(G, maxcardinality=True)
    matching = [_permute(x, sim_mat) for x in matching]
    matching = [[int(x[0]), int(x[1])] for x in matching]

    return matching 

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--asr_alignment_folder", type=str, required=True)
    parser.add_argument("--gold_alignment_folder", type=str, required=True)
    parser.add_argument("--labelled_ifile", type=str, required=True)
    parser.add_argument("--split", choices=('val', 'test'))
    parser.add_argument("--ofile", type=str, required=True)
    return parser.parse_args()

def main(args):
    split = args.split
    name, ext = args.ofile.rsplit('.', 1)
    assert os.path.exists(args.asr_alignment_folder)
    assert os.path.exists(args.gold_alignment_folder)

    with open(args.labelled_ifile.replace(args.labelled_ifile.split('/')[-1], f'{split}_id.txt')) as f_id:
        fnames = [line.split(' ')[0] for line in f_id.readlines()]
    base_fnames = [i.rsplit('/', 1)[1] for i in fnames]

    with open(args.labelled_ifile) as f_label:
        trees = [tuple(json.loads(line.strip())[:2]) for line in f_label.readlines()]
    
    assert len(fnames) == len(trees)
    print(f"Matching via duration for {split} set")
    #assert not os.path.exists(f'{name}_{split}_duration.{ext}')
    with open(f'{name}_{split}_duration.{ext}', 'w') as f:
        asr = [i.rsplit('/', 1)[1] for i in glob.glob(os.path.join(args.asr_alignment_folder, f'{split}/*/*.TextGrid'))]
        gold = [i.rsplit('/', 1)[1] for i in glob.glob(os.path.join(args.gold_alignment_folder, f'{split}/*/*.TextGrid'))]
        delimiter = '-' if '-' in asr[0] else '_'
        count = 0
        for i in tqdm(gold):
            if i not in asr or i.rsplit('.', 1)[0] not in base_fnames:
                #print(i, 'is not in gold!')
                continue
            else:
                count += 1
                tg_gold = textgrid.openTextgrid(os.path.join(args.gold_alignment_folder, split, i.split(delimiter, 1)[0], i), includeEmptyIntervals=False)
                tg_asr = textgrid.openTextgrid(os.path.join(args.asr_alignment_folder, split, i.split(delimiter, 1)[0], i), includeEmptyIntervals=False)
                gold_words = tg_gold.tierDict['words'].entryList
                asr_words = tg_asr.tierDict['words'].entryList
                matching = run(gold_words, asr_words)
                #print([a.label for a in gold_words])
                #print([a.label for a in asr_words])
                #print([(gold_words[a].label, asr_words[b].label) for a,b in matching])
                #print(i.rsplit('.', 1)[0], matching, file=f)
                idx = base_fnames.index(i.rsplit('.', 1)[0])
                print(json.dumps([fnames[idx], trees[idx][0], trees[idx][1], ' '.join([interval.label for interval in asr_words]), matching]), file=f)

if __name__ == '__main__':
    args = get_args()
    main(args)