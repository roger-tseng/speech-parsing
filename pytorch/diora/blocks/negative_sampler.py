from collections import Counter

import numpy as np
import random
import torch

from tqdm import tqdm
from diora.data.utils import boundaries_to_masks, read_textgrid


def choose_negative_samples(negative_sampler, k_neg):
    neg_samples = negative_sampler.sample(k_neg)
    # neg_samples = torch.from_numpy(neg_samples)
    return neg_samples


def calculate_freq_dist(data, vocab_size):
    # TODO: This becomes really slow on large datasets.
    counter = Counter()
    for i in range(vocab_size):
        counter[i] = 0
    for x in tqdm(data, desc='freq_dist'):
        counter.update(x)
    freq_dist = [v for k, v in sorted(counter.items(), key=lambda x: x[0])]
    freq_dist = np.asarray(freq_dist, dtype=np.float32)
    return freq_dist


class NegativeSampler:
    def __init__(self, freq_dist, dist_power, epsilon=10**-2):
        self.dist = freq_dist ** dist_power + epsilon * (1/len(freq_dist))
        self.dist = self.dist / sum(self.dist)      # Final distribution should be normalized
        self.rng = np.random.RandomState()

    def set_seed(self, seed):
        self.rng.seed(seed)

    def sample(self, num_samples):
        return torch.from_numpy(self.rng.choice(len(self.dist), num_samples, p=self.dist, replace=True))

class NegativeSamplerByFile:
    '''
    Draw negative samples from randomly selected files instead of distribution over entire corpus
    '''
    def __init__(self, file_list, options, is_train_set):
        self.file_list = file_list
        self.rng = np.random.RandomState()
        self.options = options
        if is_train_set:
            self.hdf = options.train_hdf5
            self.textgrid_folder = options.train_textgrid_folder
        else:
            self.hdf = options.valid_hdf5
            self.textgrid_folder = options.valid_textgrid_folder

    def set_seed(self, seed):
        self.rng.seed(seed)

    def sample(self, num_samples):
        reps = []
        word_alignments = []
        for fname in random.choices(self.file_list, k=num_samples):
            reps.append(self.hdf[fname][:])
            _, words = read_textgrid(fname+'.TextGrid', 16000, self.textgrid_folder)
            word_alignments.append([random.choice(words)])
        upstream_embeddings = torch.nn.utils.rnn.pad_sequence([torch.from_numpy(rep) for rep in reps], batch_first=True)

        if self.options.cuda:
            upstream_embeddings = upstream_embeddings.to('cuda')
        emb_len = upstream_embeddings.shape[1]

        word_masks = []
        for words in word_alignments:
            word_mask = boundaries_to_masks(words, emb_len)
            word_masks.append(word_mask)
        word_masks = torch.stack(word_masks)
        if self.options.cuda:
            word_masks = word_masks.cuda()
        return upstream_embeddings, word_masks

