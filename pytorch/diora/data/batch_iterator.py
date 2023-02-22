from diora.data.dataloader import FixedLengthBatchSampler, SimpleDataset
from diora.blocks.negative_sampler import choose_negative_samples
from diora.data.utils import boundaries_to_masks, read_textgrid

import torch
import numpy as np


def get_config(config, **kwargs):
    for k, v in kwargs.items():
        if k in config:
            config[k] = v
    return config


def get_default_config():

    default_config = dict(
        batch_size=16,
        forever=False,
        drop_last=False,
        sort_by_length=True,
        shuffle=True,
        random_seed=None,
        filter_length=None,
        pin_memory=False,
        include_partial=False,
        cuda=False,
        ngpus=1,
        k_neg=3,
        negative_sampler=None,
        options_path=None,
        weights_path=None,
        vocab=None,
        length_to_size=None,
        rank=None,
        textgrid_folder=None,
    )

    return default_config


class Collate(object):
    @staticmethod
    def chunk(tensor, chunks, dim=0, i=0):
        if isinstance(tensor, torch.Tensor):
            return torch.chunk(tensor, chunks, dim=dim)[i]
        index = torch.chunk(torch.arange(len(tensor)), chunks, dim=dim)[i]
        return [tensor[ii] for ii in index]

    @staticmethod
    def partition(tensor, rank, device_ids):
        if tensor is None:
            return None
        if isinstance(tensor, dict):
            for k, v in tensor.items():
                tensor[k] = Collate.partition(v, rank, device_ids)
            return tensor
        return Collate.chunk(tensor, len(device_ids), 0, rank)

    def __init__(self, batch_iterator, rank, ngpus):
        self.batch_iterator = batch_iterator
        self.rank = rank
        self.ngpus = ngpus

    def collate_fn(self, batch):
        batch_iterator = self.batch_iterator
        rank = self.rank
        ngpus = self.ngpus

        index, sents = zip(*batch)
        sents = torch.from_numpy(np.array(sents)).long()

        batch_map = {}
        batch_map['index'] = index
        batch_map['sents'] = sents

        for k, v in batch_iterator.extra.items():
            batch_map[k] = [v[idx] for idx in index]

        if ngpus > 1:
            for k in batch_map.keys():
                batch_map[k] = Collate.partition(batch_map[k], rank, range(ngpus))

        return batch_map


class BatchIterator(object):

    def __init__(self, sentences, extra={}, num_workers=0, modality='text', **kwargs):
        self.sentences = sentences
        self.num_workers = num_workers
        self.modality = modality
        self.config = config = get_config(get_default_config(), **kwargs)
        self.extra = extra
        self.loader = None

    def get_dataset_size(self):
        return len(self.sentences)

    def get_dataset_minlen(self):
        return min(map(len, self.sentences))

    def get_dataset_maxlen(self):
        return max(map(len, self.sentences))

    def get_dataset_stats(self):
        return 'size={} minlen={} maxlen={}'.format(
            self.get_dataset_size(), self.get_dataset_minlen(), self.get_dataset_maxlen()
        )

    def choose_negative_samples(self, negative_sampler, k_neg):
        return choose_negative_samples(negative_sampler, k_neg)

    def get_iterator(self, hdf5, **kwargs):
        config = get_config(self.config.copy(), **kwargs)

        random_seed = config.get('random_seed')
        batch_size = config.get('batch_size')
        filter_length = config.get('filter_length')
        pin_memory = config.get('pin_memory')
        include_partial = config.get('include_partial')
        cuda = config.get('cuda')
        ngpus = config.get('ngpus')
        rank = config.get('rank')
        k_neg = config.get('k_neg')
        negative_sampler = config.get('negative_sampler', None)
        num_workers = self.num_workers
        length_to_size = config.get('length_to_size', None)
        if self.modality == 'speech':
            textgrid_folder = config['textgrid_folder']

        collate_fn = Collate(self, rank, ngpus).collate_fn

        if self.loader is None:
            rng = np.random.RandomState(seed=random_seed)
            dataset = SimpleDataset(self.sentences)
            sampler = FixedLengthBatchSampler(dataset, batch_size=batch_size, rng=rng,
                maxlen=filter_length, include_partial=include_partial, length_to_size=length_to_size)
            loader = torch.utils.data.DataLoader(dataset, shuffle=(sampler is None), num_workers=num_workers, pin_memory=pin_memory, batch_sampler=sampler, collate_fn=collate_fn)
            self.loader = loader

        def myiterator():

            for batch in self.loader:
                index = batch['index']
                sentences = batch['sents']

                batch_size, length = sentences.shape

                neg_samples = None
                if negative_sampler is not None:
                    neg_samples = self.choose_negative_samples(negative_sampler, k_neg)

                if cuda and self.modality != 'speech':
                    sentences = sentences.cuda()
                if cuda and self.modality != 'speech' and neg_samples is not None:
                    neg_samples = neg_samples.cuda()

                batch_map = {}
                batch_map['sentences'] = sentences
                batch_map['neg_samples'] = neg_samples
                batch_map['batch_size'] = batch_size
                batch_map['length'] = length

                for k, v in self.extra.items():
                    batch_map[k] = batch[k]

                if self.modality == 'speech':
                    reps = []
                    word_alignments = []
                    for fname in batch_map['example_ids']:
                        reps.append(hdf5[fname][:])
                        _, words = read_textgrid(fname+'.TextGrid', 16000, textgrid_folder)
                        word_alignments.append(words)
                    batch_map['upstream_embeddings'] = torch.nn.utils.rnn.pad_sequence([torch.from_numpy(rep) for rep in reps], batch_first=True)

                    if cuda:
                        batch_map['upstream_embeddings'] = batch_map['upstream_embeddings'].to('cuda')
                    emb_len = batch_map['upstream_embeddings'].shape[1]

                    batch_map['word_masks'] = []

                    for words in word_alignments:
                        # if self.fixed_length_seg:
                        #     l = length
                        #     masks = []
                        #     for i in range(l):
                        #         mask = torch.zeros(emb_len)
                        #         mask[i*(emb_len//l):(i+1)*(emb_len//l)] = 1
                        #         masks.append(mask)
                        #     word_mask = torch.stack(masks)
                        word_mask = boundaries_to_masks(words, emb_len)
                        batch_map['word_masks'].append(word_mask)
                    batch_map['word_masks'] = torch.stack(batch_map['word_masks'])

                    if cuda:
                        batch_map['word_masks'] = batch_map['word_masks'].cuda()
                    
                yield batch_map

        return myiterator()

