"""
Each reader should return:

    - sentences - This is the primary input (raw text) to the model. Not tokenized.
    - extra - Additional model input such as entity or sentence labels.
    - metadata - Info about the data that is not specific to examples / batches.

"""

import collections
import os
import json

import nltk

from tqdm import tqdm


def pick(lst, k):
    return [d[k] for d in lst]


def flatten_tree(tr):
    def func(tr):
        if not isinstance(tr, (list, tuple)):
            return [tr]
        result = []
        for x in tr:
            result += func(x)
        return result
    return func(tr)


def convert_binary_bracketing(parse, lowercase=True):
    transitions = []
    tokens = []

    for word in parse.split(' '):
        if word[0] != "(":
            if word == ")":
                transitions.append(1)
            else:
                # Downcase all words to match GloVe.
                if lowercase:
                    tokens.append(word.lower())
                else:
                    tokens.append(word)
                transitions.append(0)
    return tokens, transitions


def build_tree(tokens, transitions):
    stack = []
    buf = tokens[::-1]

    for t in transitions:
        if t == 0:
            stack.append(buf.pop())
        elif t == 1:
            right = stack.pop()
            left = stack.pop()
            stack.append((left, right))

    assert len(stack) == 1

    return stack[0]


def get_spans_and_siblings(tree):
    def helper(tr, idx=0, name='root'):
        if isinstance(tr, (str, int)):
            return 1, [(idx, idx+1)], []

        l_size, l_spans, l_sibs = helper(tr[0], name='l', idx=idx)
        r_size, r_spans, r_sibs = helper(tr[1], name='r', idx=idx+l_size)

        size = l_size + r_size

        # Siblings.
        spans = [(idx, idx+size)] + l_spans + r_spans
        siblings = [(l_spans[0], r_spans[0], name)] + l_sibs + r_sibs

        return size, spans, siblings

    _, spans, siblings = helper(tree)

    return spans, siblings


def get_spans(tree):
    def helper(tr, idx=0):
        if isinstance(tr, (str, int)):
            return 1, []

        spans = []
        sofar = idx

        for subtree in tr:
            size, subspans = helper(subtree, idx=sofar)
            spans += subspans
            sofar += size

        size = sofar - idx
        spans += [(idx, sofar)]

        return size, spans

    _, spans = helper(tree)

    return spans


class BaseTextReader(object):
    def __init__(self, lowercase=True, filter_length=0, include_id=False, include_dict=None):
        self.lowercase = lowercase
        self.filter_length = filter_length if filter_length is not None else 0
        self.include_id = include_id
        self.include_dict = include_dict

    def read(self, filename):
        return self.read_sentences(filename)

    def read_sentences(self, filename):
        sentences = []
        extra = collections.defaultdict(list)
        word2idx = None
        if self.include_dict is not None:
            word2idx = json.load(open(self.include_dict, 'r'))

        example_ids = []

        with open(filename) as f:
            for line in tqdm(f, desc='read'):
                line = line.strip()
                if not line or line.startswith('#'):
                    continue

                for s in self.read_line(line):
                    if self.filter_length > 0 and len(s) > self.filter_length:
                        continue
                    if self.include_id:
                        example_id = s[0]
                        s = s[1:]
                    else:
                        example_id = len(sentences)
                    if self.lowercase:
                        s = [w.lower() for w in s]
                    if word2idx is not None:
                        s = [w if w in word2idx else '<unk>' for w in s]

                    sentences.append(s)
                    extra['example_ids'].append(example_id)
                    extra['file_order'].append(len(extra['file_order'])) # Preserves ordering that sentences were read in.
                    # extra['original_input'].append(line)

        metadata = {}
        metadata['word2idx'] = word2idx

        return {
            "sentences": sentences,
            "extra": extra,
            "metadata": metadata
            }

    def read_line(self, line):
        raise NotImplementedError


class PlainTextReader(BaseTextReader):
    def __init__(self, lowercase=True, filter_length=0, delim=' ', include_id=False, include_dict=None):
        super(PlainTextReader, self).__init__(lowercase=lowercase, filter_length=filter_length, include_id=include_id, include_dict=include_dict)
        self.delim = delim

    def read_line(self, line):
        s = line.strip().split(self.delim)
        # if self.lowercase:
        #     s = [w.lower() for w in s]
        yield s


class PTBReader(BaseTextReader):
    def read_line(self, line):
        nltk_tree = nltk.Tree.fromstring(line.strip())
        s = nltk_tree.leaves()
        if self.lowercase:
            s = [w.lower() for w in s]
        yield s


class JSONLReader(object):
    def __init__(self, lowercase=True, filter_length=0, delim=' ', include_id=False):
        self.lowercase = lowercase
        self.filter_length = filter_length if filter_length is not None else 0

    def read(self, filename):
        sentences = []

        # extra
        extra = dict()
        example_ids = []
        trees = []

        # read
        with open(filename) as f:
            for line in tqdm(f, desc='read'):
                ex = json.loads(line)
                example_id = ex['example_id']
                tr = ex['tree']
                if not 'sentence' in ex:
                    ex['sentence'] = flatten_tree(tr)
                s = ex['sentence']

                if self.filter_length > 0 and len(s) > self.filter_length:
                    continue
                if self.lowercase:
                    s = [w.lower() for w in s]

                example_ids.append(example_id)
                sentences.append(s)
                trees.append(tr)

        extra['example_ids'] = example_ids
        extra['trees'] = trees

        return {
            "sentences": sentences,
            "extra": extra
            }


class NLIReader(object):

    LABEL_MAP = {
        "entailment": 0,
        "neutral": 1,
        "contradiction": 2
    }

    def __init__(self, lowercase=True, filter_length=0):
        self.lowercase = lowercase
        self.filter_length = filter_length if filter_length is not None else 0

    @staticmethod
    def build(lowercase=True, filter_length=0):
        return NLISentenceReader(lowercase=True, filter_length=0)

    def read(self, filename):
        return self.read_sentences(filename)

    def read_sentences(self, filename):
        raise NotImplementedError

    def read_line(self, line):
        example = json.loads(line)

        try:
            label = self.read_label(example['gold_label'])
        except:
            return None

        s1, t1 = convert_binary_bracketing(example['sentence1_binary_parse'], lowercase=self.lowercase)
        s2, t2 = convert_binary_bracketing(example['sentence2_binary_parse'], lowercase=self.lowercase)
        example_id = example['pairID']

        return dict(s1=s1, label=label, s2=s2, t1=t1, t2=t2, example_id=example_id) # two sentences and corresponding parse transitions per line

    def read_label(self, label):
        return self.LABEL_MAP[label]


class NLISentenceReader(NLIReader):
    def read_sentences(self, filename):
        sentences = []
        extra = collections.defaultdict(list)
        example_ids = []

        with open(filename) as f:
            for line in tqdm(f, desc='read'):
                smap = self.read_line(line)
                if smap is None:
                    continue

                s1, s2, label = smap['s1'], smap['s2'], smap['label']
                example_id = smap['example_id']
                skip_s1 = self.filter_length > 0 and len(s1) > self.filter_length
                skip_s2 = self.filter_length > 0 and len(s2) > self.filter_length

                if not skip_s1:
                    example_ids.append(example_id + '_1')
                    extra['file_order'].append(len(extra['file_order'])) # Preserves ordering that sentences were read in.
                    sentences.append(s1)
                if not skip_s2:
                    example_ids.append(example_id + '_2')
                    extra['file_order'].append(len(extra['file_order']))
                    sentences.append(s2)

        extra['example_ids'] = example_ids

        return {
            "sentences": sentences,
            "extra": extra,
            }


class ConllReader(object):
    def __init__(self, lowercase=True, filter_length=0):
        self.lowercase = lowercase
        self.filter_length = filter_length if filter_length is not None else 0

    def read(self, filename):
        sentences = []
        extra = {}
        example_ids = []
        entity_labels = []

        with open(filename) as f:
            for line in tqdm(f, desc='read'):
                data = json.loads(line)
                s = data['sentence']

                # skip long sentences
                if self.filter_length > 0 and len(s) > self.filter_length:
                    continue

                sentences.append(s)
                example_ids.append(data['example_id'])
                entity_labels.append(data['entities'])

        extra['example_ids'] = example_ids
        extra['entity_labels'] = entity_labels

        return {
            "sentences": sentences,
            "extra": extra,
            }


class SyntheticReader(object):
    def __init__(self, nexamples=100, embedding_size=10, vocab_size=14, seed=11, minlen=10,
                 maxlen=20, length=None):
        super(SyntheticReader, self).__init__()
        self.nexamples = nexamples
        self.embedding_size = embedding_size
        self.vocab_size = vocab_size
        self.seed = seed
        self.minlen = minlen
        self.maxlen = maxlen
        self.length = length

    def read(self, filename=None):
        min_length = self.minlen
        max_length = self.maxlen

        if self.length is not None:
            min_length = self.length
            max_length = min_length + 1

        sentences = synthesize_training_data(self.nexamples, self.vocab_size,
            min_length=min_length, max_length=max_length, seed=self.seed)

        metadata = {}
        metadata['embeddings'] = np.random.randn(self.vocab_size, self.embedding_size).astype(np.float32)

        return {
            "sentences": sentences,
            "extra": extra,
            "metadata": metadata
            }

class COCOReader(object):
    '''
    from https://github.com/bobwan1995/cliora/blob/master/cliora/data/reading.py
    '''
    def __init__(self, lowercase=True, filter_length=0, delim=' '):
        self.delim = delim
        self.lowercase = lowercase
        self.filter_length = filter_length if filter_length is not None else 0

    def read(self, filename):
        sentences = []
        extra = collections.defaultdict(list)

        example_ids = []
        gts = []
        vis_feats = []
        # word2idx = None
        # if os.path.exists(filename.replace(filename.split('/')[-1], 'coco.dict.json')):
        #     word2idx = json.load(open(filename.replace(filename.split('/')[-1], 'coco.dict.json'), 'r'))

        if 'train' in filename:
            split = 'train'
        elif 'val' in filename:
            split = 'val'
        elif 'test' in filename:
            split = 'test'
        else:
            raise NotImplementedError

        with open(filename.replace(filename.split('/')[-1], f'{split}_id.txt'), 'r') as f:
            fnames = [line.split(' ')[0] for line in f.readlines()]

        with open(filename) as f:
            lines = f.readlines()

        for idx, line in tqdm(enumerate(lines), desc='read'):
            (sent, gt, _, _) = json.loads(line.strip())
            s = sent.strip().split(self.delim)

            if self.filter_length > 0 and len(s) > self.filter_length:
                continue
            if self.lowercase:
                s = [w.lower() for w in s]
            # if word2idx is not None:
            #     s = [w if w in word2idx else '<unk>' for w in s]
            example_ids.append(fnames[idx])
            extra['file_order'].append(len(extra['file_order']))
            sentences.append(s)
            gts.append([tuple(i) for i in gt])

        extra['example_ids'] = example_ids
        extra['GT'] = gts
        metadata = {}
        #metadata['word2idx'] = word2idx

        return {
            "sentences": sentences,
            "extra": extra,
            "metadata": metadata
            }

class COCOASRReader(COCOReader):
    def read(self, filename):
        sentences = []
        extra = dict()

        example_ids = []
        gts = []
        vis_feats = []
        pred2gold = []
        # word2idx = None
        # if os.path.exists(filename.replace(filename.split('/')[-1], 'coco.dict.json')):
        #     word2idx = json.load(open(filename.replace(filename.split('/')[-1], 'coco.dict.json'), 'r'))

        if 'train' in filename:
            split = 'train'
        elif 'val' in filename:
            split = 'val'
        elif 'test' in filename:
            split = 'test'
        else:
            raise NotImplementedError

        with open(filename) as f:
            lines = f.readlines()

        for idx, line in tqdm(enumerate(lines), desc='read'):
            (fname, _, gt, sent, align) = json.loads(line.strip())  # _ is ground truth text sent, and sent is ASR transcript
            s = sent.strip().split(self.delim)

            if self.filter_length > 0 and len(s) > self.filter_length:
                continue
            if self.lowercase:
                s = [w.lower() for w in s]
            # if word2idx is not None:
            #     s = [w if w in word2idx else '<unk>' for w in s]
            example_ids.append(fname)
            sentences.append(s)
            gts.append([tuple(i) for i in gt])
            pred2gold.append(align)

        extra['example_ids'] = example_ids
        extra['GT'] = gts
        extra['align'] = pred2gold
        metadata = {}
        #metadata['word2idx'] = word2idx

        return {
            "sentences": sentences,
            "extra": extra,
            "metadata": metadata
            }