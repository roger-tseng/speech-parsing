import argparse
import datetime
import math
import os
import random
import sys
import uuid

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

import h5py

from diora.analysis.utils import *
from diora.analysis.cky import ParsePredictor as CKY

from diora.data.dataset import ConsolidateDatasets, ReconstructDataset, make_batch_iterator

from diora.utils.path import package_path
from diora.logging.configuration import configure_experiment, get_logger
from diora.utils.flags import stringify_flags, init_with_flags_file, save_flags
from diora.utils.checkpoint import update_best_model

from diora.net.experiment_logger import ExperimentLogger


data_types_choices = ('coco', 'coco_asr', 'nli', 'conll_jsonl', 'txt', 'txt_id', 'synthetic', 'jsonl', 'ptb')


def count_params(net):
    return sum([x.numel() for x in net.parameters() if x.requires_grad])


def build_net(options, embeddings, batch_iterator=None):
    from diora.net.trainer import build_net

    trainer = build_net(options, embeddings, batch_iterator, random_seed=options.seed)

    logger = get_logger()
    logger.info('# of params = {}'.format(count_params(trainer.net)))

    return trainer

def seed_all(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def generate_seeds(n):
    seeds = [random.randint(0, 2**16) for _ in range(n)]
    return seeds


def run_train(options, train_iterator, trainer, validation_iterator):
    logger = get_logger()
    experiment_logger = ExperimentLogger()

    logger.info('Running train.')

    seeds = generate_seeds(options.max_epoch)

    word2idx = train_iterator.word2idx
    idx2word = {v: k for k, v in word2idx.items()}

    step = 0
    best_f1 = 0.
    best_unsup_f1 = 0.
    best_loss = 1000.
    # corpus_f1, loss_name, loss = run_eval(options, trainer, validation_iterator)
    for epoch, seed in zip(range(options.max_epoch), seeds):
        # --- Train--- #

        seed = seeds[epoch]

        logger.info('epoch={} seed={}'.format(epoch, seed))

        def myiterator():
            it = train_iterator.get_iterator(options.train_hdf5, random_seed=seed)

            count = 0

            for batch_map in it:
                # TODO: Skip short examples (optionally).
                if batch_map['length'] <= 2:
                    continue

                yield count, batch_map
                count += 1

        for batch_idx, batch_map in myiterator():
            if options.finetune and step >= options.finetune_after:
                trainer.freeze_diora()

            result = trainer.step(batch_map)

            experiment_logger.record(result)

            if step % options.log_every_batch == 0:
                experiment_logger.log_batch(epoch, step, batch_idx, batch_size=options.batch_size)

            # -- Periodic Checkpoints -- #

            if not options.multigpu or options.local_rank == 0:
                if step % options.save_latest == 0 and step >= options.save_after:
                    logger.info('Saving model (periodic).')
                    trainer.save_model(os.path.join(options.experiment_path, 'model_periodic.pt'), save_emb=(options.emb == 'none'))

                if step % options.save_distinct == 0 and step >= options.save_after:
                    logger.info('Saving model (distinct).')
                    trainer.save_model(os.path.join(options.experiment_path, 'model.step_{}.pt'.format(step)), save_emb=(options.emb == 'none'))
                    
                    if options.modality == 'speech':
                        corpus_f1, _, loss = run_eval(options, trainer, validation_iterator)
                        if corpus_f1 > best_f1:
                            best_f1 = corpus_f1
                            update_best_model(
                                os.path.join(options.experiment_path, 'model.step_{}.pt'.format(step)),
                                os.path.join(options.experiment_path, 'model.best.pt')
                            )

                        if loss < best_loss:
                            best_loss = loss
                            best_unsup_f1 = corpus_f1
                            update_best_model(
                                os.path.join(options.experiment_path, 'model.step_{}.pt'.format(step)),
                                os.path.join(options.experiment_path, 'model.unsup_best.pt')
                            )
                        logger.info('Periodic model corpus_f1 with trivial: {}, loss: {}, best_trivial_corpus_f1: {}, best_trivial_unsup_corpus_f1: {}, best_loss: {}.'.format(corpus_f1, loss, best_f1, best_unsup_f1, best_loss))
                        #logger.info('Periodic model corpus_f1: {}, best_corpus_f1: {}, sent_f1: {}.'.format(corpus_f1, best_f1, sent_f1))
            del result

            if options.max_step is not None and step >= options.max_step:
                logger.info('Max-Step={} Quitting.'.format(options.max_step))
                if options.train_hdf5 is not None:
                    options.train_hdf5.close()
                if options.valid_hdf5 is not None:
                    options.valid_hdf5.close()
                sys.exit()
            
            step += 1

        experiment_logger.log_epoch(epoch, step)

        # Epoch Eval and Checkpoints -- #
        if not options.multigpu or options.local_rank == 0:
            trainer.save_model(os.path.join(options.experiment_path, 'model.epoch_{}.pt'.format(epoch)), save_emb=(options.emb == 'none'))

            corpus_f1, _, loss = run_eval(options, trainer, validation_iterator)
            if corpus_f1 > best_f1:
                best_f1 = corpus_f1
                update_best_model(
                    os.path.join(options.experiment_path, 'model.epoch_{}.pt'.format(epoch)),
                    os.path.join(options.experiment_path, 'model.best.pt')
                )
            if loss < best_loss:
                best_loss = loss
                best_unsup_f1 = corpus_f1
                update_best_model(
                    os.path.join(options.experiment_path, 'model.epoch_{}.pt'.format(epoch)),
                    os.path.join(options.experiment_path, 'model.unsup_best.pt')
                )
            logger.info('Saving model epoch {},  corpus_f1 with trivial: {}, loss: {}, best_trivial_corpus_f1: {}, best_trivial_unsup_corpus_f1: {}, best_loss: {}.'.format(epoch, corpus_f1, loss, best_f1, best_unsup_f1, best_loss))
        
        if options.max_step is not None and step >= options.max_step:
            logger.info('Max-Step={} Quitting.'.format(options.max_step))
            if options.train_hdf5 is not None:
                options.train_hdf5.close()
            if options.valid_hdf5 is not None:
                options.valid_hdf5.close()
            sys.exit()

    if options.train_hdf5 is not None:
        options.train_hdf5.close()
    if options.valid_hdf5 is not None:
        options.valid_hdf5.close()

def run_eval(options, trainer, validation_iterator):
    logger = get_logger()
    sparseval = options.validation_data_type == 'coco_asr'
    # Eval mode.
    trainer.net.eval()
    if options.multigpu:
        diora = trainer.net.module.diora
    else:
        diora = trainer.net.diora
    # diora.outside = False s

    override_init_with_batch(diora)
    override_inside_hook(diora)
    parse_predictor = CKY(net=diora)

    batches = validation_iterator.get_iterator(options.valid_hdf5, random_seed=options.seed)

    logger.info('####### Beginning Eval #######')

    corpus_f1 = [0., 0., 0.]
    sent_f1 = []
    with torch.no_grad():
        for i, batch_map in tqdm(enumerate(batches)):
            sentences = batch_map['sentences']
            length = sentences.shape[1]

            # Skip very short sentences.
            if length <= 2:
                continue

            result = trainer.step(batch_map, train=False, compute_loss=True)
            for k, v in result.items():
                if 'loss' in k:
                    break


            # Parsing eval
            trees = parse_predictor.parse_batch(batch_map)

            for bid, tr in enumerate(trees):
                gold_spans = set(batch_map['GT'][bid])
                pred_actions = get_actions(str(tr))
                pred_spans = set(get_spans(pred_actions))
                if sparseval:
                    pred2gold = {k: v for k, v in batch_map['align'][bid]}
                    pred_spans = set((pred2gold[l], pred2gold[r]) for l, r in pred_spans if (l in pred2gold and r in pred2gold))
                tp, fp, fn = get_stats(pred_spans, gold_spans)
                corpus_f1[0] += tp
                corpus_f1[1] += fp
                corpus_f1[2] += fn

                # # SentF1
                # overlap = pred_spans.intersection(gold_spans)
                # prec = float(len(overlap)) / (len(pred_spans) + 1e-8)
                # reca = float(len(overlap)) / (len(gold_spans) + 1e-8)
                # if len(gold_spans) == 0:
                #     reca = 1.
                #     if len(pred_spans) == 0:
                #         prec = 1.
                # f1 = 2 * prec * reca / (prec + reca + 1e-8)
                # sent_f1.append(f1)

    tp, fp, fn = corpus_f1
    prec = tp / (tp + fp)
    recall = tp / (tp + fn)
    corpus_f1 = 2 * prec * recall / (prec + recall) if prec + recall > 0 else 0.
    # n_sent = len(sent_f1)
    # prec_with_trivial = (tp + n_sent) / (tp + n_sent + fp)
    # recall_with_trivial = (tp + n_sent) / (tp + n_sent + fn)
    # corpus_f1_with_trivial = 2 * prec_with_trivial * recall_with_trivial / (prec_with_trivial + recall_with_trivial) if prec_with_trivial + recall_with_trivial > 0 else 0.
    # sent_f1 = np.mean(np.array(sent_f1))
    logger.info('corpus_f1:{} \t {}:{}'.format(corpus_f1, k, v))

    # Train mode.
    diora.outside = True
    trainer.net.train()
    return corpus_f1, k, v

def get_train_dataset(options):
    return ReconstructDataset().initialize(options, text_path=options.train_path,
        embeddings_path=options.embeddings_path, filter_length=options.train_filter_length,
        data_type=options.train_data_type)


def get_train_iterator(options, dataset):
    return make_batch_iterator(options, dataset, shuffle=True,
            include_partial=False, filter_length=options.train_filter_length,
            batch_size=options.batch_size, length_to_size=options.length_to_size,
            is_train_set=True)


def get_validation_dataset(options):
    return ReconstructDataset().initialize(options, text_path=options.validation_path,
            embeddings_path=options.embeddings_path, filter_length=options.validation_filter_length,
            data_type=options.validation_data_type)


def get_validation_iterator(options, dataset):
    return make_batch_iterator(options, dataset, shuffle=False,
            include_partial=True, filter_length=options.validation_filter_length,
            batch_size=options.validation_batch_size, length_to_size=options.length_to_size,
            is_train_set=False)


def get_train_and_validation(options):
    train_dataset = get_train_dataset(options)
    validation_dataset = get_validation_dataset(options)

    # Modifies datasets. Unifying word mappings, embeddings, etc.
    if options.emb != 'none':
        ConsolidateDatasets([train_dataset, validation_dataset]).run()

    return train_dataset, validation_dataset


def run(options):

    seed_all(options.seed)

    if options.train_hdf5 is not None and options.valid_hdf5 is not None and options.train_hdf5 == options.valid_hdf5:
        options.train_hdf5 = options.valid_hdf5 = h5py.File(options.train_hdf5, 'r')
    else:
        if options.train_hdf5 is not None:
            options.train_hdf5 = h5py.File(options.train_hdf5, 'r')
        if options.valid_hdf5 is not None:
            options.valid_hdf5 = h5py.File(options.valid_hdf5, 'r')

    logger = get_logger()
    experiment_logger = ExperimentLogger()

    train_dataset, validation_dataset = get_train_and_validation(options)
    train_iterator = get_train_iterator(options, train_dataset)
    validation_iterator = get_validation_iterator(options, validation_dataset)
    embeddings = train_dataset['embeddings']

    logger.info('Initializing model.')
    trainer = build_net(options, embeddings, validation_iterator)
    logger.info('Model:')
    for name, p in trainer.net.named_parameters():
        logger.info('{} {}'.format(name, p.shape))

    if options.save_init:
        logger.info('Saving model (init).')
        trainer.save_model(os.path.join(options.experiment_path, 'model_init.pt'), save_emb=(options.emb == 'none'))

    run_train(options, train_iterator, trainer, validation_iterator)


def argument_parser():
    parser = argparse.ArgumentParser()

    # Debug.
    parser.add_argument('--seed', default=11, type=int)
    parser.add_argument('--git_sha', default=None, type=str)
    parser.add_argument('--git_branch_name', default=None, type=str)
    parser.add_argument('--git_dirty', default=None, type=str)
    parser.add_argument('--uuid', default=None, type=str)
    parser.add_argument('--model_flags', default=None, type=str,
                        help='Load model settings from a flags file.')
    parser.add_argument('--flags', default=None, type=str,
                        help='Load any settings from a flags file.')

    parser.add_argument('--master_addr', default='127.0.0.1', type=str)
    parser.add_argument('--master_port', default='29500', type=str)
    parser.add_argument('--world_size', default=None, type=int)

    # Pytorch
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--multigpu', action='store_true')
    parser.add_argument("--local_rank", default=None, type=int) # for distributed-data-parallel
    parser.add_argument("--num_workers", default=0, type=int) # for batch iterator

    # Logging.
    parser.add_argument('--default_experiment_directory', default=os.path.join(package_path(), '..', 'log'), type=str)
    parser.add_argument('--experiment_name', default=None, type=str)
    parser.add_argument('--experiment_path', default=None, type=str)
    parser.add_argument('--log_every_batch', default=10, type=int)
    parser.add_argument('--save_latest', default=1000, type=int)
    parser.add_argument('--save_distinct', default=50000, type=int)
    parser.add_argument('--save_after', default=1000, type=int)
    parser.add_argument('--save_init', action='store_true')

    # Loading.
    parser.add_argument('--load_model_path', default=None, type=str)

    # Data.
    parser.add_argument('--data_type', default='nli', choices=data_types_choices)
    parser.add_argument('--train_data_type', default=None, choices=data_types_choices)
    parser.add_argument('--validation_data_type', default=None, choices=data_types_choices)
    parser.add_argument('--train_path', default=os.path.expanduser('~/data/snli_1.0/snli_1.0_train.jsonl'), type=str)
    parser.add_argument('--validation_path', default=os.path.expanduser('~/data/snli_1.0/snli_1.0_dev.jsonl'), type=str)
    parser.add_argument('--embeddings_path', default=os.path.expanduser('~/data/glove/glove.6B.300d.txt'), type=str)
    parser.add_argument('--dict', default=None, type=str, 
                        help="Path of precomputed dictionary. If not specified, is computed on the fly, and no OOVs will exist.")

    # Data (preprocessing).
    parser.add_argument('--uppercase', action='store_true')
    parser.add_argument('--train_filter_length', default=50, type=int)
    parser.add_argument('--validation_filter_length', default=0, type=int)

    # Model.
    parser.add_argument('--arch', default='treelstm', choices=('treelstm', 'mlp', 'mlp-shared'))
    parser.add_argument('--hidden_dim', default=10, type=int)
    parser.add_argument('--normalize', default='unit', choices=('none', 'unit'))
    parser.add_argument('--compress', action='store_true',
                        help='If true, then copy root from inside chart for outside. ' + \
                             'Otherwise, learn outside root as bias.')

    # Model (Objective).
    parser.add_argument('--reconstruct_mode', default='margin', choices=('margin', 'softmax'))

    # Model (Embeddings).
    parser.add_argument('--emb', default='w2v', choices=('w2v', 'elmo', 'both', 'none'))
    parser.add_argument('--emb_dim', type=int, help='sets dimensions of word embeddings when emb is none')

    # Model (Speech).
    parser.add_argument('--modality', default='text', choices=('speech', 'text'))
    # parser.add_argument('--upstream_model', default='hubert')
    # parser.add_argument('--upstream_layer', default=6, type=int)
    parser.add_argument('--train_textgrid_folder', type=str)
    parser.add_argument('--valid_textgrid_folder', type=str)
    parser.add_argument('--train_hdf5', type=str)
    parser.add_argument('--valid_hdf5', type=str)

    # Model (Negative Sampler).
    parser.add_argument('--margin', default=1, type=float)
    parser.add_argument('--k_neg', default=3, type=int)
    parser.add_argument('--freq_dist_power', default=0.75, type=float)

    # ELMo
    parser.add_argument('--elmo_options_path', default='https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json', type=str)
    parser.add_argument('--elmo_weights_path', default='https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5', type=str)
    parser.add_argument('--elmo_cache_dir', default=None, type=str,
                        help='If set, then context-insensitive word embeddings will be cached ' + \
                             '(identified by a hash of the vocabulary).')

    # Training.
    parser.add_argument('--batch_size', default=10, type=int)
    parser.add_argument('--length_to_size', default=None, type=str,
                        help='Easily specify a mapping of length to batch_size.' + \
                             'For instance, 10:32,20:16 means that all batches' + \
                             'of length 10-19 will have batch size 32, 20 or greater' + \
                             'will have batch size 16, and less than 10 will have batch size' + \
                             'equal to the batch_size arg. Only applies to training.')
    parser.add_argument('--train_dataset_size', default=None, type=int)
    parser.add_argument('--validation_dataset_size', default=None, type=int)
    parser.add_argument('--validation_batch_size', default=None, type=int)
    parser.add_argument('--max_epoch', default=5, type=int)
    parser.add_argument('--max_step', default=None, type=int)
    parser.add_argument('--finetune', action='store_true')
    parser.add_argument('--finetune_after', default=0, type=int)

    # Parsing.
    parser.add_argument('--postprocess', action='store_true')
    parser.add_argument('--retain_file_order', action='store_true') # If true, then outputs are written in same order as read from file.

    # Optimization.
    parser.add_argument('--lr', default=4e-3, type=float)

    return parser


def parse_args(parser):
    options, other_args = parser.parse_known_args()

    # Set default flag values (data).
    options.train_data_type = options.data_type if options.train_data_type is None else options.train_data_type
    options.validation_data_type = options.data_type if options.validation_data_type is None else options.validation_data_type
    options.validation_batch_size = options.batch_size if options.validation_batch_size is None else options.validation_batch_size

    # Set default flag values (config).
    if not options.git_branch_name:
        options.git_branch_name = os.popen(
            'git rev-parse --abbrev-ref HEAD').read().strip()

    if not options.git_sha:
        options.git_sha = os.popen('git rev-parse HEAD').read().strip()

    if not options.git_dirty:
        options.git_dirty = os.popen("git diff --quiet && echo 'clean' || echo 'dirty'").read().strip()

    if not options.uuid:
        options.uuid = str(uuid.uuid4())

    if not options.experiment_name:
        options.experiment_name = '{}'.format(options.uuid[:8])

    if not options.experiment_path:
        options.experiment_path = os.path.join(options.default_experiment_directory, options.experiment_name)

    if options.length_to_size is not None:
        parts = [x.split(':') for x in options.length_to_size.split(',')]
        options.length_to_size = {int(x[0]): int(x[1]) for x in parts}

    options.lowercase = not options.uppercase

    for k, v in options.__dict__.items():
        if type(v) == str and v.startswith('~'):
            options.__dict__[k] = os.path.expanduser(v)

    # Load model settings from a flags file.
    if options.model_flags is not None:
        flags_to_use = []
        flags_to_use += ['arch']
        flags_to_use += ['compress']
        flags_to_use += ['emb']
        flags_to_use += ['hidden_dim']
        flags_to_use += ['normalize']
        flags_to_use += ['reconstruct_mode']

        options = init_with_flags_file(options, options.model_flags, flags_to_use)

    # Load any setting from a flags file.
    if options.flags is not None:
        options = init_with_flags_file(options, options.flags)

    return options


def configure(options):
    # Configure output paths for this experiment.
    configure_experiment(options.experiment_path, rank=options.local_rank)

    # Get logger.
    logger = get_logger()

    # Print flags.
    logger.info(stringify_flags(options))
    save_flags(options, options.experiment_path)


if __name__ == '__main__':
    parser = argument_parser()
    options = parse_args(parser)
    configure(options)

    run(options)
