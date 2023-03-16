import h5py
import os

from diora.net.experiment_logger import ExperimentLogger
from diora.logging.configuration import get_logger
from diora.scripts.train import argument_parser, parse_args, configure, seed_all, run_eval, get_train_and_validation, get_validation_dataset, get_validation_iterator, build_net

def run(options):

    seed_all(options.seed)
    if options.valid_hdf5 is not None:
        options.valid_hdf5 = h5py.File(options.valid_hdf5, 'r')

    logger = get_logger()
    experiment_logger = ExperimentLogger()

    if options.emb != 'none':
        train_dataset, validation_dataset = get_train_and_validation(options)
        embeddings = train_dataset['embeddings']
    else:
        validation_dataset = get_validation_dataset(options)
        embeddings = validation_dataset['embeddings']
    validation_iterator = get_validation_iterator(options, validation_dataset)
    

    logger.info('Initializing model.')
    trainer = build_net(options, embeddings, validation_iterator)
    logger.info('Model:')
    for name, p in trainer.net.named_parameters():
        logger.info('{} {}'.format(name, p.shape))

    if options.save_init:
        logger.info('Saving model (init).')
        trainer.save_model(os.path.join(options.experiment_path, 'model_init.pt'), save_emb=(options.emb == 'none'))

    run_eval(options, trainer, validation_iterator)

if __name__ == '__main__':
    parser = argument_parser()
    options = parse_args(parser)
    configure(options)

    run(options)
