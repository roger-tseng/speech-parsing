## DIORA

This is the official repo for our NAACL 2019 paper Unsupervised Latent Tree Induction with Deep Inside-Outside Recursive Autoencoders (DIORA), which presents a fully-unsupervised method for discovering syntax. If you use this code for research, please cite our paper as follows:

```
@inproceedings{drozdov2019diora,
  title={Unsupervised Latent Tree Induction with Deep Inside-Outside Recursive Autoencoders},
  author={Drozdov, Andrew and Verga, Pat and Yadav, Mohit and Iyyer, Mohit and McCallum, Andrew},
  booktitle={North American Association for Computational Linguistics},
  year={2019},
}
```

The paper is available on arXiv: https://arxiv.org/abs/1904.02142

For questions/concerns/bugs please contact adrozdov at cs.umass.edu.

### Recent Related Work

Follow up work by us:

- [Drozdov et al., 2019. Unsupervised Labeled Parsing with DIORA.](https://aclanthology.org/D19-1161/)
- [Drozdov et al., 2020. Unsupervised Parsing with S-DIORA.](https://aclanthology.org/2020.emnlp-main.392/)
- [Xu et al., 2021. Improved Latent Tree Induction with Distant Supervision via Span Constraints.](https://aclanthology.org/2021.emnlp-main.395/)

Selection of other work with DIORA:

- [Hong et al., 2020. DIORA with All-span Objective.](https://aclanthology.org/2020.coling-main.322/)
- [Anonymous, 2021. Unsupervised Vision-Language Grammar Induction with Shared Structure Modeling (CLIORA).](https://openreview.net/forum?id=N0n_QyQ5lBF)

## Quick Start

Clone repository.

```
git clone git@github.com:iesl/diora.git
cd diora
```

Download the pre-trained model.

```
wget http://diora-naacl-2019.s3.amazonaws.com/diora-checkpoints.zip
unzip diora-checkpoints.zip
```

(Optional) Download training data: To reproduce experiments from our NAACL submission, concatenate the data from [SNLI](https://nlp.stanford.edu/projects/snli/) and [MultiNLI](https://www.nyu.edu/projects/bowman/multinli/).

```
cat ./snli_1.0/snli_1.0_train.jsonl ./multinli_1.0/multinli_1.0_train.jsonl > ./data/allnli.jsonl
```

Running DIORA.

```
# Install dependencies (using conda).
conda create -n diora-latest python=3.8
source activate diora-latest

## PYTORCH for mac
conda install pytorch=1.10.1 torchvision=0.11.2 torchaudio=0.10.1 -c pytorch

## PYTORCH for linux (w/ GPU and CUDA 10.2)
conda install pytorch=1.10.1 torchvision=0.11.2 torchaudio=0.10.1 cudatoolkit=10.2 -c pytorch

pip install nltk
pip install h5py
pip install tqdm

# Example of running various commands.

export PYTHONPATH=$(pwd)/pytorch:$PYTHONPATH

## Add the --cuda flag if you have GPU access.

## Parse some text.
python pytorch/diora/scripts/parse.py \
    --batch_size 10 \
    --data_type txt_id \
    --elmo_cache_dir ./cache \
    --load_model_path ./diora-checkpoints/mlp-softmax/model.pt \
    --model_flags ./diora-checkpoints/mlp-softmax/flags.json \
    --validation_path ./pytorch/sample.txt \
    --validation_filter_length 10

## Extract vectors using latent trees,
python pytorch/diora/scripts/phrase_embed_simple.py --parse_mode latent \
    --batch_size 10 \
    --data_type txt_id \
    --elmo_cache_dir ./cache \
    --load_model_path ./diora-checkpoints/mlp-softmax/model.pt \
    --model_flags ./diora-checkpoints/mlp-softmax/flags.json \
    --validation_path ./pytorch/sample.txt \
    --validation_filter_length 10

## or specify the trees to use.
python pytorch/diora/scripts/phrase_embed_simple.py --parse_mode given \
    --batch_size 10 \
    --data_type jsonl \
    --elmo_cache_dir ./cache \
    --load_model_path ./diora-checkpoints/mlp-softmax/model.pt \
    --model_flags ./diora-checkpoints/mlp-softmax/flags.json \
    --validation_path ./pytorch/sample.jsonl \
    --validation_filter_length 10

## Train from scratch.
python -m torch.distributed.launch --nproc_per_node=4 pytorch/diora/scripts/train.py \
    --arch mlp-shared \
    --batch_size 32 \
    --data_type nli \
    --elmo_cache_dir ./cache \
    --emb elmo \
    --hidden_dim 400 \
    --k_neg 100 \
    --log_every_batch 100 \
    --lr 2e-3 \
    --normalize unit \
    --reconstruct_mode softmax \
    --save_after 1000 \
    --train_filter_length 20 \
    --train_path ./data/allnli.jsonl \
    --max_step 300000 \
    --cuda --multigpu
```

### Evaluation

First parse the data, then run evalb from our helper script.

```
# Parse the data.
python pytorch/diora/scripts/parse.py \
    --retain_file_order \
    --batch_size 10 \
    --data_type ptb \
    --elmo_cache_dir ./cache \
    --load_model_path ./diora-checkpoints/mlp-softmax/model.pt \
    --model_flags ./diora-checkpoints/mlp-softmax/flags.json \
    --experiment_path ./log/eval-ptb \
    --validation_path ./data/ptb/ptb-test.txt \
    --validation_filter_length -1

# (optional) Build EVALB if you haven't already.
(cd EVALB && make)

# Run evaluation.
python pytorch/diora/scripts/evalb.py \
    --evalb ./EVALB \
    --evalb_config ./EVALB/diora.prm \
    --out ./log/eval-ptb \
    --pred ./log/eval-ptb/parse.jsonl \
    --gold ./data/ptb/ptb-test.txt
```

Using the `mlp-softmax` checkpoint to parse the PTB test set should give the following output and results:

```
$ python pytorch/diora/scripts/evalb.py \
    --evalb ./EVALB \
    --evalb_config ./EVALB/diora.prm \
    --out ./log/eval-ptb \
    --pred ./log/eval-ptb/parse.jsonl \
    --gold ./data/ptb/ptb-test.txt

Running: ./EVALB/evalb -p ./EVALB/diora.prm ./log/eval-ptb/gold.txt ./log/eval-ptb/pred.txt > ./log/eval-ptb/evalb.out

Results are ready at: ./log/eval-ptb/evalb.out

==== PREVIEW OF RESULTS (./log/eval-ptb/evalb.out) ====

-- All --
Number of sentence        =   2416
Number of Error sentence  =      0
Number of Skip  sentence  =      0
Number of Valid sentence  =   2416
Bracketing Recall         =  57.78
Bracketing Precision      =  44.28
Bracketing FMeasure       =  50.14
Complete match            =   0.46
Average crossing          =   5.71
No crossing               =  10.10
2 or less crossing        =  29.26
Tagging accuracy          =   9.76

-- len<=40 --
Number of sentence        =   2338
Number of Error sentence  =      0
Number of Skip  sentence  =      0
Number of Valid sentence  =   2338
Bracketing Recall         =  57.96
Bracketing Precision      =  44.57
Bracketing FMeasure       =  50.39
Complete match            =   0.47
Average crossing          =   5.39
No crossing               =  10.44
2 or less crossing        =  30.24
Tagging accuracy          =   9.79
```

Notes:

- Set `--validation_filter_length -1` to read all of the data.

- Make sure to use `--retain_file_order` so that predictions line up with the reference file.

- Set `--data_type ptb`. The PTB data should have one sentence per line be in the following format:

```
(S (NP (DT The) (VBG leading) (NNS indicators)) (VP (VBP have) (VP (VBN prompted) (NP (DT some) (NNS forecasters)))))
```

- DIORA will not attempt to parse 1 or 2 word sentences, since there is only 1 possible output.

- Using the provided configuration, the EVALB evaluation will ignore part of speech and constituency labels, but does take into account unary branching.

- Our EVALB helper script automatically strips punctuation.

## Multi-GPU Training

Using `DistributedDataParallel`:

```
export CUDA_VISIBLE_DEVICES=0,1
export NGPUS=2
python -m torch.distributed.launch --nproc_per_node=$NGPUS pytorch/diora/scripts/train.py \
    --cuda \
    --multigpu \
    ... # other args
```

## Useful Command Line Arguments

*Data*

`--data_type` Specifies the format of the data. Choices = `nli`, `txt`, `txt_id`, `synthetic`. Can specify different types for trainining and validation using `--train_data_type` and `--validation_data_type`. The `synthetic` type does not require any input file.

For examples of the expected format, please refer to the following files:

- `nli` The standard JSONL format used by SNLI and MultiNLI. Although examples are sentence pairs, the model only uses one sentence at a time.
- `txt` A single space-delimited sentence per line.
- `txt_id` Same as `txt` except the first token is an example id.

`--train_path` and `validation_path` Specifies the path to the input data for training and validation.

`--train_filter_length` Only examples less than this value will used for training. To consider all examples, set this to 0. Similarly, can use `--validation_filter_length` for validation.

`--batch_size` Specifies the batch size. The batch size specifically for validation can be set using `--validation_batch_size`, otherwise it will default to `--batch_size`.

`--embeddings_path` The path to GloVe-style word embeddings.

`--emb` Set to `w2v` for GloVe, `elmo` for ELMo, and `both` for a concatenation of the two.

`--elmo_options_path` and `--elmo_weights_path` The paths to the options and weights for ELMo.

*Optimization and Model Configuration*

`--lr` The learning rate.

`--hidden_dim` The dimension associated with the TreeLSTM.

`--margin` The margin value used in the objective for reconstruction.

`--k_neg` The number of negative examples to sample.

`--freq_dist_power` The negative examples are chosen according to their frequency within the training corpus. Lower values of `--freq_dist_power` make this distribution more peaked.

`--normalize` When set to `unit`, the values of each cell will have their norm set to 1. Choices = `none`, `unit`.

`--reconstruct_mode` Specifies how to reconstruct the correct word. Choices = `margin`.

*Logging*

`--load_model_path` For evaluation, parsing, and fine-tuning you can use this parameter to specify a previous checkpoint to initialize your model.

`--experiment_path` Specifies a directory where log files and checkpoints will be saved.

`--log_every_batch` Every N gradient updates a summary will be printed to the log.

`--save_latest` Every N gradient updates, a checkpoint will be saved called `model_periodic.pt`.

`--save_distinct` Every N gradient updates, a checkpoint will be saved called `model.step_${N}.pt`.

`--save_after` Checkpoints will only be saved after N gradient updates have been applied.

`--save_init` Save the initialization of the model.

*CUDA*

`--cuda` Use the GPU if available.

`--multigpu` Use multiple GPUs if available.

*Other*

`--seed` Set the random seed.

`--num_workers` Number of processes to use for batch iterator.

`--retain_file_order` If true, then outputs are written in same order as read from file.

## Faster ELMo Usage

If you specify the `elmo_cache_dir`, then the context-insensitive ELMo vectors will be cached, making it much faster to load these vectors after the initial usage. They must be cached once per dataset (a dataset is identified as a hash of its vocabulary).

Example Usage:

```
python pytorch/diora/scripts/train.py \
    --emb elmo \
    --elmo_cache_dir ./cache \
    ... # other args
```

## Easy Argument Assignment

Every experiment generates a `flags.json` file under its `experiment_path`. This file is useful when loading a checkpoint, as it specifies important properties for model configuration such as number-of-layers or model-size.

Note: Only arguments that are related to the model configuration will be used in this scenario.

Example Usage:

```
# First, train your model.
python pytorch/diora/scripts/train.py \
    --experiment_path ./log/experiment-01 \
    ... # other args

# Later, load the model checkpoint, and specify the flags file.
python pytorch/diora/scripts/parse.py \
    --load_model_path ./log/experiment-01/model_periodic.pt \
    --model_flags ./log/experiment-01/flags.json \
    ... # other args
```

## Logging

Various logs, checkpoints, and useful files are saved to a "log" directory when running DIORA. By default, this directory will be at `/path/to/diora/pytorch/log/${uuid}`. For example, this might be the log directory: `~/code/diora/pytorch/3d10566e`. You can specify your own directory using the `--experiment_path` flag.

Some files stored in the log directory are:

```
- flags.json  # All the arguments the experiment was run with as a JSON file.
- model_periodic.pt  # The latest model checkpoint, saved every N batches.
- model.step_X.pt  # Another checkpoint is saved every X batches.
```

## License

Copyright 2018, University of Massachusetts Amherst

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
