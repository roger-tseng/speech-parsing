## Unsupervised Spoken Constituency Parsing

Official repo for our ICASSP 2023 paper: [Cascading and Direct Approaches to Unsupervised Constituency Parsing on Spoken Sentences](https://arxiv.org/abs/2303.08809), which investigates two approaches to unsupervised constituency parsing on spoken sentences, given 1) raw speech & 2) unlabeled text. 

<p align="center">
<img src="./pdf/paper.png" alt="Diagram of our proposed direct approach to unsupervised spoken constituency parsing, using only raw speech and unpaired text. Textual transcripts of the input sentence are only shown for illustrative purpose"
width="800px"></p>

If you use this code for research, please cite our paper as follows:
```
@inproceedings{tseng2023parsing,
  title={Cascading and Direct Approaches to Unsupervised Constituency Parsing on Spoken Sentences},
  author={Tseng, Yuan and Lai, Cheng-I and Lee, Hung-yi},,
  booktitle={ICASSP 2023 - 2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  year={2023},
}
```

The paper is available on arXiv: https://arxiv.org/abs/2303.08809

For questions/concerns/bugs please contact r11942082 at ntu.edu.tw

## Preprocessing

See `pytorch/diora/speech/README.md`

## Training & Evaluation (WIP)

1. (Topline) Training with ground truth segmentation
```
conda activate parse
export PYTHONPATH=$(pwd)/pytorch:$PYTHONPATH

for i in {1..5}
do
python pytorch/diora/scripts/train.py \
--arch mlp-shared \
--batch_size 32 \
--cuda \
--emb none \
--emb_dim 1024 \
--hidden_dim 400 \
--k_neg 100 \
--log_every_batch 100 \
--lr 5e-3 \
--modality speech \
--normalize unit \
--reconstruct_mode softmax \
--save_after 100 \
--save_distinct 100 \
--seed $i \
--train_filter_length 30 \
--train_textgrid_folder ./data/SpokenCOCO/alignments/train \
--train_path ./data/train_id_speech.txt \
--train_data_type txt_id \
--train_hdf5 "./data/SpokenCOCO/features/train-xlsr_53-14.hdf5" \
--valid_textgrid_folder ./data/SpokenCOCO/alignments/val \
--validation_path ./data/val_parse_with_gt.txt \
--validation_data_type coco \
--valid_hdf5 "./data/SpokenCOCO/features/val-xlsr_53-14.hdf5" \
--experiment_path "./log/xlsr_gold/run$i" \
--max_step 2000 \
--max_epoch 1; \
done
```

2. Training with fixed-length segments
```
conda activate parse
export PYTHONPATH=$(pwd)/pytorch:$PYTHONPATH

for i in {1..5}
do
python pytorch/diora/scripts/train.py \
--arch mlp-shared \
--batch_size 32 \
--cuda \
--emb none \
--emb_dim 1024 \
--hidden_dim 400 \
--k_neg 100 \
--log_every_batch 100 \
--lr 5e-3 \
--modality speech \
--normalize unit \
--reconstruct_mode softmax \
--save_after 100 \
--save_distinct 100 \
--seed $i \
--train_filter_length 30 \
--train_textgrid_folder ./data/SpokenCOCO/fixed_alignments/train \
--train_path ./data/fixed/train_id_speech.txt \
--train_data_type txt_id \
--train_hdf5 "./data/SpokenCOCO/features/train-xlsr_53-14.hdf5" \
--valid_textgrid_folder ./data/SpokenCOCO/fixed_alignments/val \
--validation_path ./data/fixed/match_val_duration.txt \
--validation_data_type coco_asr \
--valid_hdf5 "./data/SpokenCOCO/features/val-xlsr_53-14.hdf5" \
--experiment_path "./log/xlsr_fixed/run$i" \
--max_step 2000 \
--max_epoch 1; \
done
```

## Attribution

This repo is modified from [DIORA](https://github.com/iesl/diora). Original README for DIORA can be found in README_old.md)
