## Dependencies

1. General libraries:
```bash
git clone https://github.com/roger-tseng/speech-parsing
cd speech-parsing
conda create -n parse python=3.8 -y
conda activate parse

conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch -y
pip install nltk h5py tqdm praatio benepar networkx
python -m spacy download en_core_web_md
python -m spacy download ko_core_news_md
```

```python
import benepar
benepar.download('benepar_en3')
benepar.download('benepar_ko2')
```

2. Install sox and S3PRL toolkit
```bash
# Install sox and S3PRL toolkit
# Linux
# apt-get install sox 
# MacOS
# brew install sox
git clone https://github.com/s3prl/s3prl
cd s3prl
pip install -e ./
cd ..
```

## Preprocessing SpokenCOCO

1. Download [SpokenCOCO](https://groups.csail.mit.edu/sls/downloads/placesaudio/) 

2. Download train/val/test split and word segmentation files we used [here](https://gntuedutw-my.sharepoint.com/:u:/g/personal/b07901163_g_ntu_edu_tw/ES7fjV4G7qBCpWQs6en-knsBC2VpWWRKi9qUcFJw18N5Fw). 
Place files under `data/SpokenCOCO`.
Initial data structure should look like:
```
data/SpokenCOCO
│
└───alignments
│   └─  train
│   └─  val
│   └─  test
│
└───asr_alignments
│   └─  ...
│
└───asr_st_alignments
│   └─  ...
│
└───fixed_alignments
│   └─  ...
│
└───train.txt
│
└───val.txt
│
└───test.txt
│
└───rest.txt
```

3. Assuming SpokenCOCO is downloaded at `$SPOKENCOCO_DIR`:
```bash
# Reorder SpokenCOCO by speaker and split
python reorder_by_split.py \
    --data_dir $SPOKENCOCO_DIR \
    --out_dir ./data/SpokenCOCO/

# Save XLSR-53 features in HDF5 format
upstream="xlsr_53"
for split in "train" "val" "test"
do 
    python pytorch/diora/speech/feature_extraction.py \
        --audio_root "./data/SpokenCOCO/split/$split" \
        --out_hdf5 "./data/SpokenCOCO/features/$split-$upstream.hdf5" \
        --upstream $upstream \
        --ext wav \
        --cuda \
        --upstream_layers 14;
done

# Write train manifest for topline
python pytorch/diora/speech/spokencoco/train_id.py \
    --coco_dir ./data/SpokenCOCO/split/ \
    --train_split ./data/train.txt \
    --alignment_dir ./data/SpokenCOCO/alignments/ \
    --out_path ./data/train_id_speech.txt

# Using Berkeley Neural Parser to induce ground truth parse trees for validation & testing set
python pytorch/diora/speech/spokencoco/parse_new.py \
    --type val \
    --in_fname ./data/val.txt \
    --wav_dir ./data/SpokenCOCO/split/ \
    --alignments_dir ./data/SpokenCOCO/alignments/val \
    --txt_fname ./data/val_id.txt \
    --parse_fname ./data/val_parse.txt

python pytorch/diora/speech/spokencoco/parse_new.py \
    --type test \
    --in_fname ./data/test.txt \
    --wav_dir ./data/SpokenCOCO/split/ \
    --alignments_dir ./data/SpokenCOCO/alignments/test \
    --txt_fname ./data/test_id.txt \
    --parse_fname ./data/test_parse.txt
```

4. Reformat with [xcfg](https://github.com/roger-tseng/xcfg/)
```python
# Prepare text
from binarize import save_labeled_tree
from argparse import Namespace
args = Namespace()
args.ifile = './data/SpokenCOCO/train_parse.txt'
args.ofile = './data/SpokenCOCO/train_parse_with_gt.txt'
save_labeled_tree(args)
args.ifile = './data/SpokenCOCO/val_parse.txt'
args.ofile = './data/SpokenCOCO/val_parse_with_gt.txt'
save_labeled_tree(args)
```

5. Prep for ASR & fixed-length segmentation settings
```bash
for setup in 'fixed' 'asr' 'asr_st' 
do
    mkdir -p ./data/${setup}
    echo "Writing train list for ${setup} setup"
    python pytorch/diora/speech/train_id.py \
        --coco_dir ./data/SpokenCOCO/split/ \
        --train_split ./data/train.txt \
        --alignment_dir ./data/SpokenCOCO/${setup}_alignments/ \
        --out_path ./data/${setup}/train_id_speech.txt
    echo "Creating 1-to-1 mapping between ${setup} imperfect and ground truth word segmentation via maximum duration overlap"
    for split in 'val' 'test'
    do
        python pytorch/diora/speech/matching.py \
        --asr_alignment_folder ./data/SpokenCOCO/${setup}_alignments \
        --gold_alignment_folder ./data/SpokenCOCO/alignments \
        --ofile ./data/${setup}/match.txt \
        --split $split \
        --labelled_ifile ./data/${split}_parse_with_gt.txt 
    done
done
```

6. Optionally remove unused files
```bash
# for split in "train" "val" "test"
# do 
#     rm -rf ${split}.txt
#     rm -rf ${split}_parse.txt
# done 
```

## Preprocessing Zeroth-Korean (WIP)
1. Download [Zeroth-Korean](https://www.openslr.org/40/)
<!-- ```
python pytorch/diora/speech/ko/write_fixed_txt.py
``` -->