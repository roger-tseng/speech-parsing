## Unsupervised Spoken Constituency Parsing

Official repo for our ICASSP 2023 paper: Cascading and Direct Approaches to Unsupervised Constituency Parsing on Spoken Sentences, which investigates two approaches to unsupervised constituency parsing on spoken sentences, given 1) raw speech & 2) unlabeled text. If you use this code for research, please cite our paper as follows:

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

## Usage (WIP)

Download alignment files we used here: [link](https://gntuedutw-my.sharepoint.com/:u:/g/personal/b07901163_g_ntu_edu_tw/EUW9ztUOQt5JlD1QxcDGyN0BJ2ULIW0IUGlNyuglAuFOAQ?e=t8Z5hO)

```bash
git clone https://github.com/roger-tseng/speech-parsing
git clone https://github.com/s3prl/s3prl
cd speech-parsing
conda create -n parse python=3.8 -y
conda activate parse
# PYTORCH for linux (w/ GPU and CUDA 11.3)
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch -y
pip install nltk h5py tqdm praatio benepar networkx

# Install S3PRL toolkit
# Install sox
cd ../s3prl
pip install -e ./

python -m spacy download en_core_web_md
# Install benepar
```

This repo is modified from [DIORA](https://github.com/iesl/diora). Original README for DIORA can be found in README_old.md)

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
