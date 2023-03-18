import argparse
import glob
from praatio import textgrid
from tqdm import tqdm
import os

parser = argparse.ArgumentParser()
parser.add_argument("--coco_dir", type=str)
parser.add_argument("--train_split", type=str)
parser.add_argument("--alignment_dir", type=str)
parser.add_argument("--out_path", type=str)
args = parser.parse_args()

train = glob.glob(os.path.join(args.alignment_dir, 'train/*/*'))
t = dict((i.rsplit('.',1)[0].rsplit('/',1)[1], i) for i in train)

with open(args.train_split) as f, open(args.out_path, 'w') as fw:
	for line in tqdm(f):
		fname = line.split(' ', 1)[0]
		basename = fname[:-4].rsplit('/', 1)[1]
		if basename not in t:
			continue
		else:
			tg = textgrid.openTextgrid(t[basename], includeEmptyIntervals=False)
			txt = ' '.join(word.label for word in tg.tierDict['words'].entryList)
			print(os.path.join(os.path.abspath(args.coco_dir), fname[:-4]), txt, file=fw)