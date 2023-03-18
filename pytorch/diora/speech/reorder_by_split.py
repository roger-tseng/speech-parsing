import os

def main(args):

    data = args.data_dir
    out_dir = args.out_dir

    for i in ['train', 'val', 'test']:
        f = open(os.path.join(out_dir, f'{i}.txt'))
        f_out = open(os.path.join(out_dir, '..', f'{i}.txt'), 'w')
        for line in f:
            fname, txt = line.split(' ', 1)
            basename = fname.rsplit('/', 1)[1]
            spk = basename.split('-', 1)[0]
            os.makedirs(os.path.join(out_dir, f'split/{i}/{spk}'), exist_ok=True)
            os.symlink(os.path.abspath(os.path.join(data, fname)), os.path.join(out_dir, f'split/{i}/{spk}/{basename}'))
            with open(os.path.join(out_dir, f'split/{i}/{spk}/{basename[:-4]}.txt'), 'w') as fw:
                print(txt, file=fw, end="")
            print(f'{i}/{spk}/{basename} {txt.strip()}', file=f_out)
        f.close()
        f_out.close()

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str)
    parser.add_argument("--out_dir", type=str)
    args = parser.parse_args()
    main(args)
