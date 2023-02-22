from praatio import textgrid
import torchaudio

def boundaries_to_masks(boundaries, emb_len, downsample=320):
    import torch

    masks = []
    for boundary in boundaries:
        # Turn word/phone boundaries from raw wav frame-level boundaries to representation-level boundaries
        # For models with downsample rate of 320 (i.e. wav2vec series & HuBERT):
        # frame : | 400 frames | 320 frames | 320 frames | ... | 320 frames |
        # rep   : |     0th    |     1th    |     2th    | ... |     Nth    |
        boundary = ((boundary[0]-80)//downsample, (boundary[1]-80)//downsample)

        # Turn word/phone boundaries into masks to apply on repr. of sentence 
        mask = torch.zeros(emb_len)
        mask[boundary[0]:boundary[1]+1] = 1

        masks.append(mask)

    return torch.stack(masks)

def get_upstream(model_name, cuda=False):
    # inputs list of raw wav, outputs SSL representation, see https://github.com/s3prl/s3prl for more
    import s3prl.hub as hub
    upstream = getattr(hub, model_name)()
    upstream.eval() 
    if cuda:
        upstream = upstream.to('cuda')
    return upstream

def read_textgrid(fname, sr, textgrid_folder=None):
    _, _, _, file_name = fname.rsplit('/', 3)
    if '-' in file_name: # TEMP: hack for reading both SpokenCOCO and Zeroth-Korean
        spk = file_name.split('-', 1)[0]
    else:
        spk = file_name.split('_', 1)[0]
    fname = textgrid_folder+ '/' + spk + '/' + file_name
    tg = textgrid.openTextgrid(fname, includeEmptyIntervals=False)
    # phones = tg.tierDict['phones'].entryList
    words = tg.tierDict['words'].entryList

    # Turn start/end times into frame number
    # phones = [(int(interval.start*sr), int(interval.end*sr), interval.label) for interval in phones]
    words = [(int(interval.start*sr), int(interval.end*sr), interval.label) for interval in words]
    return [], words

def read_file(fname, ext="wav", textgrid_folder=None):
    # fname: full path to utterance without extension
    wav, sr = torchaudio.load(fname+'.'+ext)
    wav = wav.squeeze()
    phones, words = read_textgrid(fname+'.TextGrid', sr, textgrid_folder)
    txt = " ".join([i[2] for i in words]).lower()
    return wav, sr, txt, phones, words
