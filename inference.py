import os
import argparse
from path import Path
import glob
from itertools import groupby
import re
import numpy as np
from tqdm import tqdm
import torch
from torch.utils import data
import torch.nn as nn
from .models.CrackPropNet import CrackPropNet
from .data.dataLoader import VisDataset

parser = argparse.ArgumentParser(description='CrackPropNet inference',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('data',
                    metavar='DIR',
                    help='path to images folder (e.g., ./img). Images need to be in .png format.')
parser.add_argument('--output',
                    '-o',
                    metavar='DIR',
                    default=None,
                    help='path to output folder. If not set, will be created in data folder')
parser.add_argument('--pretrained',
                    metavar='PTH',
                    help='path to pre-trained model')


device = torch.device(
    "cuda") if torch.cuda.is_available() else torch.device("cpu")


@torch.no_grad()
def main():
    global args, save_path
    args = parser.parse_args()

    data_dir = Path(args.data)
    print("=> fetching img pairs in '{}'".format(args.data))

    if args.output is None:
        save_path = data_dir / 'flow'
    else:
        save_path = Path(args.output)
    print('=> will save outputs to {}'.format(save_path))
    save_path.makedirs_p()

    # Create model
    model = CrackPropNet.to(device)
    model.load_state_dict(torch.load(args.pretrained))
    model.eval()

    # Data loading
    img_dir = data_dir + '/*.png'
    file_list = glob.glob(img_dir)
    def keyf(text): return (re.findall(".+\-0", text) + [text])[0]
    grouped = [list(items)
               for gr, items in groupby(sorted(file_list), key=keyf)]
    final_list = np.empty((1, 2))
    for i in range(len(grouped)):
        group_list = grouped[i]
        ref_list = [group_list[0]] * len(group_list)
        two_col_list = np.array([ref_list, group_list]).T
        final_list = np.vstack((final_list, two_col_list[1:, :]))
    img_pairs = [tuple(x) for x in final_list[1:, :].tolist()]

    # Prediction
    dataset = VisDataset(img_pairs)
    dataloader = data.DataLoader(dataset, batch_size=1, shuffle=False,
                                 num_workers=2, drop_last=False)
    Sigmoid = nn.Sigmoid()

    for i, batch in enumerate(tqdm(dataloader)):
        img, filename = batch
        img = img.to(device)
        pred = ((Sigmoid(model(img)) > 0.5).float().cpu().numpy()) * 255
        pred = pred[0, 0, :, :].astype(np.uint8)
        output = os.path.join(save_path, filename[0])
        Image.fromarray(pred).save(output)


if __name__ == '__main__':
    main()
