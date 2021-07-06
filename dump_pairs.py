import argparse
import sys
import os
import deepdish as dd
import numpy as np
from tqdm import tqdm
import pickle

sys.path.append('../')
from utils.io_helper import save_h5, load_h5

def str2bool(v):
    return v.lower() in ("true", "1")
# Parse command line arguments.
parser = argparse.ArgumentParser(description='extract sift.')
parser.add_argument('--raw_data_path', type=str, default='/cvlabdata2/home/chzhao/data/PhotoTourism/',
  help='raw data path. default:../raw_data/')
parser.add_argument('--dump_dir', type=str, default='/cvlabdata2/home/chzhao/data/PhotoTourism/data_dump/',
  help='data dump path. default:../data_dump')
parser.add_argument('--desc_name', type=str, default='sift-2000',
  help='prefix of desc filename, default:sift-2000')
parser.add_argument('--vis_th', type=int, default=0.3,
  help='visibility threshold')
parser.add_argument('--pair_num', type=int, default=100000,
  help='pair num. 1000 for each seq')

np.random.seed(1234)

def dump_pairs(config, seqs):
    for seq in seqs:
        if os.path.exists(os.path.join(config.raw_data_path, seq, 'pairs.pkl')):
            continue

        pair_path = os.path.join(config.raw_data_path, seq, 'dense', 'stereo', 'pairs-dilation-0.00-fixed2.h5')
        pair_info = dd.io.load(pair_path)
        filtered = []

        print("Pair generation for {}".format(seq))

        for idx, p in enumerate(tqdm(pair_info)):
            if pair_info[p][0] >= config.vis_th and pair_info[p][1] >= config.vis_th:
                idx1, idx2 = p
                filtered += [p]

        pairs = [filtered[i] for i in np.random.permutation(len(filtered))[:config.pair_num]]

        print("Raw len: {}, Selected len: {}".format(len(filtered), len(pairs)))
        with open(os.path.join(config.raw_data_path, seq, 'pairs.pkl'), 'wb') as f:
            pickle.dump({"pairs": pairs}, f)
        f.close()

if __name__ == "__main__":
    config = parser.parse_args()
    train_seqs = ['brandenburg_gate', 'buckingham_palace', 'colosseum_exterior','grand_place_brussels', \
                'hagia_sophia_interior', 'notre_dame_front_facade', 'palace_of_westminster', 'pantheon_exterior', \
                'taj_mahal', 'temple_nara_japan', 'trevi_fountain', 'westminster_abbey']  #'prague_old_town_square' pair_info does not exist

    dump_pairs(config, train_seqs)

    
