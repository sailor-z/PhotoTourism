import numpy as np
import argparse
import os
import glob
from tqdm import tqdm, trange
import cv2
import h5py
import pickle
import torchvision.transforms as transforms
import torch.nn as nn
import torch
import sys
sys.path.append('../third_party/')
from colmap.scripts.python.read_write_model import read_model, qvec2rotmat
from colmap.scripts.python.read_write_dense import read_array
from utils import saveh5, loadh5
def str2bool(v):
    return v.lower() in ("true", "1")
# Parse command line arguments.
parser = argparse.ArgumentParser(description='extract sift.')
parser.add_argument('--input_path', type=str, default='/cvlabdata2/home/chzhao/data/PhotoTourism/',
  help='Image directory or movie file or "camera" (for webcam).')
parser.add_argument('--img_glob', type=str, default='*/images/*.jpg',
  help='Glob match if directory of images is specified (default: \'*/images/*.jpg\').')
parser.add_argument('--num_kp', type=int, default='2000',
  help='keypoint number, default:2000')
parser.add_argument('--suffix', type=str, default='sift-2000',
  help='suffix of filename, default:sift-2000')
parser.add_argument('--thr', type=float, default=10.0,
  help='The threshold of inlier reprojection error')

class ExtractSIFT(object):
    def __init__(self, num_kp, contrastThreshold=1e-5):
        self.sift = cv2.xfeatures2d.SIFT_create(nfeatures=num_kp, contrastThreshold=contrastThreshold)

    def run(self, img_path):
        img = cv2.imread(img_path)
        cv_kp, desc = self.sift.detectAndCompute(img, None)

        kp = np.array([[_kp.pt[0], _kp.pt[1], _kp.size, _kp.angle] for _kp in cv_kp]) # N*4

        return kp, desc

class ExtractORB(object):
    def __init__(self, num_kp):
        self.orb = cv2.ORB_create(nfeatures=num_kp)
    def run(self, img_path):
        img = cv2.imread(img_path, 0)
        kp = self.orb.detect(img, None)
        kp, desc = self.orb.compute(img, kp)

        kp = np.array([[_kp.pt[0], _kp.pt[1], _kp.size, _kp.angle] for _kp in kp]) # N*4

        return kp, desc

def visualization(img1, img2, pt1, pt2):
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    vis = np.zeros((h1 + h2, max(w1, w2), 3), np.uint8)
    vis[:h1, :w1] = img1

    vis[h1:h1 + h2, :w2] = img2

    green = (0, 255, 0)
    red = (0, 0, 255)
    white = (255, 255, 255)
    thickness = 1
    num = 0

    for i in range(pt1.shape[0]):
        x1 = int(pt1[i, 0])
        y1 = int(pt1[i, 1])
        x2 = int(pt2[i, 0])
        y2 = int(pt2[i, 1] + h1)
        cv2.line(vis, (x1, y1), (x2, y2), green, int(thickness))

    return vis

def write_feature(pts, desc, filename):
    with h5py.File(filename, "w") as ifp:
        ifp.create_dataset('keypoints', pts.shape, dtype=np.float32)
        ifp.create_dataset('descriptors', desc.shape, dtype=np.float32)
        ifp["keypoints"][:] = pts
        ifp["descriptors"][:] = desc
    ifp.close()

def norm_kp(K, kp):
    # New kp
    kp = (kp - np.array([[K[0, 2], K[1, 2]]])) / np.asarray([[K[0, 0], K[1, 1]]])
    return kp

def np_skew_symmetric(v):

    zero = np.zeros_like(v[:, 0])

    M = np.stack([
        zero, -v[:, 2], v[:, 1],
        v[:, 2], zero, -v[:, 0],
        -v[:, 1], v[:, 0], zero,
    ], axis=1)

    return M

def get_episym(x1, x2, dR, dt):

    num_pts = len(x1)

    # Make homogeneous coordinates
    x1 = np.concatenate([
        x1, np.ones((num_pts, 1))
    ], axis=-1).reshape(-1, 3, 1)
    x2 = np.concatenate([
        x2, np.ones((num_pts, 1))
    ], axis=-1).reshape(-1, 3, 1)

    # Compute Fundamental matrix
    dR = dR.reshape(1, 3, 3)
    dt = dt.reshape(1, 3)
    F = np.repeat(np.matmul(
        np.reshape(np_skew_symmetric(dt), (-1, 3, 3)),
        dR
    ).reshape(-1, 3, 3), num_pts, axis=0)

    x2Fx1 = np.matmul(x2.transpose(0, 2, 1), np.matmul(F, x1)).flatten()
    Fx1 = np.matmul(F, x1).reshape(-1, 3)
    Ftx2 = np.matmul(F.transpose(0, 2, 1), x2).reshape(-1, 3)

    ys = x2Fx1**2 * (
        1.0 / (Fx1[..., 0]**2 + Fx1[..., 1]**2) +
        1.0 / (Ftx2[..., 0]**2 + Ftx2[..., 1]**2))

    return ys.flatten()

def extract_features(config, seqs, detector):
    for seq in seqs:
        print("Extracting features for {}".format(seq))

        search = os.path.join(config.input_path, seq, config.img_glob)
        listing = glob.glob(search)

        for img_path in tqdm(listing):
            save_path = img_path + '.' + config.suffix + '.hdf5'
            if os.path.exists(save_path):
                continue
            kp, desc = detector.run(img_path)
            write_feature(kp, desc, save_path)

def get_image(src_path, images, cameras, idx):
    im = cv2.imread(src_path + '/dense/images/' + images[idx].name)
    depth = read_array(src_path + '/dense/stereo/depth_maps/' + images[idx].name + '.photometric.bin')
    min_depth, max_depth = np.percentile(depth, [5, 95])
    depth[depth < min_depth] = min_depth
    depth[depth > max_depth] = max_depth

    # reformat data
    q = images[idx].qvec
    R = qvec2rotmat(q)
    T = images[idx].tvec
    p = images[idx].xys
    pars = cameras[idx].params
    K = np.array([[pars[0], 0, pars[2]], [0, pars[1], pars[3]], [0, 0, 1]])
    pids = images[idx].point3D_ids
    v = pids >= 0
#    print('Number of (valid) points: {}'.format((pids > -1).sum()))
#    print('Number of (total) points: {}'.format(v.size))

    # get also the clean depth maps
    base = '.'.join(images[idx].name.split('.')[:-1])
    with h5py.File(src_path + '/dense/stereo/depth_maps_clean_300_th_0.10/' + base + '.h5', 'r') as f:
        depth_clean = f['depth'].value

    return {
        'image': im,
        'depth_raw': depth,
        'depth': depth_clean,
        'K': K,
        'q': q,
        'R': R,
        'T': T,
        'xys': p,
        'ids': pids,
        'valid': v}

def computeNN(desc_ii, desc_jj):
    desc_ii, desc_jj = torch.from_numpy(desc_ii), torch.from_numpy(desc_jj)
    d1 = (desc_ii**2).sum(1)
    d2 = (desc_jj**2).sum(1)
    distmat = (d1.unsqueeze(1) + d2.unsqueeze(0) - 2*torch.matmul(desc_ii, desc_jj.transpose(0,1))).sqrt()
    distVals, nnIdx1 = torch.topk(distmat, k=2, dim=1, largest=False)
    nnIdx1 = nnIdx1[:,0]
    _, nnIdx2 = torch.topk(distmat, k=1, dim=0, largest=False)
    nnIdx2= nnIdx2.squeeze()
    mutual_nearest = (nnIdx2[nnIdx1] == torch.arange(0, nnIdx1.shape[0]).long()).numpy()
    ratio_test = (distVals[:,0] / distVals[:,1].clamp(min=1e-10)).numpy()
    idx_sort = [np.arange(nnIdx1.shape[0]), nnIdx1.numpy()]
    return idx_sort, ratio_test, mutual_nearest

def dump_nn(config, seqs):
    for seq in seqs:
        print("Computing match for {}".format(seq))
        pair_path = os.path.join(config.input_path, seq, 'pairs.pkl')
        with open(pair_path, 'rb') as f:
            pairs = pickle.load(f)
        f.close()

        cameras, images, points = read_model(path=os.path.join(config.input_path, seq, 'dense/sparse'), ext='.bin')
        nn_dist = {}
        dump_file = os.path.join(config.input_path, seq, 'nn.h5')
        for idx in trange(len(pairs['pairs'])):
            img1_path = images[pairs['pairs'][idx][0]].name
            img2_path = images[pairs['pairs'][idx][1]].name

            desc_ii = loadh5(os.path.join(config.input_path, seq, 'dense/images', img1_path+'.'+config.suffix+'.hdf5'))["descriptors"]
            desc_jj = loadh5(os.path.join(config.input_path, seq, 'dense/images', img2_path+'.'+config.suffix+'.hdf5'))["descriptors"]

            idx_sort, ratio_test, mutual_nearest = computeNN(desc_ii, desc_jj)

            # Dump to disk
            dump_dict = {}
            dump_dict["idx_sort"] = idx_sort
            dump_dict["ratio_test"] = ratio_test
            dump_dict["mutual_nearest"] = mutual_nearest
            dump_dict["pair"] = (pairs['pairs'][idx][0], pairs['pairs'][idx][1])
            nn_dist[str(idx)] = dump_dict
        saveh5(nn_dist, dump_file)

def dump_pair(config, seqs):
    for seq in seqs:
        print("Pair info processing for {}".format(seq))
        pair_path = os.path.join(config.input_path, seq, 'pairs.pkl')
        with open(pair_path, 'rb') as f:
            pairs = pickle.load(f)
        f.close()

        cameras, images, points = read_model(path=os.path.join(config.input_path, seq, 'dense/sparse'), ext='.bin')
        matching_info = loadh5(os.path.join(config.input_path, seq, 'nn.h5'))

        var_name = ['xs', 'ys', 'labels', 'K1s', 'K2s', 'Rs', 'Ts', 'ratios', 'mutuals']
        res_dict = {}
        xs, ys, labels, K1s, K2s, Rs, Ts, ratios, mutuals = [], [], [], [], [], [], [], [], []

        for idx in trange(len(pairs['pairs'])):
            idx1, idx2 = pairs['pairs'][idx]
        #    print('Showing pair: ({}, {})'.format(idx1, idx2))
            data1 = get_image(os.path.join(config.input_path, seq), images, cameras, idx1)
            data2 = get_image(os.path.join(config.input_path, seq), images, cameras, idx2)

            kpts1 = loadh5(os.path.join(config.input_path, seq, 'dense/images', images[idx1].name + '.' + config.suffix + '.hdf5'))["keypoints"]
            kpts2 = loadh5(os.path.join(config.input_path, seq, 'dense/images', images[idx2].name + '.' + config.suffix + '.hdf5'))["keypoints"]
            kpts1 = kpts1[:, :2]
            kpts2 = kpts2[:, :2]

            info = matching_info[str(idx)]
            kpts2 = kpts2[info["idx_sort"][1], :]

            K1, R1, T1 = data1["K"], data1["R"], data1["T"]
            K2, R2, T2 = data2["K"], data2["R"], data2["T"]

            u_xy1s = kpts1.T

            # Convert to homogeneous coordinates
            u_xy1s = np.concatenate([u_xy1s, np.ones([1, u_xy1s.shape[1]])], axis=0)

            # Get depth (on image 1) for each point
            u_xy1s_int = u_xy1s.astype(np.int32)
            z1 = data1['depth'][u_xy1s_int[1], u_xy1s_int[0]]

            # Eliminate points on occluded areas
            not_void = z1 > 0

            # Move to world coordinates
            n_xyz1s = np.dot(np.linalg.inv(K1), u_xy1s)
            n_xyz1s = n_xyz1s * z1 / n_xyz1s[2, :]
            xyz_w = np.dot(R1.T, n_xyz1s - T1[:,None])

            # Reproject into image 2
            n_xyz2s = np.dot(R2, xyz_w) + T2[:,None]
            u_xy2s = np.dot(K2, n_xyz2s)
            z2 = u_xy2s[2,:]
            u_xy2s = u_xy2s / z2

            reproj_err = np.sum((kpts2.T - u_xy2s[:2, :]) ** 2, axis=0) ** 0.5
            inlier_index = (reproj_err < config.thr) * not_void

        #    print(f'Valid points: {sum(inlier_index)}/{len(inlier_index)}')

            # Relative pose
            x1 = norm_kp(K1, kpts1)
            x2 = norm_kp(K2, kpts2)

            dR = np.dot(R2, R1.T)

            dt = T2 - np.dot(dR, T1)
            if np.sqrt(np.sum(dt**2)) <= 1e-5:
                return []
            dtnorm = np.sqrt(np.sum(dt**2))
            dt /= dtnorm

            geod_d = get_episym(x1, x2, dR, dt)
            geod_d = geod_d.reshape(-1,1)

            x12 = np.concatenate([x1, x2], axis=1).reshape(1,-1,4)

            ## Visualization
            '''
            kpts1 = kpts1[inlier_index, :]
            kpts2 = kpts2[inlier_index, :]
            viz = visualization(data1['image'], data2['image'], kpts1, kpts2)
            cv2.imwrite('./viz_%03d.jpg' % (idx), viz)
            exit()
            '''
            xs.append(x12), ys.append(geod_d), labels.append(inlier_index)
            K1s.append(K1.reshape(3, 3)), K2s.append(K2.reshape(3, 3))
            Rs.append(dR.reshape(3, 3)), Ts.append(dt.reshape(3, 1))
            ratios.append(info["ratio_test"]), mutuals.append(info["mutual_nearest"])

        res_dict['xs'] = xs
        res_dict['ys'] = ys
        res_dict['labels'] = labels
        res_dict['K1s'] = K1s
        res_dict['K2s'] = K2s
        res_dict['Rs'] = Rs
        res_dict['Ts'] = Ts
        res_dict['ratios'] = ratios
        res_dict['mutuals'] = mutuals

        for name in var_name:
            out_file_name = os.path.join(config.input_path, seq, name) + ".pkl"
            with open(out_file_name, "wb") as ofp:
                pickle.dump(res_dict[name], ofp)
            ofp.close()


if __name__ == "__main__":
    config = parser.parse_args()

    detector = ExtractSIFT(config.num_kp)
    #detector = ExtractORB(opt.num_kp)
    train_seqs = ['brandenburg_gate', 'buckingham_palace', 'colosseum_exterior','grand_place_brussels', \
              'hagia_sophia_interior', 'notre_dame_front_facade', 'palace_of_westminster', 'pantheon_exterior', \
              'taj_mahal', 'temple_nara_japan', 'trevi_fountain', 'westminster_abbey']
    extract_features(config, train_seqs, detector)
    dump_nn(config, train_seqs)
    dump_pair(config, train_seqs)
