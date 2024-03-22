import os
import sys

import numpy as np
import torch

sys.path.insert(0, "../")  # run under the current directory
from common.option import TestOptions
import model_new

def get_index_xy(L,d):
    diag = 2 * d + 1

    index_x = []
    index_y = []
    ref2index = np.zeros((L, diag), dtype=np.int_) - 1
    cnt = 0
    for x in range(L):
        for y in range(L):
            if abs(x-y) <= d:
                index_x.append(x)
                index_y.append(y)
                ref2index[x,y-x] = cnt
                cnt += 1

    index_x = np.array(index_x,dtype=np.int_)
    index_y = np.array(index_y, dtype=np.int_)

    index_compress = index_x * L + index_y
    return index_compress,ref2index

def get_index_xyz(L,d):
    diag = 2 * d + 1

    index_x = []
    index_y = []
    index_z = []
    ref2index = np.zeros((L, diag, diag), dtype=np.int_) - 1
    cnt = 0
    for x in range(L):
        for y in range(L):
            for z in range(L):
                if abs(x-y) <= d and abs(x-z)<=d:
                    index_x.append(x)
                    index_y.append(y)
                    index_z.append(z)
                    ref2index[x,y-x,z-x] = cnt
                    cnt += 1

    index_x = np.array(index_x,dtype=np.int_)
    index_y = np.array(index_y, dtype=np.int_)
    index_z = np.array(index_z, dtype=np.int_)

    index_compress = index_x * L * L + index_y * L + index_z
    return index_compress,ref2index

def get_index_xyzt(L,d):
    diag = 2 * d + 1

    index_x = []
    index_y = []
    index_z = []
    index_t = []
    ref2index = np.zeros((L, diag, diag, diag), dtype=np.int_) - 1
    cnt = 0
    for x in range(L):
        for y in range(L):
            for z in range(L):
                for t in range(L):
                    if abs(x-y) <= d and abs(x-z)<=d and abs(x-t)<=d:
                        index_x.append(x)
                        index_y.append(y)
                        index_z.append(z)
                        index_t.append(t)
                        ref2index[x,y-x,z-x,t-x] = cnt
                        cnt += 1

    index_x = np.array(index_x,dtype=np.int_)
    index_y = np.array(index_y, dtype=np.int_)
    index_z = np.array(index_z, dtype=np.int_)
    index_t = np.array(index_t, dtype=np.int_)

    index_compress = index_x * L * L * L + index_y * L * L + index_z * L + index_t
    return index_compress,ref2index


def compress_from_lut_larger_interval(lut,L,interval):
    B, C = lut.shape
    lut = lut.reshape(L, L, L, L, C)

    if interval==5:
        k = 2
    elif interval==6:
        k = 4
    else:
        k = 8

    compressed_lut = lut[::k, ::k, ::k, ::k, :].reshape(-1, C)
    return compressed_lut

def trans_lut2compress_spf():
    # exp_dir = '../models/SR/ablation_xyzt_2'
    # compression_type = 'xyz'
    # d = 11
    # interval = 6
    # save_dir = '../models/SR/ablation_xyzt_2/spflut_{}{}i{}'.format(compression_type,d,interval)
    # modes = ['s', 'd', 'y']
    # L = 17
    exp_dir = '../models_rebuttal/SR/spf_lut_test'
    compression_type = 'xyzt'
    d = 1
    interval = 5
    save_dir = '../models_rebuttal/SR/rebuttal/spflut_{}{}i{}'.format(compression_type,d,interval)
    modes = ['s', 'd', 'y']
    L = 17

    if compression_type =='xy':
        index_compress, ref2index = get_index_xy(L, d)
    elif compression_type == 'xyz':
        index_compress, ref2index = get_index_xyz(L, d)
    else:
        index_compress,ref2index = get_index_xyzt(L,d)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    np.save(os.path.join(save_dir, 'ref2index_L{}_d{}.npy'.format(L, d)), ref2index)

    for s in range(6):
        if s+1==5:
            lut_path = os.path.join(exp_dir, 'weight_s{}_channel.npy'.format(s + 1))
            lut = np.load(lut_path)
            save_path = os.path.join(save_dir, 'weight_s{}_channel.npy'.format(s + 1))
            np.save(save_path, lut)
            print("Resulting LUT size: ", lut.shape, "Saved to", save_path)
            continue
        for mode in modes:
            if s+1==6:
                for c in range(4):
                    lut_path = os.path.join(exp_dir, 'weight_s{}c{}_{}.npy'.format(s + 1, c, mode))
                    lut = np.load(lut_path)
                    _,C = lut.shape
                    # print('weight_s{}c{}_{}.npy'.format(s + 1, c, mode),lut.shape)

                    if compression_type == 'xy':
                        lut_c1 = lut.reshape((L*L,L,L,-1))[index_compress,:,:, :].reshape((-1,C))
                    elif compression_type == 'xyz':
                        lut_c1 = lut.reshape((L * L * L, L, -1))[index_compress, :, :].reshape((-1, C))
                    else:
                        lut_c1 = lut[index_compress,:]

                    lut_c2 = compress_from_lut_larger_interval(lut,L,interval)

                    save_path = os.path.join(save_dir, 'weight_s{}c{}_{}_compress1.npy'.format(s + 1, c, mode))
                    np.save(save_path, lut_c1)
                    print("Resulting LUT size: ", lut_c1.shape, "Saved to", save_path)

                    save_path = os.path.join(save_dir, 'weight_s{}c{}_{}_compress2.npy'.format(s + 1, c, mode))
                    np.save(save_path, lut_c2)
                    print("Resulting LUT size: ", lut_c2.shape, "Saved to", save_path)
            else:
                lut_path = os.path.join(exp_dir, 'weight_s{}c0_{}.npy'.format(s + 1, mode))
                lut = np.load(lut_path)
                _, C = lut.shape
                # print('weight_s{}c0_{}.npy'.format(s + 1, mode), lut.shape)

                if compression_type == 'xy':
                    lut_c1 = lut.reshape((L * L, L, L, -1))[index_compress, :, :, :].reshape((-1, C))
                elif compression_type == 'xyz':
                    lut_c1 = lut.reshape((L * L * L, L, -1))[index_compress, :, :].reshape((-1, C))
                else:
                    lut_c1 = lut[index_compress, :]
                lut_c2 = compress_from_lut_larger_interval(lut,L,interval)

                save_path = os.path.join(save_dir, 'weight_s{}c0_{}_compress1.npy'.format(s + 1, mode))
                np.save(save_path, lut_c1)
                print("Resulting LUT size: ", lut_c1.shape, "Saved to", save_path)

                save_path = os.path.join(save_dir, 'weight_s{}c0_{}_compress2.npy'.format(s + 1, mode))
                np.save(save_path, lut_c2)
                print("Resulting LUT size: ", lut_c2.shape, "Saved to", save_path)

def trans_lut2compress_spf_v2():
    # exp_dir = '../models/SR/ablation_xyzt_2'
    # compression_type = 'xyz'
    # d = 11
    # interval = 6
    # save_dir = '../models/SR/ablation_xyzt_2/spflut_{}{}i{}'.format(compression_type,d,interval)
    # modes = ['s', 'd', 'y']
    # L = 17
    exp_dir = '../models_rebuttal/SR/spf_lut_test'
    compression_type = 'xyzt'
    d = 2
    interval = 5
    save_dir = '../models_rebuttal/SR/rebuttal/spflut_{}{}i{}_partial_channel'.format(compression_type,d,interval)
    modes = ['s', 'd', 'y']
    L = 17

    if compression_type =='xy':
        index_compress, ref2index = get_index_xy(L, d)
    elif compression_type == 'xyz':
        index_compress, ref2index = get_index_xyz(L, d)
    else:
        index_compress,ref2index = get_index_xyzt(L,d)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    np.save(os.path.join(save_dir, 'ref2index_L{}_d{}.npy'.format(L, d)), ref2index)

    for s in range(6):
        if s+1==5:
            lut_path = os.path.join(exp_dir, 'weight_s{}_channel.npy'.format(s + 1))
            lut = np.load(lut_path)

            lut = lut.reshape((-1,4))
            lut = compress_from_lut_larger_interval(lut,L,5)
            lut = lut.reshape((-1,))

            save_path = os.path.join(save_dir, 'weight_s{}_channel.npy'.format(s + 1))
            np.save(save_path, lut)
            print("Resulting LUT size: ", lut.shape, "Saved to", save_path)
            continue
        for mode in modes:
            if s+1==6:
                for c in range(4):
                    lut_path = os.path.join(exp_dir, 'weight_s{}c{}_{}.npy'.format(s + 1, c, mode))
                    lut = np.load(lut_path)
                    _,C = lut.shape
                    # print('weight_s{}c{}_{}.npy'.format(s + 1, c, mode),lut.shape)

                    if compression_type == 'xy':
                        lut_c1 = lut.reshape((L*L,L,L,-1))[index_compress,:,:, :].reshape((-1,C))
                    elif compression_type == 'xyz':
                        lut_c1 = lut.reshape((L * L * L, L, -1))[index_compress, :, :].reshape((-1, C))
                    else:
                        lut_c1 = lut[index_compress,:]

                    if (mode in ['d', 'y'] and c ==3) or (mode in ['d', 'y'] and c ==1) or (mode in ['d', 'y'] and c ==0):
                        lut_c2 = compress_from_lut_larger_interval(lut,L,6)
                    else:
                        lut_c2 = compress_from_lut_larger_interval(lut,L,interval)

                    save_path = os.path.join(save_dir, 'weight_s{}c{}_{}_compress1.npy'.format(s + 1, c, mode))
                    np.save(save_path, lut_c1)
                    print("Resulting LUT size: ", lut_c1.shape, "Saved to", save_path)

                    save_path = os.path.join(save_dir, 'weight_s{}c{}_{}_compress2.npy'.format(s + 1, c, mode))
                    np.save(save_path, lut_c2)
                    print("Resulting LUT size: ", lut_c2.shape, "Saved to", save_path)
            else:
                lut_path = os.path.join(exp_dir, 'weight_s{}c0_{}.npy'.format(s + 1, mode))
                lut = np.load(lut_path)
                _, C = lut.shape
                # print('weight_s{}c0_{}.npy'.format(s + 1, mode), lut.shape)

                if compression_type == 'xy':
                    lut_c1 = lut.reshape((L * L, L, L, -1))[index_compress, :, :, :].reshape((-1, C))
                elif compression_type == 'xyz':
                    lut_c1 = lut.reshape((L * L * L, L, -1))[index_compress, :, :].reshape((-1, C))
                else:
                    lut_c1 = lut[index_compress, :]
                lut_c2 = compress_from_lut_larger_interval(lut,L,interval)

                save_path = os.path.join(save_dir, 'weight_s{}c0_{}_compress1.npy'.format(s + 1, mode))
                np.save(save_path, lut_c1)
                print("Resulting LUT size: ", lut_c1.shape, "Saved to", save_path)

                save_path = os.path.join(save_dir, 'weight_s{}c0_{}_compress2.npy'.format(s + 1, mode))
                np.save(save_path, lut_c2)
                print("Resulting LUT size: ", lut_c2.shape, "Saved to", save_path)

if __name__ == '__main__':
    trans_lut2compress_spf_v2()

