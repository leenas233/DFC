import os
import sys

import numpy as np
import torch

sys.path.insert(0, "../")  # run under the current directory
from common.option import TestOptions
import model_new


def get_input_tensor(opt):
    # 1D input
    base = torch.arange(0, 257, 2 ** opt.interval)  # 0-256
    base[-1] -= 1
    L = base.size(0)

    # 2D input
    # 256*256   0 0 0...    |1 1 1...     |...|255 255 255...
    first = base.cuda().unsqueeze(1).repeat(1, L).reshape(-1)
    # 256*256   0 1 2 .. 255|0 1 2 ... 255|...|0 1 2 ... 255
    second = base.cuda().repeat(L)
    onebytwo = torch.stack([first, second], 1)  # [256*256, 2]

    # 3D input
    # 256*256*256   0 x65536|1 x65536|...|255 x65536
    third = base.cuda().unsqueeze(1).repeat(1, L * L).reshape(-1)
    onebytwo = onebytwo.repeat(L, 1)
    onebythree = torch.cat(
        [third.unsqueeze(1), onebytwo], 1)  # [256*256*256, 3]

    # 4D input
    fourth = base.cuda().unsqueeze(1).repeat(1, L * L * L).reshape(
        -1)  # 256*256*256*256   0 x16777216|1 x16777216|...|255 x16777216
    onebythree = onebythree.repeat(L, 1)
    # [256*256*256*256, 4]
    onebyfourth = torch.cat([fourth.unsqueeze(1), onebythree], 1)

    # Rearange input: [N, 4] -> [N, C=1, H=2, W=2]
    input_tensor = onebyfourth.unsqueeze(1).unsqueeze(
        1).reshape(-1, 1, 2, 2).float() / 255.0
    return input_tensor

def get_input_tensor_1D():
    # 1D input
    base = torch.arange(0, 256).float()/255.0  # 0-255
    input_tensor = base.cuda().reshape((-1,1,1,1))

    return input_tensor

def get_input_tensor2(opt):
    # 1D input
    base = torch.arange(0, 257, 2 ** opt.interval)  # 0-256
    base[-1] -= 1
    L = base.size(0)

    base2 = torch.arange(0, 257, 2 ** (opt.interval - 1))  # 0-256
    base2[-1] -= 1
    L2 = base2.size(0)

    # 2D input
    # 256*256   0 0 0...    |1 1 1...     |...|255 255 255...
    first = base.cuda().unsqueeze(1).repeat(1, L).reshape(-1)
    # 256*256   0 1 2 .. 255|0 1 2 ... 255|...|0 1 2 ... 255
    second = base.cuda().repeat(L)
    onebytwo = torch.stack([first, second], 1)  # [256*256, 2]

    # 3D input
    # 256*256*256   0 x65536|1 x65536|...|255 x65536
    third = base2.cuda().unsqueeze(1).repeat(1, L * L).reshape(-1)
    onebytwo = onebytwo.repeat(L2, 1)
    onebythree = torch.cat(
        [third.unsqueeze(1), onebytwo], 1)  # [256*256*256, 3]

    # 4D input
    fourth = base2.cuda().unsqueeze(1).repeat(1, L2 * L * L).reshape(
        -1)  # 256*256*256*256   0 x16777216|1 x16777216|...|255 x16777216
    onebythree = onebythree.repeat(L2, 1)
    # [256*256*256*256, 4]
    onebyfourth = torch.cat([fourth.unsqueeze(1), onebythree], 1)

    # Rearange input: [N, 4] -> [N, C=1, H=2, W=2]
    input_tensor = onebyfourth.unsqueeze(1).unsqueeze(
        1).reshape(-1, 1, 2, 2).float() / 255.0
    return input_tensor


def get_mode_input_tensor(input_tensor, mode):
    if mode == "d":
        input_tensor_dil = torch.zeros(
            (input_tensor.shape[0], input_tensor.shape[1], 3, 3), dtype=input_tensor.dtype).to(input_tensor.device)
        input_tensor_dil[:, :, 0, 0] = input_tensor[:, :, 0, 0]
        input_tensor_dil[:, :, 0, 2] = input_tensor[:, :, 0, 1]
        input_tensor_dil[:, :, 2, 0] = input_tensor[:, :, 1, 0]
        input_tensor_dil[:, :, 2, 2] = input_tensor[:, :, 1, 1]
        input_tensor = input_tensor_dil
    elif mode == "y":
        input_tensor_dil = torch.zeros(
            (input_tensor.shape[0], input_tensor.shape[1], 3, 3), dtype=input_tensor.dtype).to(input_tensor.device)
        input_tensor_dil[:, :, 0, 0] = input_tensor[:, :, 0, 0]
        input_tensor_dil[:, :, 1, 1] = input_tensor[:, :, 0, 1]
        input_tensor_dil[:, :, 1, 2] = input_tensor[:, :, 1, 0]
        input_tensor_dil[:, :, 2, 1] = input_tensor[:, :, 1, 1]
        input_tensor = input_tensor_dil
    else:
        # more sampling modes can be implemented similarly
        raise ValueError("Mode {} not implemented.".format(mode))
    return input_tensor


def compress_lut(input_tensor):
    base = torch.arange(0, 257, 2 ** opt.interval)  # 0-256
    base[-1] -= 1
    L = base.size(0)
    d = 2
    diag = 2 * d + 1
    N = diag * L + (1 - diag ** 2) // 4

    input_tensor = input_tensor.reshape(L * L, L, L, 1, 2, 2)
    index_i = torch.zeros((N,)).type(torch.int64)
    index_j = torch.zeros((N,)).type(torch.int64)
    cnt = 0
    for i in range(L):
        for j in range(L):
            if abs(i - j) <= d:
                index_i[cnt] = i
                index_j[cnt] = j
                cnt += 1

    index_compress = index_i * L + index_j
    compressed_input_tensor = input_tensor[index_compress, ...].reshape(-1, 1, 2, 2)
    return compressed_input_tensor


def compress_lut_xyz(input_tensor):
    base = torch.arange(0, 257, 2 ** opt.interval)  # 0-256
    base[-1] -= 1
    L = base.size(0)
    d = 2
    diag = 2 * d + 1

    input_tensor = input_tensor.reshape(L * L * L, L, 1, 2, 2)
    ref_x = []
    ref_y = []
    ref_z = []
    cnt = 0
    ref2index = np.zeros((L, diag, diag), dtype=np.int_) - 1
    for x in range(L):
        for y in range(L):
            for z in range(L):
                if abs(x - y) <= d and abs(x - z) <= d:
                    ref_x.append(x)
                    ref_y.append(y)
                    ref_z.append(z)
                    ref2index[x, y - x, z - x] = cnt
                    cnt += 1
    np.save(os.path.join(opt.expDir, 'ref2index_L{}_d{}.npy'.format(L, d)),ref2index)
    ref_x = torch.Tensor(ref_x).type(torch.int64)
    ref_y = torch.Tensor(ref_y).type(torch.int64)
    ref_z = torch.Tensor(ref_z).type(torch.int64)

    index_compress = ref_x * L * L + ref_y * L + ref_z
    compressed_input_tensor = input_tensor[index_compress, ...].reshape(-1, 1, 2, 2)
    return compressed_input_tensor

def compress_lut_xyzt(input_tensor):
    base = torch.arange(0, 257, 2 ** opt.interval)  # 0-256
    base[-1] -= 1
    L = base.size(0)
    d = 14
    diag = 2 * d + 1

    input_tensor = input_tensor.reshape(L * L * L * L, 1, 2, 2)
    ref_x = []
    ref_y = []
    ref_z = []
    ref_t = []
    cnt = 0
    ref2index = np.zeros((L, diag, diag, diag), dtype=np.int_) - 1
    for x in range(L):
        for y in range(L):
            for z in range(L):
                for t in range(L):
                    if abs(x - y) <= d and abs(x - z) <= d and abs(x - t) <= d:
                        ref_x.append(x)
                        ref_y.append(y)
                        ref_z.append(z)
                        ref_t.append(t)
                        ref2index[x, y - x, z - x, t - x] = cnt
                        cnt += 1
    np.save(os.path.join(opt.expDir, 'ref2index_L{}_d{}.npy'.format(L, d)),ref2index)
    ref_x = torch.Tensor(ref_x).type(torch.int64)
    ref_y = torch.Tensor(ref_y).type(torch.int64)
    ref_z = torch.Tensor(ref_z).type(torch.int64)
    ref_t = torch.Tensor(ref_t).type(torch.int64)

    index_compress = ref_x * L * L * L + ref_y * L * L + ref_z * L + ref_t
    compressed_input_tensor = input_tensor[index_compress, ...].reshape(-1, 1, 2, 2)
    return compressed_input_tensor

def compress_lut2(input_tensor):
    base = torch.arange(0, 257, 2 ** opt.interval)  # 0-256
    base[-1] -= 1
    L = base.size(0)
    L2 = 2 ** (8 - opt.interval + 1) + 1
    d = 4
    diag = 2 * d + 1
    N = diag * L2 + (1 - diag ** 2) // 4

    input_tensor = input_tensor.reshape(L2 * L2, L, L, 1, 2, 2)
    index_i = torch.zeros((N,)).type(torch.int64)
    index_j = torch.zeros((N,)).type(torch.int64)
    cnt = 0
    for i in range(L2):
        for j in range(L2):
            if abs(i - j) <= d:
                index_i[cnt] = i
                index_j[cnt] = j
                cnt += 1

    index_compress = index_i * L2 + index_j
    compressed_input_tensor = input_tensor[index_compress, ...].reshape(-1, 1, 2, 2)
    return compressed_input_tensor


def compress_lut_larger_interval(input_tensor):
    base = torch.arange(0, 257, 2 ** opt.interval)  # 0-256
    base[-1] -= 1
    L = base.size(0)
    input_tensor = input_tensor.reshape(L, L, L, L, 1, 2, 2)

    compressed_input_tensor = input_tensor[::4, ::4, ::4, ::4, ...].reshape(-1, 1, 2, 2)
    return compressed_input_tensor


def save_lut(input_tensor, lut_path, mode, model_G):
    # Split input to not over GPU memory
    B = input_tensor.size(0) // 100
    outputs = []

    # Extract input-output pairs
    with torch.no_grad():
        model_G.eval()
        for b in range(100):
            if b == 99:
                batch_input = input_tensor[b * B:]
            else:
                batch_input = input_tensor[b * B:(b + 1) * B]

            batch_output = getattr(model_G, "s{}_{}".format(str(s + 1), mode))(batch_input)

            results = torch.round(torch.tanh(batch_output) * 127).cpu().data.numpy().astype(np.int8)
            outputs += [results]

    results = np.concatenate(outputs, 0)
    results = results.reshape(input_tensor.size(0), -1)
    np.save(lut_path, results)
    print("Resulting LUT size: ", results.shape, "Saved to", lut_path)


def compress_from_lut_xy(lut,d,save_dir):
    B, C = lut.shape

    L = 17
    diag = 2 * d + 1
    N = diag * L + (1 - diag ** 2) // 4

    lut = lut.reshape(L * L, L, L, -1)
    index_x = []
    index_y = []
    ref2index = np.zeros((L, diag), dtype=np.int_) - 1
    cnt = 0
    for x in range(L):
        for y in range(L):
            if abs(x - y) <= d:
                index_x.append(x)
                index_y.append(y)
                ref2index[x, y - x] = cnt
                cnt += 1
    np.save(os.path.join(save_dir, 'ref2index_L{}_d{}.npy'.format(L, d)), ref2index)
    index_x = np.array(index_x, dtype=np.int_)
    index_y = np.array(index_y, dtype=np.int_)

    index_compress = index_x * L + index_y
    compressed_lut = lut[index_compress, :, :, :].reshape(-1, C)
    return compressed_lut

def compress_from_lut_xyz(lut,d,save_dir):
    B, C = lut.shape

    L = 17
    diag = 2 * d + 1

    lut = lut.reshape(L * L * L, L, -1)
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
    np.save(os.path.join(save_dir, 'ref2index_L{}_d{}.npy'.format(L, d)), ref2index)
    index_x = np.array(index_x,dtype=np.int_)
    index_y = np.array(index_y, dtype=np.int_)
    index_z = np.array(index_z, dtype=np.int_)

    index_compress = index_x * L * L + index_y * L + index_z
    compressed_lut = lut[index_compress, :, :].reshape(-1, C)
    return compressed_lut

def compress_from_lut_xyzt(lut,d,save_dir):
    B, C = lut.shape

    L = 17
    diag = 2 * d + 1

    lut = lut.reshape(L * L * L * L, -1)
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
    np.save(os.path.join(save_dir, 'ref2index_L{}_d{}.npy'.format(L, d)), ref2index)
    index_x = np.array(index_x,dtype=np.int_)
    index_y = np.array(index_y, dtype=np.int_)
    index_z = np.array(index_z, dtype=np.int_)
    index_t = np.array(index_t, dtype=np.int_)

    index_compress = index_x * L * L * L + index_y * L * L + index_z * L + index_t
    compressed_lut = lut[index_compress, :].reshape(-1, C)
    return compressed_lut

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


def trans_lut2compress():
    # stages = 2
    # exp_dir = '../models/SR/ablation_interval/mulutx2'
    # compression_type = 'xyzt'
    # d = 10
    # interval = 5
    # save_dir = '../models/SR/ablation_interval/mulutx2_{}{}i{}'.format(compression_type, d, interval)
    # modes = ['s', 'd', 'y']
    # L = 17

    stages = 2
    exp_dir = '../models_rebuttal/SR/mulutx2_test'
    compression_type = 'xyzt'
    d = 5
    interval = 5
    save_dir = '../models_rebuttal/SR/rebuttal/mulutx2_{}{}i{}'.format(compression_type, d, interval)
    modes = ['s', 'd', 'y']
    L = 17

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for s in range(stages):
        for mode in modes:
            lut_path = os.path.join(exp_dir, 'LUT_ft_x4_4bit_int8_s{}_{}.npy'.format(s + 1, mode))
            lut = np.load(lut_path)

            if compression_type == 'xy':
                lut_c1 = compress_from_lut_xy(lut, d, save_dir)
            elif compression_type == 'xyz':
                lut_c1 = compress_from_lut_xyz(lut, d, save_dir)
            else:
                lut_c1 = compress_from_lut_xyzt(lut,d,save_dir)
            lut_c2 = compress_from_lut_larger_interval(lut,L,interval)

            save_path = os.path.join(save_dir, 'LUT_x4_4bit_int8_s{}_{}_compress1.npy'.format(s + 1, mode))
            np.save(save_path, lut_c1)
            print("Resulting LUT size: ", lut_c1.shape, "Saved to", save_path)

            save_path = os.path.join(save_dir, 'LUT_x4_4bit_int8_s{}_{}_compress2.npy'.format(s + 1, mode))
            np.save(save_path, lut_c2)
            print("Resulting LUT size: ", lut_c2.shape, "Saved to", save_path)

def trans_lut2compress_partial():
    # stages = 2
    # exp_dir = '../models/SR/ablation_interval/mulutx2'
    # compression_type = 'xyzt'
    # d = 10
    # interval = 5
    # save_dir = '../models/SR/ablation_interval/mulutx2_{}{}i{}'.format(compression_type, d, interval)
    # modes = ['s', 'd', 'y']
    # L = 17

    stages = 2
    exp_dir = '../models_rebuttal/SR/mulutx2_test'
    compression_type = 'xyzt'
    d = 6
    interval = 5
    save_dir = '../models_rebuttal/SR/rebuttal/mulutx2_{}{}i{}_partial'.format(compression_type, d, interval)
    modes = ['s', 'd', 'y']
    L = 17

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for s in range(stages):
        for mode in modes:
            lut_path = os.path.join(exp_dir, 'LUT_ft_x4_4bit_int8_s{}_{}.npy'.format(s + 1, mode))
            lut = np.load(lut_path)

            if compression_type == 'xy':
                lut_c1 = compress_from_lut_xy(lut, d, save_dir)
            elif compression_type == 'xyz':
                lut_c1 = compress_from_lut_xyz(lut, d, save_dir)
            else:
                lut_c1 = compress_from_lut_xyzt(lut,d,save_dir)
            
            if s==1 and mode in ['d', 'y']:
                lut_c2 = compress_from_lut_larger_interval(lut,L,6)
            else:
                lut_c2 = compress_from_lut_larger_interval(lut,L,interval)

            save_path = os.path.join(save_dir, 'LUT_x4_4bit_int8_s{}_{}_compress1.npy'.format(s + 1, mode))
            np.save(save_path, lut_c1)
            print("Resulting LUT size: ", lut_c1.shape, "Saved to", save_path)

            save_path = os.path.join(save_dir, 'LUT_x4_4bit_int8_s{}_{}_compress2.npy'.format(s + 1, mode))
            np.save(save_path, lut_c2)
            print("Resulting LUT size: ", lut_c2.shape, "Saved to", save_path)

def compress_MuLUT_IMDB():
    def save_MuLUT_IMDB(x, lut_path, module):
        # Split input to not over GPU memory
        B = x.size(0) // 100
        outputs = []

        # Extract input-output pairs
        with torch.no_grad():
            model_G.eval()
            for b in range(100):
                if b == 99:
                    batch_input = x[b * B:]
                else:
                    batch_input = x[b * B:(b + 1) * B]

                batch_output = module(batch_input)

                results = torch.round(torch.tanh(batch_output) * 127).cpu().data.numpy().astype(np.int8)
                outputs += [results]

        results = np.concatenate(outputs, 0)
        results = results.reshape(x.size(0), -1)
        np.save(lut_path, results)
        print("Resulting LUT size: ", results.shape, "Saved to", lut_path)


    # load model
    # opt = TestOptions().parse()

    modes = [i for i in opt.modes]
    stages = opt.stages

    model = getattr(model_new, 'MuLUT_IMDBV2')

    model_G = model(nf=opt.nf, scale=opt.scale, modes=modes, stages=stages).cuda()

    lm = torch.load(os.path.join(opt.expDir, 'Model_{:06d}.pth'.format(opt.loadIter)))
    model_G.load_state_dict(lm.state_dict(), strict=True)

    input_tensor = get_input_tensor(opt)
    for mode in modes:
        input_tensor_c1 = input_tensor.clone()
        # input_tensor_c1 = compress_lut_xyzt(input_tensor)
        # input_tensor_c2 = compress_lut_larger_interval(input_tensor)

        if mode != 's':
            # input_tensor_c1 = get_mode_input_tensor(input_tensor_c1, mode)
            # input_tensor_c2 = get_mode_input_tensor(input_tensor_c2, mode)
            input_tensor_c1 = get_mode_input_tensor(input_tensor_c1, mode)

        # conv1
        module = model_G.convblock1.module_dict['DepthwiseBlock{}_{}'.format(0, mode)]
        # lut_path = os.path.join(opt.expDir, 'LUT_x4_4bit_int8_s{}c0_{}_compress1.npy'.format(1, mode))
        lut_path = os.path.join(opt.expDir, 'LUT_x4_4bit_int8_s{}c0_{}.npy'.format(1, mode))
        save_MuLUT_IMDB(input_tensor_c1, lut_path, module)
        # lut_path = os.path.join(opt.expDir, 'LUT_x4_4bit_int8_s{}c0_{}_compress2.npy'.format(1, mode))
        # save_MuLUT_IMDB(input_tensor_c2, lut_path, module)

        # conv2
        module = model_G.convblock2.module_dict['DepthwiseBlock{}_{}'.format(0, mode)]
        # lut_path = os.path.join(opt.expDir, 'LUT_x4_4bit_int8_s{}c0_{}_compress1.npy'.format(2, mode))
        lut_path = os.path.join(opt.expDir, 'LUT_x4_4bit_int8_s{}c0_{}.npy'.format(2, mode))
        save_MuLUT_IMDB(input_tensor_c1, lut_path, module)
        # lut_path = os.path.join(opt.expDir, 'LUT_x4_4bit_int8_s{}c0_{}_compress2.npy'.format(2, mode))
        # save_MuLUT_IMDB(input_tensor_c2, lut_path, module)

        # conv3
        module = model_G.convblock3.module_dict['DepthwiseBlock{}_{}'.format(0, mode)]
        # lut_path = os.path.join(opt.expDir, 'LUT_x4_4bit_int8_s{}c0_{}_compress1.npy'.format(3, mode))
        lut_path = os.path.join(opt.expDir, 'LUT_x4_4bit_int8_s{}c0_{}.npy'.format(3, mode))
        save_MuLUT_IMDB(input_tensor_c1, lut_path, module)
        # lut_path = os.path.join(opt.expDir, 'LUT_x4_4bit_int8_s{}c0_{}_compress2.npy'.format(3, mode))
        # save_MuLUT_IMDB(input_tensor_c2, lut_path, module)

        # conv4
        module = model_G.convblock4.module_dict['DepthwiseBlock{}_{}'.format(0, mode)]
        # lut_path = os.path.join(opt.expDir, 'LUT_x4_4bit_int8_s{}c0_{}_compress1.npy'.format(4, mode))
        lut_path = os.path.join(opt.expDir, 'LUT_x4_4bit_int8_s{}c0_{}.npy'.format(4, mode))
        save_MuLUT_IMDB(input_tensor_c1, lut_path, module)
        # lut_path = os.path.join(opt.expDir, 'LUT_x4_4bit_int8_s{}c0_{}_compress2.npy'.format(4, mode))
        # save_MuLUT_IMDB(input_tensor_c2, lut_path, module)

        # conv6
        for c in range(4):
            module = model_G.upblock.module_dict['DepthwiseBlock{}_{}'.format(c, mode)]
            # lut_path = os.path.join(opt.expDir, 'LUT_x4_4bit_int8_s{}c{}_{}_compress1.npy'.format(6,c, mode))
            lut_path = os.path.join(opt.expDir, 'LUT_x4_4bit_int8_s{}c{}_{}.npy'.format(6, c, mode))
            save_MuLUT_IMDB(input_tensor_c1, lut_path, module)
            # lut_path = os.path.join(opt.expDir, 'LUT_x4_4bit_int8_s{}c{}_{}_compress2.npy'.format(6,c, mode))
            # save_MuLUT_IMDB(input_tensor_c2, lut_path, module)
    # conv5
    input_tensor = input_tensor.reshape((-1,4,1,1))
    module = model_G.ChannelConv
    lut_path = os.path.join(opt.expDir, 'LUT_x4_4bit_int8_s{}_channel.npy'.format(5))
    save_MuLUT_IMDB(input_tensor, lut_path, module)


def compress_RCLUT():
    def save_RCLUT(x, lut_path, module):
        # Split input to not over GPU memory
        B = x.size(0) // 100
        outputs = []

        # Extract input-output pairs
        with torch.no_grad():
            model_G.eval()
            for b in range(100):
                if b == 99:
                    batch_input = x[b * B:]
                else:
                    batch_input = x[b * B:(b + 1) * B]

                batch_output = module(batch_input,True)

                results = torch.round(torch.tanh(batch_output) * 127).cpu().data.numpy().astype(np.int8)
                outputs += [results]

        results = np.concatenate(outputs, 0)
        results = results.reshape(x.size(0), -1)
        np.save(lut_path, results)
        print("Resulting LUT size: ", results.shape, "Saved to", lut_path)

    def save_RCModule(x,lut_path,module,mlp_field):
        outputs = []
        with torch.no_grad():
            model_G.eval()
            for i in range(mlp_field * mlp_field):
                num = i + 1
                module1 = getattr(module, 'linear{}'.format(num))
                module2 = getattr(module, 'out{}'.format(num))

                x1 = module1(x[:, :, 0, 0]).unsqueeze(1) # (256,1,nf)

                x2 = module2(x1).unsqueeze(0).cpu().data.numpy() # (1, 256, 1, 1)
                outputs.append(x2)
        results = np.concatenate(outputs, 0)
        np.save(lut_path, results)
        print("Resulting LUT size: ", results.shape, "Saved to", lut_path)


    # load model
    # opt = TestOptions().parse()

    modes = [i for i in opt.modes]
    stages = opt.stages

    model = getattr(model_new, 'RC_SRNets')

    model_G = model(nf=opt.nf, scale=opt.scale, modes=modes, stages=stages).cuda()

    lm = torch.load(os.path.join(opt.expDir, 'Model_{:06d}.pth'.format(opt.loadIter)))
    model_G.load_state_dict(lm.state_dict(), strict=True)

    input_tensor = get_input_tensor(opt)
    input_tensor_1D = get_input_tensor_1D()
    mlp_field = {'s':5,'d':7,'y':3}
    for mode in modes:
        # input_tensor_c1 = input_tensor.clone()
        input_tensor_c1 = compress_lut_xyzt(input_tensor)
        input_tensor_c2 = compress_lut_larger_interval(input_tensor)

        # conv1
        module = model_G.convblock1.module_dict['DepthwiseBlock{}_{}'.format(0, mode)]
        lut_path = os.path.join(opt.expDir, 'LUT_x4_4bit_int8_s{}c0_{}_compress1.npy'.format(1, mode))
        # lut_path = os.path.join(opt.expDir, 'LUT_x4_4bit_int8_s{}c0_{}.npy'.format(1, mode))
        save_RCLUT(input_tensor_c1, lut_path, module)
        lut_path = os.path.join(opt.expDir, 'LUT_x4_4bit_int8_s{}c0_{}_compress2.npy'.format(1, mode))
        save_RCLUT(input_tensor_c2, lut_path, module)

        mlp_num = mlp_field[mode]
        lut_path = os.path.join(opt.expDir, 'LUT_RC{}x{}_s{}c0_{}.npy'.format(mlp_num,mlp_num,1, mode))
        save_RCModule(input_tensor_1D,lut_path,
                      model_G.convblock1.module_dict['DepthwiseBlock{}_{}'.format(0, mode)].model.conv1,mlp_num)

        # conv2
        if mode!='s':
            module = model_G.convblock2.module_dict['DepthwiseBlock{}_{}'.format(0, mode)]
            lut_path = os.path.join(opt.expDir, 'LUT_x4_1D_s{}c0_{}.npy'.format(2, mode))
            save_RCLUT(input_tensor_1D, lut_path, module)
        else:
            module = model_G.convblock2.module_dict['DepthwiseBlock{}_{}'.format(0, mode)]
            lut_path = os.path.join(opt.expDir, 'LUT_x4_4bit_int8_s{}c0_{}_compress1.npy'.format(2, mode))
            # lut_path = os.path.join(opt.expDir, 'LUT_x4_4bit_int8_s{}c0_{}.npy'.format(2, mode))
            save_RCLUT(input_tensor_c1, lut_path, module)
            lut_path = os.path.join(opt.expDir, 'LUT_x4_4bit_int8_s{}c0_{}_compress2.npy'.format(2, mode))
            save_RCLUT(input_tensor_c2, lut_path, module)

        mlp_num = mlp_field[mode]
        lut_path = os.path.join(opt.expDir, 'LUT_RC{}x{}_s{}c0_{}.npy'.format(mlp_num, mlp_num, 2, mode))
        save_RCModule(input_tensor_1D, lut_path,
                      model_G.convblock2.module_dict['DepthwiseBlock{}_{}'.format(0, mode)].model.conv1, mlp_num)

def compress_RCLUT_4_4():
    def save_RCLUT(x, lut_path, module):
        # Split input to not over GPU memory
        B = x.size(0) // 100
        outputs = []

        # Extract input-output pairs
        with torch.no_grad():
            model_G.eval()
            for b in range(100):
                if b == 99:
                    batch_input = x[b * B:]
                else:
                    batch_input = x[b * B:(b + 1) * B]

                batch_output = module(batch_input,True)

                results = torch.round(torch.tanh(batch_output) * 127).cpu().data.numpy().astype(np.int8)
                outputs += [results]

        results = np.concatenate(outputs, 0)
        results = results.reshape(x.size(0), -1)
        np.save(lut_path, results)
        print("Resulting LUT size: ", results.shape, "Saved to", lut_path)

    def save_RCModule(x,lut_path,module,mlp_field):
        outputs = []
        with torch.no_grad():
            model_G.eval()
            for i in range(mlp_field * mlp_field):
                num = i + 1
                module1 = getattr(module, 'linear{}'.format(num))
                module2 = getattr(module, 'out{}'.format(num))

                x1 = module1(x[:, :, 0, 0]).unsqueeze(1) # (256,1,nf)

                x2 = module2(x1).unsqueeze(0).cpu().data.numpy() # (1, 256, 1, 1)
                # print(x2.max(),x2.min())
                outputs.append(x2)
        results = np.concatenate(outputs, 0)
        np.save(lut_path, results)
        print("Resulting LUT size: ", results.shape, "Saved to", lut_path)


    # load model
    # opt = TestOptions().parse()

    modes = [i for i in opt.modes]
    stages = opt.stages

    model = getattr(model_new, 'RC_SRNets')

    model_G = model(nf=opt.nf, scale=opt.scale, modes=modes, stages=stages).cuda()

    lm = torch.load(os.path.join(opt.expDir, 'Model_{:06d}.pth'.format(opt.loadIter)))
    model_G.load_state_dict(lm.state_dict(), strict=True)

    input_tensor = get_input_tensor(opt)
    input_tensor_1D = get_input_tensor_1D()
    mlp_field = {'s':5,'d':7,'y':3}
    for mode in modes:
        # input_tensor_c1 = input_tensor.clone()
        input_tensor_c1 = compress_lut_xyzt(input_tensor)
        input_tensor_c2 = compress_lut_larger_interval(input_tensor)

        # conv1
        module = model_G.convblock1.module_dict['DepthwiseBlock{}_{}'.format(0, mode)]
        lut_path = os.path.join(opt.expDir, 'LUT_x4_4bit_int8_s{}c0_{}_compress1.npy'.format(1, mode))
        # lut_path = os.path.join(opt.expDir, 'LUT_x4_4bit_int8_s{}c0_{}.npy'.format(1, mode))
        save_RCLUT(input_tensor_c1, lut_path, module)
        lut_path = os.path.join(opt.expDir, 'LUT_x4_4bit_int8_s{}c0_{}_compress2.npy'.format(1, mode))
        save_RCLUT(input_tensor_c2, lut_path, module)

        mlp_num = mlp_field[mode]
        lut_path = os.path.join(opt.expDir, 'LUT_RC{}x{}_s{}c0_{}.npy'.format(mlp_num,mlp_num,1, mode))
        save_RCModule(input_tensor_1D,lut_path,
                      model_G.convblock1.module_dict['DepthwiseBlock{}_{}'.format(0, mode)].model.conv1,mlp_num)

        # conv2
        if mode!='s':
            input_tensor_channel = input_tensor.reshape((-1, 4, 1, 1))

            module = model_G.convblock2.module_dict['DepthwiseBlock{}_{}'.format(0, mode)]
            lut_path = os.path.join(opt.expDir, 'LUT_x4_block1-4_s{}c0_{}.npy'.format(2, mode))
            save_RCLUT(input_tensor_channel, lut_path, module)
        else:
            module = model_G.convblock2.module_dict['DepthwiseBlock{}_{}'.format(0, mode)]
            lut_path = os.path.join(opt.expDir, 'LUT_x4_4bit_int8_s{}c0_{}_compress1.npy'.format(2, mode))
            # lut_path = os.path.join(opt.expDir, 'LUT_x4_4bit_int8_s{}c0_{}.npy'.format(2, mode))
            save_RCLUT(input_tensor_c1, lut_path, module)
            lut_path = os.path.join(opt.expDir, 'LUT_x4_4bit_int8_s{}c0_{}_compress2.npy'.format(2, mode))
            save_RCLUT(input_tensor_c2, lut_path, module)

        mlp_num = mlp_field[mode]
        lut_path = os.path.join(opt.expDir, 'LUT_RC{}x{}_s{}c0_{}.npy'.format(mlp_num, mlp_num, 2, mode))
        save_RCModule(input_tensor_1D, lut_path,
                      model_G.convblock2.module_dict['DepthwiseBlock{}_{}'.format(0, mode)].model.conv1, mlp_num)


if __name__ == "__main__":
    # opt = TestOptions().parse()
    # compress_MuLUT_IMDB()
    # compress_RCLUT()
    # trans_lut2compress_partial()

    opt_inst = TestOptions()
    opt = opt_inst.parse()
    
    # load model
    opt = TestOptions().parse()
    
    modes = [i for i in opt.modes]
    stages = opt.stages
    
    model = getattr(model_new, 'BaseSRNets')
    
    model_G = model(nf=opt.nf, modes=modes, stages=stages, scale=opt.scale).cuda()
    
    lm = torch.load(os.path.join(opt.expDir, 'Model_{:06d}.pth'.format(opt.loadIter)))
    model_G.load_state_dict(lm.state_dict(), strict=True)
    
    for s in range(stages):
        stage = s + 1
    
        for mode in modes:
            input_tensor = get_input_tensor(opt)
            input_tensor_c1 = compress_lut_xyzt(input_tensor)
            input_tensor_c2 = compress_lut_larger_interval(input_tensor)
    
            if mode != 's':
                input_tensor_c1 = get_mode_input_tensor(input_tensor_c1, mode)
                input_tensor_c2 = get_mode_input_tensor(input_tensor_c2, mode)
    
            lut_path = os.path.join(opt.expDir, 'LUT_x4_4bit_int8_s{}_{}_compress1.npy'.format(str(stage), mode))
            save_lut(input_tensor_c1, lut_path, mode, model_G)
    
            lut_path = os.path.join(opt.expDir, 'LUT_x4_4bit_int8_s{}_{}_compress2.npy'.format(str(stage), mode))
            save_lut(input_tensor_c2, lut_path, mode, model_G)




    # opt_inst = TestOptions()
    # opt = opt_inst.parse()
    #
    # # load model
    # opt = TestOptions().parse()
    #
    # modes = [i for i in opt.modes]
    # stages = opt.stages
    #
    # model = getattr(model_new, 'BaseSRNets')
    #
    # model_G = model(nf=opt.nf, modes=modes, stages=stages).cuda()
    #
    # lm = torch.load(os.path.join(opt.expDir, 'Model_{:06d}.pth'.format(opt.loadIter)))
    # model_G.load_state_dict(lm.state_dict(), strict=True)
    #
    # for s in range(stages):
    #     stage = s + 1
    #
    #     for mode in modes:
    #         input_tensor1 = get_input_tensor2(opt)
    #         input_tensor_c1 = compress_lut2(input_tensor1)
    #
    #         input_tensor2 = get_input_tensor(opt)
    #         input_tensor_c2 = compress_lut_larger_interval(input_tensor2)
    #
    #         if mode != 's':
    #             input_tensor_c1 = get_mode_input_tensor(input_tensor_c1, mode)
    #             input_tensor_c2 = get_mode_input_tensor(input_tensor_c2, mode)
    #
    #         lut_path = os.path.join(opt.expDir, 'LUT_x4_4bit_int8_s{}_{}_compress1.npy'.format(str(stage), mode))
    #         save_lut(input_tensor_c1, lut_path, mode, model_G)
    #
    #         lut_path = os.path.join(opt.expDir, 'LUT_x4_4bit_int8_s{}_{}_compress2.npy'.format(str(stage), mode))
    #         save_lut(input_tensor_c2, lut_path, mode, model_G)
