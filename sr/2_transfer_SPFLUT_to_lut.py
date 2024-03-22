import os
import sys

import numpy as np
import torch

sys.path.insert(0, "../")  # run under the current directory
from common.option import TestOptions
import model as Model


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


def transfer_SPFLUT_to_lut(opt):
    def save_SPFLUT(x, lut_path, module):
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



    modes = [i for i in opt.modes]
    stages = opt.stages

    model = getattr(Model, 'SPF_LUT_net')

    model_G = model(nf=opt.nf, scale=opt.scale, modes=modes, stages=stages).cuda()

    lm = torch.load(os.path.join(opt.expDir, 'Model_{:06d}.pth'.format(opt.loadIter)))
    model_G.load_state_dict(lm.state_dict(), strict=True)

    input_tensor = get_input_tensor(opt)
    for mode in modes:
        input_tensor_c1 = input_tensor.clone()
        if mode != 's':
            input_tensor_c1 = get_mode_input_tensor(input_tensor_c1, mode)

        # conv1
        module = model_G.convblock1.module_dict['DepthwiseBlock{}_{}'.format(0, mode)]
        lut_path = os.path.join(opt.expDir, '{}_s{}c0_{}.npy'.format(opt.lutName, 1, mode))
        save_SPFLUT(input_tensor_c1, lut_path, module)


        # conv2
        module = model_G.convblock2.module_dict['DepthwiseBlock{}_{}'.format(0, mode)]
        lut_path = os.path.join(opt.expDir, '{}_s{}c0_{}.npy'.format(opt.lutName, 2, mode))
        save_SPFLUT(input_tensor_c1, lut_path, module)

        # conv3
        module = model_G.convblock3.module_dict['DepthwiseBlock{}_{}'.format(0, mode)]
        lut_path = os.path.join(opt.expDir, '{}_s{}c0_{}.npy'.format(opt.lutName, 3, mode))
        save_SPFLUT(input_tensor_c1, lut_path, module)

        # conv4
        module = model_G.convblock4.module_dict['DepthwiseBlock{}_{}'.format(0, mode)]
        lut_path = os.path.join(opt.expDir, '{}_s{}c0_{}.npy'.format(opt.lutName, 4, mode))
        save_SPFLUT(input_tensor_c1, lut_path, module)

        # conv6
        for c in range(4):
            module = model_G.upblock.module_dict['DepthwiseBlock{}_{}'.format(c, mode)]
            lut_path = os.path.join(opt.expDir, '{}_s{}c{}_{}.npy'.format(opt.lutName, 6, c, mode))
            save_SPFLUT(input_tensor_c1, lut_path, module)

    # conv5
    input_tensor = input_tensor.reshape((-1,4,1,1))
    module = model_G.ChannelConv
    lut_path = os.path.join(opt.expDir, '{}_s{}_channel.npy'.format(opt.lutName, 5))
    save_SPFLUT(input_tensor, lut_path, module)


if __name__ == "__main__":
    opt = TestOptions().parse()
    transfer_SPFLUT_to_lut(opt)
    
