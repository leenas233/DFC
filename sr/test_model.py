import numpy as np
import sys
sys.path.insert(0, "../")  # run under the current directory
from common.utils import PSNR, cal_ssim, modcrop, _rgb2ycbcr
from PIL import Image
import os
import torch
import model_new
import model
from data import SRBenchmark

# valDir = '../data/Benchmark'
# # all_datasets = ['Set5', 'Set14', 'B100', 'Manga109', 'Urban100']
# scale = 4
# valoutDir = '../result_rebuttal/SR'
# expDir = '../models/SR/sr_RCLUT_nf64_xyzt2i5_exp3_notanh2_finetune'
# lut_name = 'sr_RCLUT_nf64_xyzt2i5_exp3_notanh2_finetune'
# startIter = 198000
# nf = 64
# modes = 'sdy'
# stages = 2


# valDir = '../data/SRBenchmark'
# # all_datasets = ['Set5', 'Set14', 'B100', 'Manga109', 'Urban100']
# scale = 4
# valoutDir = '../result_rebuttal/SR'
# expDir = '../models_rebuttal/SR/rebuttal/spflut_xyzt2i5_partial_channel'
# lut_name = 'spflut_xyzt2i5_partial_channel'
# startIter = 198000
# nf = 64
# modes = 'sdy'
# stages = 2


# valDir = '../data/SRBenchmark'
# # all_datasets = ['Set5', 'Set14', 'B100', 'Manga109', 'Urban100']
# scale = 4
# valoutDir = '../result_rebuttal/SR'
# expDir = '../models_rebuttal/SR/rebuttal/mulutx2_xyzt6i5_partial'
# lut_name = 'mulutx2_xyzt6i5_partial'
# startIter = 198000
# nf = 64
# modes = 'sdy'
# stages = 2

# valDir = '../data/Benchmark'
# # all_datasets = ['Set5', 'Set14', 'B100', 'Manga109', 'Urban100']
# scale = 4
# valoutDir = '../result_rebuttal/SR'
# expDir = '../models_rebuttal/SR/srlut_test'
# lut_name = 'srlut_test'
# startIter = 194000
# nf = 64
# modes = 's'
# stages = 1

valDir = '../data/Benchmark'
# all_datasets = ['Set5', 'Set14', 'B100', 'Manga109', 'Urban100']
scale = 4
valoutDir = '../result_rebuttal/SR'
expDir = '../models/sr_mulut'
lut_name = 'LUT_ft_x4_4bit_int8'
compressed_dimensions = 'xyzt'
diagonal_width = 2
sampling_interval = 5
modes = 'sdy'
stages = 2

def valid_steps(model_G, valid,exp_name):
    # datasets = ['Set5', 'Set14', 'B100', 'Urban100', 'Manga109']
    datasets = ['Set5', 'Set14']

    with torch.no_grad():
        model_G.eval()

        for i in range(len(datasets)):
            psnrs = []
            ssim = []
            files = valid.files[datasets[i]]

            # result_path = os.path.join(valoutDir, datasets[i])
            result_path = os.path.join(valoutDir, exp_name, datasets[i], "X{}".format(scale))
            if not os.path.isdir(result_path):
                os.makedirs(result_path)

            for j in range(len(files)):
                key = datasets[i] + '_' + files[j][:-4]

                lb = valid.ims[key]
                input_im = valid.ims[key + 'x%d' % scale]

                input_im = input_im.astype(np.float32) / 255.0
                im = torch.Tensor(np.expand_dims(
                    np.transpose(input_im, [2, 0, 1]), axis=0)).cuda()

                # pred = mulut_predict(model_G, im, 'valid', opt)
                pred = model_G(im, 'valid')
                # print(pred.max(), pred.min())

                pred = np.transpose(np.squeeze(
                    pred.data.cpu().numpy(), 0), [1, 2, 0])
                pred = np.round(np.clip(pred, 0, 255)).astype(np.uint8)

                left, right = _rgb2ycbcr(pred)[:, :, 0], _rgb2ycbcr(lb)[:, :, 0]
                psnrs.append(PSNR(left, right, scale))  # single channel, no scale change
                ssim.append(cal_ssim(right, left))

                Image.fromarray(pred).save(
                    os.path.join(result_path, '{}_{}_{}bit.png'.format(files[j][:-4], lut_name,4)))

            print('Dataset {} | AVG Val PSNR: {:02f} | AVG Val SSIM: {:02f}'.format(datasets[i],
                                                                                    np.mean(np.asarray(psnrs)),
                                                                                    np.mean(np.asarray(ssim))))

if __name__ == '__main__':

    # model = getattr(model_new, 'BaseSRNets')
    
    # model_G = model(nf=nf, scale=scale, modes=modes, stages=stages).cuda()
    
    # lm = torch.load(os.path.join(expDir, 'Model_{:06d}.pth'.format(startIter)))
    
    # model_G.load_state_dict(lm.state_dict(), strict=True)
    # lm = 0


    # model = getattr(model_new, 'cBaseMuLUT_xyzt')
    # model_G = model(lut_folder=expDir, stages=stages, modes=modes, upscale=scale, interval=4, d=14, phase = 'test').cuda()

    # valid = SRBenchmark(valDir, scale=scale)

    # valid_steps(model_G, valid,expDir.split('/')[-1])

    # SPF-LUT +DFC
    # model = getattr(model, 'SPF_LUT_DFC')
    # model_G = model(lut_folder=expDir, stages=stages, modes=modes, upscale=scale, interval=4, phase = 'test',
    #                 compressed_dimensions=compressed_dimensions, diagonal_width=diagonal_width, sampling_interval=sampling_interval,
    #                 lutName = lut_name).cuda()

    # valid = SRBenchmark(valDir, scale=scale)

    # valid_steps(model_G, valid,expDir.split('/')[-1])

    # SPF-LUT
    # model = getattr(model, 'SPF_LUT')
    # model_G = model(lut_folder=expDir, stages=stages, modes=modes, upscale=scale, interval=4,
    #                 lutName = lut_name).cuda()

    # valid = SRBenchmark(valDir, scale=scale)

    # valid_steps(model_G, valid,expDir.split('/')[-1])


    # MuLUT +DFC and SR-LUT +DFC
    # model = getattr(model, 'BaseMuLUT_DFC')
    # model_G = model(lut_folder=expDir, stages=stages, modes=modes, upscale=scale, interval=4, phase = 'test',
    #                 compressed_dimensions=compressed_dimensions, diagonal_width=diagonal_width, sampling_interval=sampling_interval,
    #                 lutName = lut_name).cuda()

    # valid = SRBenchmark(valDir, scale=scale)

    # valid_steps(model_G, valid,expDir.split('/')[-1])


    # MuLUT
    model = getattr(model, 'MuLUT')
    model_G = model(lut_folder=expDir, stages=stages, modes=modes, upscale=scale, interval=4,
                    lutName = lut_name).cuda()

    valid = SRBenchmark(valDir, scale=scale)

    valid_steps(model_G, valid,expDir.split('/')[-1])

