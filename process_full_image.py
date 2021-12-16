import cv2
import os
import argparse
import glob
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from models import *
from utils import *
from denoiser import *
from PIL import Image
import scipy.io as sio
import tifffile as tiff

# the limitation range of each type of noise level: [0]Gaussian [1]Impulse
limit_set = [[0, 75], [0, 80]]


def img_normalize(data):
    return data / 255.


"""
def main(scale=1, wbin=128, ps=0, ps_scale=2, real=1, real_n=0, k=0, mode="MC", color=1, output_map=0, zeroout=0, keep_ind=0,
         num_of_layers=20, test_data_gnd="Set12", delog="logs/logs_color_MC_AWGN_RVIN", cond=1, refine=0, refine_opt=1, test_data="beijing",
         out_dir="results/beijing"):
"""


def main(opt):
    if not os.path.exists(opt.out_dir):
        os.makedirs(opt.out_dir)

    # Build model
    print('Loading model ...\n')
    c = 1 if opt.color == 0 else 3
    net = DnCNN_c(channels=c, num_of_layers=opt.num_of_layers, num_of_est=2 * c)
    est_net = Estimation_direct(c, 2 * c)

    device_ids = [0]
    model = nn.DataParallel(net, device_ids=device_ids).cuda()
    model.load_state_dict(torch.load(os.path.join(opt.delog, 'net.pth')))
    model.eval()

    # Estimator Model
    model_est = nn.DataParallel(est_net, device_ids=device_ids).cuda()
    model_est.load_state_dict(torch.load(os.path.join(opt.delog, 'est_net.pth')))
    model_est.eval()

    # load data info
    print('Loading data info ...\n')
    files_source = glob.glob(os.path.join('data', opt.test_data, '*.*'))
    files_source.sort()

    # process images with pre-defined noise level
    for f in files_source:
        print(f)
        # file_name = f.split('/')[-1].split('.')[0]
        file_name = os.path.basename(f).split(".")[0]
        file_extension = os.path.basename(f).split(".")[1]
        if opt.real_n == 2:  # have ground truth
            gnd_file_path = os.path.join('data', opt.test_data_gnd, file_name + '_mean.png')
            print(gnd_file_path)
            Img_gnd = cv2.imread(gnd_file_path)
            Img_gnd = Img_gnd[:, :, ::-1]
            Img_gnd = cv2.resize(Img_gnd, (0, 0), fx=opt.scale, fy=opt.scale)
            Img_gnd = img_normalize(np.float32(Img_gnd))

        # image
        Img = cv2.imread(f)  # input image with w*h*c

        w, h, _ = Img.shape
        Img = Img[:, :, ::-1]  # change it to RGB
        Img = cv2.resize(Img, (0, 0), fx=opt.scale, fy=opt.scale)
        if opt.color == 0:
            Img = Img[:, :, 0]  # For gray images
            Img = np.expand_dims(Img, 2)
        pss = 1
        if opt.ps == 1:
            pss = decide_scale_factor(Img / 255., model_est, color=opt.color, thre=0.008, plot_flag=1, stopping=4,
                                      mark=opt.out_dir + '/' + file_name)[0]
            print(pss)
            Img = pixelshuffle(Img, pss)
        elif opt.ps == 2:
            pss = opt.ps_scale

        merge_out = np.zeros([w, h, 3])
        max_NM_tensor_out = np.zeros([w, h, 3])
        max_Res_out = np.zeros([w, h, 3])

        print('Splitting and Testing.....')

        i = 0
        total_patches = 0
        while i < w:
            i_end = min(i + opt.wbin, w)
            j = 0
            while j < h:
                j_end = min(j + opt.wbin, h)
                j = j_end
                total_patches += 1
            i = i_end

        print("Total patches:", total_patches)
        i = 0
        patches = 0
        while i < w:
            i_end = min(i + opt.wbin, w)
            j = 0
            while j < h:
                j_end = min(j + opt.wbin, h)
                patch = Img[i:i_end, j:j_end, :]
                print("Doing patch {patches} of {total_patches}".format(patches=patches, total_patches=total_patches))

                patch_merge_out_numpy, max_Res_out_patch, max_NM_tensor_out_patch = denoiser(patch, c, pss, model,
                                                                                             model_est, opt)
                merge_out[i:i_end, j:j_end, :] = patch_merge_out_numpy

                # print("MaxRes:", max_Res_out_patch.shape)
                # print("max_NM:", max_NM_tensor_out_patch.shape)

                max_Res_out[i:i_end, j:j_end, :] = max_Res_out_patch
                max_NM_tensor_out[i:i_end, j:j_end, :] = max_NM_tensor_out_patch

                j = j_end
                patches += 1
            i = i_end

        export_path = os.path.normpath(
            os.path.join(opt.out_dir, file_name + '_denoised_' + str(pss) + '_k' + str(opt.k) + '.png'))
        export_path_res = os.path.normpath(
            os.path.join(opt.out_dir, file_name + '_mask_pss' + str(pss) + '_k' + str(opt.k) + '.png'))
        export_path_nm = os.path.normpath(
            os.path.join(opt.out_dir, file_name + '_nm_pss' + str(pss) + '_k' + str(opt.k) + '.png'))

        print("Exporting images: ")
        print(export_path)
        cv2.imwrite(export_path, merge_out[:, :, ::-1], [cv2.IMWRITE_PNG_COMPRESSION, 0])
        print(export_path_res)
        cv2.imwrite(export_path_res, max_Res_out[:, :, ::-1], [cv2.IMWRITE_PNG_COMPRESSION, 0])
        print(export_path_nm)
        cv2.imwrite(export_path_nm, max_NM_tensor_out[:, :, ::-1], [cv2.IMWRITE_PNG_COMPRESSION, 0])
        print('done!')


class Opt:
    color = 1
    cond = 1
    delog = 'logs/logs_color_MC_AWGN_RVIN'
    ext_test_noise_level = None
    k = 0.0
    keep_ind = [0]
    mode = 'MC'
    num_of_layers = 20
    out_dir = 'results'
    output_map = 0
    ps = 0
    ps_scale = 2
    real_n = 1
    refine = 0
    refine_opt = 1
    rescale = 1
    scale = 1.0
    spat_n = 0
    test_data = 'astro'
    test_data_gnd = 'Set12'
    test_noise_level = None
    wbin = 128
    zeroout = 0


if __name__ == "__main__":
    main(Opt)
