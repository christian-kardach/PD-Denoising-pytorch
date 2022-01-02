import os
import glob
import argparse

import numpy as np
import progressbar
import tifffile
import cv2

import utils
from models import *
from denoiser import *
import zlicer

DEBUG_OUTPUT = False

# the limitation range of each type of noise level: [0]Gaussian [1]Impulse
limit_set = [[0, 75], [0, 80]]


def calculate_number_of_patches(img, opt):
    w, h, _ = img.shape
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

    return total_patches


def main(input_file, opt):
    if not os.path.exists(opt.out_dir):
        os.makedirs(opt.out_dir)

    # Load model
    print('Loading model...')

    c = 1 if opt.color == 0 else 3

    if opt.color == 0:
        opt.delog = 'logs/logs_gray_MC_AWGN_RVIN'
    else:
        opt.delog = 'logs/logs_color_MC_AWGN_RVIN'

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
    print('Loading data info...\n')

    # process images with pre-defined noise level
    if opt.color == 1:
        image_type = "color"
    else:
        image_type = "grayscale"

    print("Starting to process: {input_file} of type {image_type}".format(input_file=input_file, image_type=image_type))

    file_name = os.path.basename(input_file).split(".")[0]
    file_extension = os.path.basename(input_file).split(".")[1]

    if opt.real_n == 2:  # have ground truth
        gnd_file_path = os.path.join('data', opt.test_data_gnd, file_name + '_mean.png')
        print(gnd_file_path)
        Img_gnd = cv2.imread(gnd_file_path)
        Img_gnd = Img_gnd[:, :, ::-1]
        Img_gnd = cv2.resize(Img_gnd, (0, 0), fx=opt.scale, fy=opt.scale)
        Img_gnd = utils.img_normalize(np.float32(Img_gnd))

    # image
    if file_extension == "tif":
        opt.tif_file = True
        Img = tifffile.imread(input_file)
    else:
        Img = cv2.imread(input_file)  # input image with w*h*c

    # NOTE: OpenCV - ROWS, COLS - So a 1920x1080 image will be 1080 ROWS by 1920 COLUMNS (1080,1920)
    rows, cols, chan = Img.shape
    print("Image size {width}, {height}, {c}".format(width=cols, height=rows, c=chan))

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

    # Slice image
    zl = zlicer.Zlicer(Img, opt.wbin, opt.patch_margin)
    noise_map_output = np.zeros([rows, cols, 3])
    margin = opt.patch_margin

    if DEBUG_OUTPUT:
        print("ROWS: " + str(zl.rows))
        print("COLS: " + str(zl.columns))
        print("PATCH COUNT: " + str(zl.patch_count))
        print("Img SHAPE: " + str(Img.shape))

        thickness = 1

        dark_grey = (10, 10, 10)
        white = (255, 255, 255)
        blue = (255, 0, 0)
        green = (0, 255, 0)
        red = (0, 0, 255)

        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 0.3

        # PATCHES AND ID's
        for patch in zl.patches:
            noise_map_output = cv2.putText(noise_map_output, "{id}".format(id=patch.id), (patch.x + 64, patch.y + 15),
                                           font,
                                           fontScale, white,
                                           thickness,
                                           cv2.LINE_AA)

            noise_map_output = cv2.rectangle(noise_map_output, (patch.x, patch.y), (patch.x_end, patch.y_end),
                                             dark_grey,
                                             thickness)

    if opt.new_slicer:
        # So a 1920x1080 image will be 1080 ROWS by 1920 COLUMNS (1080,1920)
        # TODO: Write a re-usable function for handling margins and patches
        bar = progressbar.ProgressBar(max_value=zl.patch_count)
        patches_done = 0
        for patch in zl.patches:
            # Top Left
            if patch.column == 0 and patch.row == 0:
                patch_data = Img[patch.y:patch.y_end + margin, patch.x:patch.x_end + margin, :]
                patch_merge_out_numpy, noise_patch, details_patch, background_patch = denoiser(patch_data, c, pss,
                                                                                               model,
                                                                                               model_est, opt)

                noise_map_output[patch.y:patch.y_end, patch.x:patch.x_end, :] = noise_patch[
                                                                                0:patch.patch_size_y,
                                                                                0:patch.patch_size_x, :]
                patch.done = True

            # Bottom Left
            if patch.row == zl.rows and patch.column == 0:
                patch_data = Img[patch.y - margin:patch.y_end, patch.x:patch.x_end + margin, :]
                patch_merge_out_numpy, noise_patch, details_patch, background_patch = denoiser(patch_data, c, pss,
                                                                                               model,
                                                                                               model_est, opt)

                noise_map_output[patch.y:patch.y_end, patch.x:patch.x_end, :] = noise_patch[
                                                                                margin:patch.patch_size_y + margin,
                                                                                0:patch.patch_size_x, :]
                patch.done = True

            # Top Right
            elif patch.column == zl.columns and patch.row == 0:
                patch_data = Img[patch.y:patch.y_end + margin, patch.x - margin:patch.x_end, :]
                patch_merge_out_numpy, noise_patch, details_patch, background_patch = denoiser(patch_data, c, pss,
                                                                                               model,
                                                                                               model_est, opt)

                noise_map_output[patch.y:patch.y_end, patch.x:patch.x_end, :] = noise_patch[
                                                                                0:patch.patch_size_y,
                                                                                margin:patch.patch_size_x + margin, :]
                patch.done = True

            # Bottom Right
            elif patch.column == zl.columns and patch.row == zl.rows:
                patch_data = Img[patch.y - margin:patch.y_end, patch.x - margin:patch.x_end, :]
                patch_merge_out_numpy, noise_patch, details_patch, background_patch = denoiser(patch_data, c, pss,
                                                                                               model,
                                                                                               model_est, opt)

                noise_map_output[patch.y:patch.y_end, patch.x:patch.x_end, :] = noise_patch[
                                                                                margin:patch.patch_size_y + margin,
                                                                                margin:patch.patch_size_x + margin,
                                                                                :]

                patch.done = True

            # Top Row
            elif patch.row == 0 and not patch.done:
                patch_data = Img[patch.y:patch.y_end + margin, patch.x - margin:patch.x_end + margin, :]
                patch_merge_out_numpy, noise_patch, details_patch, background_patch = denoiser(patch_data, c, pss,
                                                                                               model,
                                                                                               model_est, opt)

                noise_map_output[patch.y:patch.y_end, patch.x:patch.x_end, :] = noise_patch[
                                                                                0:patch.patch_size_y,
                                                                                margin:patch.patch_size_x + margin,
                                                                                :]
                patch.done = True

            # Left Column
            elif patch.column == 0 and not patch.done:
                patch_data = Img[patch.y - margin:patch.y_end + margin, patch.x:patch.x_end + margin, :]
                patch_merge_out_numpy, noise_patch, details_patch, background_patch = denoiser(patch_data, c, pss,
                                                                                               model,
                                                                                               model_est, opt)

                noise_map_output[patch.y:patch.y_end, patch.x:patch.x_end, :] = noise_patch[
                                                                                margin:patch.patch_size_y + margin,
                                                                                0:patch.patch_size_x,
                                                                                :]
                patch.done = True

            # Bottom Row
            elif patch.row == zl.rows and not patch.done:
                patch_data = Img[patch.y - margin:patch.y_end, patch.x - margin:patch.x_end + margin, :]
                patch_merge_out_numpy, noise_patch, details_patch, background_patch = denoiser(patch_data, c, pss,
                                                                                               model,
                                                                                               model_est, opt)

                noise_map_output[patch.y:patch.y_end, patch.x:patch.x_end, :] = noise_patch[
                                                                                margin:patch.patch_size_y + margin,
                                                                                margin:patch.patch_size_x + margin,
                                                                                :]
                patch.done = True

            # Right Column
            elif patch.column == zl.columns and not patch.done:
                patch_data = Img[patch.y - margin:patch.y_end + margin, patch.x - margin:patch.x_end, :]
                patch_merge_out_numpy, noise_patch, details_patch, background_patch = denoiser(patch_data, c, pss,
                                                                                               model,
                                                                                               model_est, opt)

                noise_map_output[patch.y:patch.y_end, patch.x:patch.x_end, :] = noise_patch[
                                                                                margin:patch.patch_size_y + margin,
                                                                                margin:patch.patch_size_x + margin, :]
                patch.done = True

            # Second to last row - need to make sure we have coverage for padding
            elif patch.row == zl.rows - 1 and not patch.done:
                if zl.bottom_edge_sample.patch_size_y >= margin:
                    patch_data = Img[patch.y - margin:patch.y_end + margin, patch.x - margin:patch.x_end + margin, :]
                    patch_merge_out_numpy, noise_patch, details_patch, background_patch = denoiser(patch_data, c, pss,
                                                                                                   model,
                                                                                                   model_est, opt)

                    noise_map_output[patch.y:patch.y_end, patch.x:patch.x_end, :] = noise_patch[
                                                                                    margin:patch.patch_size_y + margin,
                                                                                    margin:patch.patch_size_x + margin,
                                                                                    :]
                    patch.done = True

                else:
                    patch_data = Img[patch.y - margin:patch.y_end + test_patch.patch_size_y,
                                 patch.x - margin:patch.x_end + margin, :]
                    patch_merge_out_numpy, noise_patch, details_patch, background_patch = denoiser(patch_data, c, pss,
                                                                                                   model,
                                                                                                   model_est, opt)

                    noise_map_output[patch.y:patch.y_end, patch.x:patch.x_end, :] = noise_patch[
                                                                                    margin:patch.patch_size_y + margin,
                                                                                    margin:patch.patch_size_x + margin,
                                                                                    :]
                    patch.done = True

            # Second to last row - need to make sure we have coverage for padding
            elif patch.column == zl.columns - 1 and not patch.done:
                test_patch = zl.get_right_sample()
                if test_patch.patch_size_x >= margin:
                    patch_data = Img[patch.y - margin:patch.y_end + margin, patch.x - margin:patch.x_end + margin, :]
                    patch_merge_out_numpy, noise_patch, details_patch, background_patch = denoiser(patch_data, c, pss,
                                                                                                   model,
                                                                                                   model_est, opt)

                    noise_map_output[patch.y:patch.y_end, patch.x:patch.x_end, :] = noise_patch[
                                                                                    margin:patch.patch_size_y + margin,
                                                                                    margin:patch.patch_size_x + margin,
                                                                                    :]
                    patch.done = True

                else:
                    patch_data = Img[patch.y - margin:patch.y_end + margin,
                                 patch.x - margin:patch.x_end + test_patch.patch_size_x, :]
                    patch_merge_out_numpy, noise_patch, details_patch, background_patch = denoiser(patch_data, c, pss,
                                                                                                   model,
                                                                                                   model_est, opt)

                    noise_map_output[patch.y:patch.y_end, patch.x:patch.x_end, :] = noise_patch[
                                                                                    margin:patch.patch_size_y + margin,
                                                                                    margin:patch.patch_size_x + margin,
                                                                                    :]
                    patch.done = True

            elif not patch.done:
                patch_data = Img[patch.y - margin:patch.y_end + margin,
                             patch.x - margin:patch.x_end + margin, :]
                patch_merge_out_numpy, noise_patch, details_patch, background_patch = denoiser(patch_data, c, pss,
                                                                                               model,
                                                                                               model_est, opt)

                noise_map_output[patch.y:patch.y_end, patch.x:patch.x_end, :] = noise_patch[
                                                                                margin:patch.patch_size_y + margin,
                                                                                margin:patch.patch_size_x + margin,
                                                                                :]
                patch.done = True

            bar.update(patches_done)
            patches_done += 1

    else:
        # ORIGINAL
        img_rows, img_cols, _ = Img.shape
        noise_map_output = np.zeros([rows, cols, 3])
        patch_count = 0
        i = 0
        while i < img_rows:
            i_end = min(i + opt.wbin, img_rows)

            j = 0
            while j < img_cols:
                j_end = min(j + opt.wbin, img_cols)
                patch = Img[i:i_end, j:j_end, :]
                # bar.update(total_patches)

                patch_merge_out_numpy, noise_patch, details_patch, background_patch = denoiser(patch, c, pss, model,
                                                                                               model_est, opt)

                noise_map_output[i:i_end, j:j_end, :] = noise_patch

                j = j_end
                patch_count += 1
            i = i_end

    if DEBUG_OUTPUT:
        cv2.namedWindow("result", cv2.WINDOW_NORMAL)
        cv2.imshow("result", noise_map_output)
        cv2.waitKey(0)

    if opt.tif_file:
        export_path_noise = os.path.normpath(
            os.path.join(opt.out_dir, file_name + '_noise_mask.tif'))

    else:
        export_path_noise = os.path.normpath(
            os.path.join(opt.out_dir, file_name + '_noise_mask.png'))

    print("\nExporting images: ")
    print(export_path_noise)
    if opt.tif_file:
        tifffile.imwrite(export_path_noise, noise_map_output[:, :, ::-1], photometric='rgb')
    else:
        cv2.imwrite(export_path_noise, noise_map_output[:, :, ::-1], [cv2.IMWRITE_PNG_COMPRESSION, 0])

    print('done!')


class Opt:
    color = 1  # [0]gray [1]color - automatically set
    cond = 1  # Testing mode using noise map of: [0]Groundtruth [1]Estimated [2]External Input
    delog = ''  # model path, not used
    ext_test_noise_level = None  # external noise level input used if cond==2
    k = 0.0  # merging factor between details and background
    keep_ind = [0]  # [0 1 2]Gaussian [3 4 5]Impulse
    mode = 'MC'  # DnCNN-B (B) or MC-AWGN-RVIN (MC)
    num_of_layers = 20
    out_dir = 'results'
    output_map = 0  # whether to output maps
    ps = 0  # pixel shuffle [0]no pixel-shuffle [1]adaptive pixel-ps [2]pre-set stride
    ps_scale = 2  # if ps==2, use this pixel shuffle stride
    real_n = 1  # [0]synthetic noises [1]real noisy image wo gnd [2]real noisy image with gnd'
    refine = 0  # [0]no refinement of estimation [1]refinement of the estimation
    refine_opt = 1  # [0]get the most frequent [1]the maximum [2]Gaussian smooth [3]average value of 0 and 1 opt
    rescale = 1  # resize it back to the original size after downscaling
    scale = 1.0  # resize the original images
    spat_n = 0  # whether to add spatial-variant signal-dependent noise, [0]no spatial [1]Gaussian-possion noise
    test_data = 'astro'
    test_data_gnd = 'Set12'
    test_noise_level = None
    wbin = 128
    patch_margin = 20
    zeroout = 0
    tif_file = False
    new_slicer = True


if __name__ == "__main__":
    # Set sample patch size
    parser = argparse.ArgumentParser(description="PD-denoising")
    parser.add_argument("--wbin", type=int, default=128, help='patch size while testing on full images')
    parser.add_argument("--pmargin", type=int, default=20, help='patch size while testing on full images')
    args = parser.parse_args()

    Opt.wbin = args.wbin
    Opt.patch_margin = args.pmargin

    # Check images
    print('Checking image types...')
    files_source = glob.glob(os.path.join('data', Opt.test_data, '*.*'))
    files_source.sort()

    for f in files_source:
        is_grayscale = utils.isgray(f)
        if is_grayscale:
            Opt.color = 0
        else:
            Opt.color = 1

        main(f, Opt)
