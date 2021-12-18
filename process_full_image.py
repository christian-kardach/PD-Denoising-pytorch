import os
import glob

import utils
from models import *
from denoiser import *
import progressbar

# the limitation range of each type of noise level: [0]Gaussian [1]Impulse
limit_set = [[0, 75], [0, 80]]


def img_normalize(data):
    return data / 255.


def main(input_file, opt):
    if not os.path.exists(opt.out_dir):
        os.makedirs(opt.out_dir)

    # Build model
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
    #file_extension = os.path.basename(input_file).split(".")[1]

    if opt.real_n == 2:  # have ground truth
        gnd_file_path = os.path.join('data', opt.test_data_gnd, file_name + '_mean.png')
        print(gnd_file_path)
        Img_gnd = cv2.imread(gnd_file_path)
        Img_gnd = Img_gnd[:, :, ::-1]
        Img_gnd = cv2.resize(Img_gnd, (0, 0), fx=opt.scale, fy=opt.scale)
        Img_gnd = img_normalize(np.float32(Img_gnd))

    # image
    Img = cv2.imread(input_file)  # input image with w*h*c

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
    noise_map_output = np.zeros([w, h, 3])
    background_out = np.zeros([w, h, 3])
    details_out = np.zeros([w, h, 3])

    # print('Splitting and Testing.....')

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

    bar = progressbar.ProgressBar(max_value=total_patches)
    i = 0
    patches = 0
    while i < w:
        i_end = min(i + opt.wbin, w)
        j = 0
        while j < h:
            j_end = min(j + opt.wbin, h)
            patch = Img[i:i_end, j:j_end, :]
            bar.update(patches)

            patch_merge_out_numpy, noise_patch, details_patch, background_patch = denoiser(patch, c, pss, model,
                                                                                         model_est, opt)
            merge_out[i:i_end, j:j_end, :] = patch_merge_out_numpy

            noise_map_output[i:i_end, j:j_end, :] = noise_patch
            details_out[i:i_end, j:j_end, :] = details_patch
            background_out[i:i_end, j:j_end, :] = background_patch

            j = j_end
            patches += 1
        i = i_end

    export_path_merged = os.path.normpath(
        os.path.join(opt.out_dir, file_name + '_merged.png'))

    export_path_noise = os.path.normpath(
        os.path.join(opt.out_dir, file_name + '_noise_mask.png'))

    export_path_details = os.path.normpath(
        os.path.join(opt.out_dir, file_name + '_details.png'))

    export_path_background = os.path.normpath(
        os.path.join(opt.out_dir, file_name + '_background.png'))

    print("\nExporting images: ")
    print(export_path_merged)
    cv2.imwrite(export_path_merged, merge_out[:, :, ::-1], [cv2.IMWRITE_PNG_COMPRESSION, 0])

    print(export_path_noise)
    cv2.imwrite(export_path_noise, noise_map_output[:, :, ::-1], [cv2.IMWRITE_PNG_COMPRESSION, 0])

    print(export_path_details)
    cv2.imwrite(export_path_details, details_out[:, :, ::-1], [cv2.IMWRITE_PNG_COMPRESSION, 0])

    print(export_path_background)
    cv2.imwrite(export_path_background, background_out[:, :, ::-1], [cv2.IMWRITE_PNG_COMPRESSION, 0])

    print('done!')


class Opt:
    color = 1
    cond = 1
    delog = ''  # model path, not used
    ext_test_noise_level = None
    k = 0.0
    keep_ind = [0]
    mode = 'MC'
    num_of_layers = 20
    out_dir = 'results'
    output_map = 0  # whether to output maps
    ps = 0  # pixel shuffle [0]no pixel-shuffle [1]adaptive pixel-ps [2]pre-set stride
    ps_scale = 2 # if ps==2, use this pixel shuffle stride
    real_n = 1  # [0]synthetic noises [1]real noisy image wo gnd [2]real noisy image with gnd'
    refine = 0  # [0]no refinement of estimation [1]refinement of the estimation
    refine_opt = 0  # [0]get the most frequent [1]the maximum [2]Gaussian smooth [3]average value of 0 and 1 opt
    rescale = 1  # resize it back to the original size after downscaling
    scale = 1.0  # resize the original images
    spat_n = 0  # whether to add spatial-variant signal-dependent noise, [0]no spatial [1]Gaussian-possion noise
    test_data = 'astro'
    test_data_gnd = 'Set12'
    test_noise_level = None
    wbin = 128
    zeroout = 0


if __name__ == "__main__":
    # load data info
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
