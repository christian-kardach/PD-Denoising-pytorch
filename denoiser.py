from utils import *

# the limitation range of each type of noise level: [0]Gaussian [1]Impulse
limit_set = [[0, 75], [0, 80]]


def denoiser(img, c, pss, model, model_est, opt):
    w, h, _ = img.shape
    img = pixelshuffle(img, pss)
    if opt.tif_file:
        img = img_normalize_tiff(np.float32(img))
    else:
        img = img_normalize(np.float32(img))

    noise_level_list = np.zeros((2 * c, 1))  # two noise types with three channels
    if opt.cond == 0:  # if we use the ground truth of noise for denoising, and only one single noise type
        noise_level_list = np.array(opt.test_noise_level)
    elif opt.cond == 2:  # if we use an external fixed input condition for denoising
        noise_level_list = np.array(opt.ext_test_noise_level)

    # Clean Image Tensor for evaluation
    ISource = np2ts(img)
    # noisy image and true residual
    if opt.real_n == 0 and opt.spat_n == 0:  # no spatial noise setting, and synthetic noise
        if opt.tif_file:
            noisy_img = generate_comp_noisy(img, np.array(opt.test_noise_level) / 65536.)
        else:
            noisy_img = generate_comp_noisy(img, np.array(opt.test_noise_level) / 255.)
        if opt.color == 0:
            noisy_img = np.expand_dims(noisy_img[:, :, 0], 2)
    elif opt.real_n == 1 or opt.real_n == 2:  # testing real noisy images
        noisy_img = img
    elif opt.spat_n == 1:
        noisy_img = generate_noisy(img, 2, 0, 20, 40)
    INoisy = np2ts(noisy_img, opt.color)
    INoisy = torch.clamp(INoisy, 0., 1.)
    True_Res = INoisy - ISource
    ISource, INoisy, True_Res = ISource.cuda(), INoisy.cuda(), True_Res.cuda()

    if opt.mode == "MC":
        # obtain the corresponding input_map
        if opt.cond == 0 or opt.cond == 2:  # if we use ground choose level or the given fixed level
            # normalize noise leve map to [0,1]
            noise_level_list_n = np.zeros((2 * c, 1))
            for noise_type in range(2):
                for chn in range(c):
                    noise_level_list_n[noise_type * c + chn] = normalize(noise_level_list[noise_type * 3 + chn], 1,
                                                                         limit_set[noise_type][0],
                                                                         limit_set[noise_type][
                                                                             1])  # normalize the level value
            # generate noise maps
            noise_map = np.zeros((1, 2 * c, img.shape[0], img.shape[1]))  # initialize the noise map
            noise_map[0, :, :, :] = np.reshape(np.tile(noise_level_list_n, img.shape[0] * img.shape[1]),
                                               (2 * c, img.shape[0], img.shape[1]))
            NM_tensor = torch.from_numpy(noise_map).type(torch.FloatTensor)
            NM_tensor = NM_tensor.cuda()

        # use the estimated noise-level map for blind denoising
        elif opt.cond == 1:  # if we use the estimated map directly
            NM_tensor = torch.clamp(model_est(INoisy), 0., 1.)
            if opt.refine == 1:  # if we need to refine the map before putting it to the denoiser
                NM_tensor_bundle = level_refine(NM_tensor, opt.refine_opt,
                                                2 * c)  # refine_opt can be max, freq and their average
                NM_tensor = NM_tensor_bundle[0]
                noise_estimation_table = np.reshape(NM_tensor_bundle[1], (2 * c,))

            if opt.zeroout == 1:
                NM_tensor = zeroing_out_maps(NM_tensor, opt.keep_ind)
        Res = model(INoisy, NM_tensor)

    elif opt.mode == "B":
        Res = model(INoisy)

    Out = torch.clamp(INoisy - Res, 0., 1.)  # Output image after denoising

    # get the maximum denoising result
    max_NM_tensor = level_refine(NM_tensor, 1, 2 * c)[0]
    max_Res = model(INoisy, max_NM_tensor)
    max_Out = torch.clamp(INoisy - max_Res, 0., 1.)
    max_out_numpy = visual_va2np(max_Out, opt.color, opt.ps, pss, 1, opt.rescale, w, h, c)

    if opt.tif_file:
        noise_map_output = visual_va2np_tiff(max_Res, opt.color, opt.ps, pss, 1, opt.rescale, w, h, c)
    else:
        noise_map_output = visual_va2np(max_Res, opt.color, opt.ps, pss, 1, opt.rescale, w, h, c)

    del max_Out
    del max_Res
    del max_NM_tensor

    if (opt.ps == 1 or opt.ps == 2) and pss != 1:  # pixelshuffle multi-scale
        # create batch of images with one subsitution
        # mosaic_den = visual_va2np(Out, opt.color, 1, pss, 1, opt.rescale, w, h, c)
        out_numpy = np.zeros((pss ** 2, c, w, h))

        # compute all the images in the ps scale set
        for row in range(pss):
            for column in range(pss):
                re_test = visual_va2np(Out, opt.color, 1, pss, 1, opt.rescale, w, h, c, 1,
                                       visual_va2np(INoisy, opt.color), [row, column]) / 255.
                re_test = np.expand_dims(re_test, 0)
                if opt.color == 0:  # if gray image
                    re_test = np.expand_dims(re_test[:, :, :, 0], 3)
                re_test_tensor = torch.from_numpy(np.transpose(re_test, (0, 3, 1, 2))).type(torch.FloatTensor)
                re_test_tensor = re_test_tensor.cuda()
                re_NM_tensor = torch.clamp(model_est(re_test_tensor), 0., 1.)
                if opt.refine == 1:  # if we need to refine the map before putting it to the denoiser
                    re_NM_tensor_bundle = level_refine(re_NM_tensor, opt.refine_opt,
                                                       2 * c)  # refine_opt can be max, freq and their average
                    re_NM_tensor = re_NM_tensor_bundle[0]
                re_Res = model(re_test_tensor, re_NM_tensor)
                Out2 = torch.clamp(re_test_tensor - re_Res, 0., 1.)

                out_numpy[row * pss + column, :, :, :] = Out2.data.cpu().numpy()
                del Out2
                del re_Res
                del re_test_tensor
                del re_NM_tensor
                del re_test

        out_numpy = np.mean(out_numpy, 0)
        out_numpy = np.transpose(out_numpy, (1, 2, 0)) * 255.0

    elif opt.ps == 0 or pss == 1:  # other cases
        out_numpy = visual_va2np(Out, opt.color, 0, 1, 1, opt.rescale, w, h, c)

    out_numpy = out_numpy.astype(np.float32)  # details
    max_out_numpy = max_out_numpy.astype(np.float32)  # background

    # merging the details and background to balance the effect
    k = opt.k
    merge_out_numpy = (1 - k) * out_numpy + k * max_out_numpy
    merge_out_numpy = merge_out_numpy.astype(np.float32)

    return merge_out_numpy, noise_map_output, out_numpy, max_out_numpy
