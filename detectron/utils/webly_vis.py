from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import cv2
import numpy as np
import os
import math

from caffe2.python import workspace

from detectron.core.config import cfg
from detectron.core.config import get_output_dir


def vis_training(cur_iter):
    prefix = ''
    if cfg.WEBLY.MINING:
        prefix = 'mining_'
    if not (cfg.WSL.DEBUG or
            (cfg.WSL.SAMPLE and cur_iter % cfg.WSL.SAMPLE_ITER == 0)):
        return

    output_dir = get_output_dir(cfg.TRAIN.DATASETS, training=True)
    sample_dir = os.path.join(output_dir, 'sample')
    if not os.path.exists(sample_dir):
        os.makedirs(sample_dir)

    for gpu_id in range(cfg.NUM_GPUS):
        data_ids = workspace.FetchBlob('gpu_{}/{}'.format(gpu_id, 'data_ids'))
        ims = workspace.FetchBlob('gpu_{}/{}'.format(gpu_id, 'data'))
        labels_oh = workspace.FetchBlob('gpu_{}/{}'.format(
            gpu_id, 'labels_oh'))
        im_score = workspace.FetchBlob('gpu_{}/{}'.format(gpu_id, 'cls_prob'))
        roi_score = workspace.FetchBlob('gpu_{}/{}'.format(
            gpu_id, prefix + 'rois_pred'))
        # roi_score_softmax = workspace.FetchBlob('gpu_{}/{}'.format(
        # gpu_id, prefix + 'rois_pred_softmax'))
        rois = workspace.FetchBlob('gpu_{}/{}'.format(gpu_id, prefix + 'rois'))
        # anchor_argmax = workspace.FetchBlob('gpu_{}/{}'.format(
        # gpu_id, 'anchor_argmax'))

        preffix = 'iter_' + str(cur_iter) + '_gpu_' + str(gpu_id)
        save_im(labels_oh, im_score, ims, cfg.PIXEL_MEANS, preffix, sample_dir)

        if cfg.WEBLY.ENTROPY:
            classes_weight = workspace.FetchBlob('gpu_{}/{}'.format(
                gpu_id, 'rois_classes_weight'))
        else:
            classes_weight = np.ones_like(im_score)
        save_rois(labels_oh, classes_weight, roi_score, ims, rois,
                  cfg.PIXEL_MEANS, preffix, '', sample_dir)
        # save_rois(labels_oh, classes_weight, roi_score, ims, rois,
        # cfg.PIXEL_MEANS, preffix, '', sample_dir, anchor_argmax)
        # save_rois(labels_oh, classes_weight, roi_score_softmax, ims, rois,
        # cfg.PIXEL_MEANS, preffix, '_softmax', sample_dir)


def save_im(labels_oh, im_score, ims, pixel_means, prefix, output_dir):
    batch_size, num_classes = im_score.shape
    for b in range(batch_size):
        for c in range(num_classes):
            if labels_oh[b][c] == 0.0:
                continue
            im = ims[b, :, :, :].copy()
            channel_swap = (1, 2, 0)
            im = im.transpose(channel_swap)
            im += pixel_means
            im = im.astype(np.uint8)
            file_name = os.path.join(
                output_dir, prefix + '_b_' + str(b) + '_c_' + str(c) + '_' +
                str(im_score[b][c]) + '_' + '.png')
            cv2.imwrite(file_name, im)


def save_rois(labels_oh, classes_weight, roi_score, ims, rois, pixel_means,
              prefix, suffix, output_dir):
    num_rois, num_classes = roi_score.shape
    batch_size, _, height, weight = ims.shape
    for b in range(batch_size):
        for c in range(num_classes):
            if labels_oh[b][c] == 0.0:
                continue
            im = ims[b, :, :, :].copy()
            channel_swap = (1, 2, 0)
            im = im.transpose(channel_swap)
            im += pixel_means
            im = im.astype(np.uint8)
            im_S = im.copy()
            im_A = im.copy()

            argsort = np.argsort(np.abs(roi_score[:, c]))
            scale_p = 1.0 / roi_score[:, c].max()
            scale_p = 1.0
            for n in range(num_rois):
                roi = rois[argsort[n]]
                if roi[0] != b:
                    continue
                jet = gray2jet(roi_score[argsort[n]][c] * scale_p)
                cv2.rectangle(im_S, (roi[1], roi[2]), (roi[3], roi[4]), jet, 1)
            file_name = os.path.join(
                output_dir, prefix + '_b_' + str(b) + '_c_' + str(c) + '_' +
                str(classes_weight[b][c]) + '_' + suffix + '.png')
            cv2.imwrite(file_name, im_S)

            continue
            num_anchors = anchor_argmax.shape[0]
            for n in range(num_rois):
                roi = rois[n]
                if roi[0] != b:
                    continue

                for a in range(num_anchors):
                    if anchor_argmax[a][n] == 1.0:
                        break

                jet = gray2jet(1.0 * a / num_anchors)
                cv2.rectangle(im_A, (roi[1], roi[2]), (roi[3], roi[4]), jet, 1)
            file_name = os.path.join(
                output_dir, prefix + '_b_' + str(b) + '_c_' + str(c) + '_A_' +
                suffix + '.png')
            cv2.imwrite(file_name, im_A)


def dump_proto_files(model, output_dir):
    """Save prototxt descriptions of the training network and parameter
    initialization network."""
    with open(os.path.join(output_dir, model.net.Proto().name), 'w') as fid:
        fid.write(str(model.net.Proto()))
    with open(
            os.path.join(output_dir,
                         model.param_init_net.Proto().name), 'w') as fid:
        fid.write(str(model.param_init_net.Proto()))


def gray2jet(f):
    # plot short rainbow RGB
    a = f / 0.25  # invert and group
    X = math.floor(a)  # this is the integer part
    Y = math.floor(255 * (a - X))  # fractional part from 0 to 255
    Z = math.floor(128 * (a - X))  # fractional part from 0 to 128

    if X == 0:
        r = 0
        g = Y
        b = 128 - Z
    elif X == 1:
        r = Y
        g = 255
        b = 0
    elif X == 2:
        r = 255
        g = 255 - Z
        b = 0
    elif X == 3:
        r = 255
        g = 128 - Z
        b = 0
    elif X == 4:
        r = 255
        g = 0
        b = 0
    # opencv is bgr, not rgb
    return (b, g, r)
