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
    # if not (cfg.WSL.DEBUG or
    #         (cfg.WSL.SAMPLE and cur_iter % cfg.WSL.SAMPLE_ITER == 0)):
    #     return
    if cur_iter % 100 != 0:
        return

    output_dir = get_output_dir(cfg.TRAIN.DATASETS, training=True)
    sample_dir = os.path.join(output_dir, 'DB_sample')
    if not os.path.exists(sample_dir):
        os.makedirs(sample_dir)

    for gpu_id in range(cfg.NUM_GPUS):
        db2_add = workspace.FetchBlob('gpu_{}/{}'.format(gpu_id, 'db2_add'))
        conv5_3 = workspace.FetchBlob('gpu_{}/{}'.format(gpu_id, 'conv5_3'))
        ims = workspace.FetchBlob('gpu_{}/{}'.format(gpu_id, 'data'))

        prefix = 'iter_' + str(cur_iter) + '_gpu_' + str(gpu_id)
        if False:
            save_im(ims, cfg.PIXEL_MEANS, prefix, sample_dir)

        save_conv(ims, db2_add, cfg.PIXEL_MEANS, prefix, "_db", sample_dir)
        save_conv(ims, conv5_3, cfg.PIXEL_MEANS, prefix, "_con53", sample_dir)



def save_im(ims, pixel_means, prefix, output_dir):
    batch_size, _, _, _ = ims.shape
    for b in range(batch_size):
        im = ims[b, :, :, :].copy()
        channel_swap = (1, 2, 0)
        im = im.transpose(channel_swap)
        im += pixel_means
        im = im.astype(np.uint8)
        file_name = os.path.join(output_dir, prefix + '_b_' + str(b) + '.png')
        cv2.imwrite(file_name, im)


def save_conv(ims, convs, pixel_means, prefix, suffix, output_dir):
    feature_map_combination = []
    batch_size, channel, _, _ = convs.shape
    h, w = ims.shape[2:4]
    for b in range(batch_size):
        conv = convs[b, :, :, :].copy()
        channel_swap = (1, 2, 0)
        conv = conv.transpose(channel_swap)
        for c in range(channel):
            feature_map_combination.append(conv[:, :, c])
        feature_map_sum = sum(ele for ele in feature_map_combination)
        max_value = np.max(feature_map_sum)
        if max_value > 0:
            feature_map_sum = feature_map_sum / max_value * 255
        feature_map_sum = feature_map_sum.astype(np.uint8)
        im_color = cv2.applyColorMap(feature_map_sum, cv2.COLORMAP_JET)
        conv_img = cv2.resize(im_color, (w, h))
        cv2.imwrite(os.path.join(output_dir, prefix + '_b_' + str(b) + suffix + '.png'), conv_img)
        # save original im
        im = ims[b, :, :, :].copy()
        channel_swap = (1, 2, 0)
        im = im.transpose(channel_swap)
        im += pixel_means
        im = im.astype(np.uint8)
        im_file_name = os.path.join(output_dir, prefix + '_b_' + str(b) + '.png')
        cv2.imwrite(im_file_name, im)




def save_cpg(cpgs, labels_oh, prefix, output_dir):
    batch_size, num_classes, _, _ = cpgs.shape
    for b in range(batch_size):
        for c in range(num_classes):
            if labels_oh[b][c] == 0.0:
                continue
            cpg = cpgs[b, c, :, :].copy()
            max_value = np.max(cpg)
            if max_value > 0:
                cpg = cpg / max_value * 255
            cpg = cpg.astype(np.uint8)
            im_color = cv2.applyColorMap(cpg, cv2.COLORMAP_JET)
            file_name = os.path.join(
                output_dir,
                prefix + '_b_' + str(b) + '_c_' + str(c) + '_cpg.png')
            cv2.imwrite(file_name, im_color)


def save_common(datas, labels_oh, prefix, suffix, output_dir):
    if datas is None:
        return
    if len(datas.shape) == 3:
        datas = datas[np.newaxis, :]
    batch_size, num_classes, _, _ = datas.shape
    # print(datas.shape, labels_oh.shape)
    for b in range(batch_size):
        for c in range(num_classes):
            if labels_oh[b][c] == 0.0 and num_classes > 1:
                continue
            data = datas[b, c, :, :].copy()
            data = data * 255
            data = data.astype(np.uint8)
            im_color = cv2.applyColorMap(data, cv2.COLORMAP_JET)
            file_name = os.path.join(
                output_dir, prefix + '_b_' + str(b) + '_c_' + str(c) + '_' +
                suffix + '.png')
            cv2.imwrite(file_name, im_color)


def save_sigmoid(datas, labels_oh, prefix, suffix, output_dir):
    batch_size, num_classes, _, _ = datas.shape
    for b in range(batch_size):
        for c in range(num_classes):
            if labels_oh[b][c] == 0.0:
                continue
            data = datas[b, c, :, :].copy()
            data = np.reciprocal(1 + np.exp(-data))
            data = data * 255
            data = data.astype(np.uint8)
            im_color = cv2.applyColorMap(data, cv2.COLORMAP_JET)
            file_name = os.path.join(
                output_dir, prefix + '_b_' + str(b) + '_c_' + str(c) + '_' +
                suffix + '.png')
            cv2.imwrite(file_name, im_color)


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


def drawline(img, pt1, pt2, color, thickness=1, style='dotted', gap=20):
    dist = ((pt1[0] - pt2[0])**2 + (pt1[1] - pt2[1])**2)**.5
    pts = []
    for i in np.arange(0, dist, gap):
        r = i / dist
        x = int((pt1[0] * (1 - r) + pt2[0] * r) + .5)
        y = int((pt1[1] * (1 - r) + pt2[1] * r) + .5)
        p = (x, y)
        pts.append(p)

    if len(pts) == 0:
        return

    if style == 'dotted':
        for p in pts:
            cv2.circle(img, p, thickness, color, -1)
    else:
        s = pts[0]
        e = pts[0]
        i = 0
        for p in pts:
            s = e
            e = p
            if i % 2 == 1:
                cv2.line(img, s, e, color, thickness)
            i += 1


def drawpoly(
        img,
        pts,
        color,
        thickness=1,
        style='dotted',
):
    s = pts[0]
    e = pts[0]
    pts.append(pts.pop(0))
    for p in pts:
        s = e
        e = p
        drawline(img, s, e, color, thickness, style)


def drawrect(img, pt1, pt2, color, thickness=1, style='dotted'):
    pts = [pt1, (pt2[0], pt1[1]), pt2, (pt1[0], pt2[1])]
    drawpoly(img, pts, color, thickness, style)
