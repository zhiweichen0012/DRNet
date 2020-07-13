from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from detectron.core.config import cfg
from detectron.utils.net import get_group_gn

from detectron.modeling.ResNet import basic_bn_stem
from detectron.modeling.ResNet import basic_gn_stem
from detectron.modeling.ResNet import basic_bn_shortcut
from detectron.modeling.ResNet import basic_gn_shortcut

# ---------------------------------------------------------------------------- #
# Bits for specific architectures (ResNet50, ResNet101, ...)
# ---------------------------------------------------------------------------- #


def add_WSODNet_conv4_body(model):
    return add_shallow_WSODNet_convX_body(model, (2, 2, 2))


def add_WSODNet_conv5_body(model):
    return add_shallow_WSODNet_convX_body(model, (2, 2, 2, 2))


# ---------------------------------------------------------------------------- #
# Generic ResNet components
# ---------------------------------------------------------------------------- #


def add_stage(
    model,
    prefix,
    blob_in,
    n,
    dim_in,
    dim_out,
    dim_inner,
    dilation,
    stride_init=2
):
    """Add a ResNet stage to the model by stacking n residual blocks."""
    # e.g., prefix = res2
    for i in range(n):
        blob_in = add_residual_block(
            model,
            '{}_{}'.format(prefix, i),
            blob_in,
            dim_in,
            dim_out,
            dim_inner,
            dilation,
            stride_init,
            # Not using inplace for the last block;
            # it may be fetched externally or used by FPN
            inplace_sum=i < n - 1
        )
        dim_in = dim_out
    return blob_in, dim_in


def add_shallow_WSODNet_convX_body(model, block_counts):
    """Add a ResNet body from input data up through the res5 (aka conv5) stage.
    The final res5/conv5 stage may be optionally excluded (hence convX, where
    X = 4 or 5)."""
    freeze_at = cfg.TRAIN.FREEZE_AT
    assert freeze_at in [0, 2, 3, 4, 5]

    # add the stem (by default, conv1 and pool1 with bn; can support gn)
    p, dim_in = globals()[cfg.RESNETS.STEM_FUNC](model, 'data')

    stages = []
    stris = []
    dims = []

    (n1, n2, n3) = block_counts[:3]
    s, dim_in = add_stage(model, 'res2', p, n1, dim_in, 64, 0, 1)
    if freeze_at == 2:
        model.StopGradient(s, s)
    s, dim_in = add_stage(
        model, 'res3', s, n2, dim_in, 128, 0, 1
    )

    stages.append(s)
    stris.append(1. / 8.)
    dims.append(128)

    if freeze_at == 3:
        model.StopGradient(s, s)
    s, dim_in = add_stage(
        model, 'res4', s, n3, dim_in, 256, 0, 1
    )

    stages.append(s)
    stris.append(1. / 16.)
    dims.append(256)

    if freeze_at == 4:
        model.StopGradient(s, s)
    if len(block_counts) == 4:
        n4 = block_counts[3]
        s, dim_in = add_stage(
            model, 'res5', s, n4, dim_in, 512, 0,
            cfg.RESNETS.RES5_DILATION
        )

        stages.append(s)
        stris.append(1. / 32.)
        dims.append(512)

        stages, dims, stris = add_shallow_WSODNet_convX_boost(model, block_counts, stages, dims, stris)


        if freeze_at == 5:
            model.StopGradient(s, s)
        # return s, dim_in, 1. / 32. * cfg.RESNETS.RES5_DILATION
        return stages, dims, stris
    else:
        # return s, dim_in, 1. / 16.
        return stages, dims, stris


def add_shallow_WSODNet_convX_boost(model, block_counts, stages, dims, stris):
    weight_init = ("MSRAFill", {})

    new_feats = []

    cnt_feat = len(stages)
    for i in range(cnt_feat):
        feats = []

        for j in range(cnt_feat):
            if i < j:
                # conv 3x3 -> BN -> ReLU
                cur = model.ConvAffine(
                    stages[j],
                    str(j) + '_' + str(i),
                    dims[j],
                    dims[i],
                    kernel=1,
                    stride=1,
                    pad=0,
                    inplace=True,
                    weight_init=weight_init,
                )
                cur = model.net.UpsampleBilinearWSL(
                    [cur, stages[i]],
                    cur + '_up',
                )

                feats.append(cur)
            elif i > j:
                feat_in = stages[j]
                for k in range(i - j):
                    if k == i - j -1:
                        # conv 3x3 -> BN -> ReLU
                        cur = model.ConvAffine(
                            feat_in,
                            str(j) + '_' + str(i),
                            dims[j],
                            dims[i],
                            kernel=3,
                            stride=2,
                            pad=1,
                            inplace=True,
                            weight_init=weight_init,
                        )
                        feats.append(cur)
                    else:
                        # conv 3x3 -> BN -> ReLU
                        cur = model.ConvAffine(
                            feat_in,
                            str(j) + '_' + str(i) + '__' + str(k + j + 1),
                            dims[j],
                            dims[j],
                            kernel=3,
                            stride=2,
                            pad=1,
                            inplace=True,
                            weight_init=weight_init,
                        )
                        cur = model.Relu(cur, cur)

                        feat_in = cur
            else:
                # conv 3x3 -> BN -> ReLU
                cur = model.ConvAffine(
                    stages[i],
                    str(j) + '_' + str(i),
                    dims[j],
                    dims[i],
                    kernel=1,
                    stride=1,
                    pad=0,
                    inplace=True,
                    weight_init=weight_init,
                )
                feats.append(cur)

        cur = model.net.Sum(feats, 'fuse_' + str(i))
        cur = model.Relu(cur, cur)

        blob_in = add_residual_block(
            model,
            'fuse_{}'.format(i),
            cur,
            dims[i],
            dims[i],
            0,
            1,
            1,
            inplace_sum=True,
        )

        new_feats.append(blob_in)

    return new_feats, dims, stris



def add_WSODNet_roi_conv5_head(model, blob_in, dim_in, spatial_scale):
    """Adds an RoI feature transformation (e.g., RoI pooling) followed by a
    res5/conv5 head applied to each RoI."""
    # TODO(rbg): This contains Fast R-CNN specific config options making it non-
    # reusable; make this more generic with model-specific wrappers
    model.RoIFeatureTransform(
        blob_in,
        'pool5',
        blob_rois='rois',
        method=cfg.FAST_RCNN.ROI_XFORM_METHOD,
        resolution=cfg.FAST_RCNN.ROI_XFORM_RESOLUTION,
        sampling_ratio=cfg.FAST_RCNN.ROI_XFORM_SAMPLING_RATIO,
        spatial_scale=spatial_scale
    )

    stride_init = int(cfg.FAST_RCNN.ROI_XFORM_RESOLUTION / 7)
    s, dim_in = add_stage(
        model, 'res5', 'pool5', 2, dim_in, 512, 0, 1,
        stride_init
    )
    s = model.AveragePool(s, 'res5_pool', kernel=7)
    return s, 512


def add_residual_block(
    model,
    prefix,
    blob_in,
    dim_in,
    dim_out,
    dim_inner,
    dilation,
    stride_init=2,
    inplace_sum=False
):
    """Add a residual block to the model."""
    # prefix = res<stage>_<sub_stage>, e.g., res2_3

    # Max pooling is performed prior to the first stage (which is uniquely
    # distinguished by dim_in = 64), thus we keep stride = 1 for the first stage
    stride = stride_init if (
        dim_in != dim_out and 'res2' not in prefix and dilation == 1
    ) else 1

    # transformation blob
    # tr = globals()[cfg.RESNETS.TRANS_FUNC](
    tr = residual_transformation(
        model,
        blob_in,
        dim_in,
        dim_out,
        stride,
        prefix,
        dim_inner,
        group=cfg.RESNETS.NUM_GROUPS,
        dilation=dilation
    )

    # sum -> ReLU
    # shortcut function: by default using bn; support gn
    add_shortcut = globals()[cfg.RESNETS.SHORTCUT_FUNC]
    sc = add_shortcut(model, prefix, blob_in, dim_in, dim_out, stride)
    if inplace_sum:
        s = model.net.Sum([tr, sc], tr)
    else:
        s = model.net.Sum([tr, sc], prefix + '_sum')

    return model.Relu(s, s)


# ------------------------------------------------------------------------------
# various stems (may expand and may consider a new helper)
# ------------------------------------------------------------------------------


def basic_bn_stem_nouse(model, data, **kwargs):
    """Add a basic ResNet stem. For a pre-trained network that used BN.
    An AffineChannel op replaces BN during fine-tuning.
    """

    weight_init = ("MSRAFill", {})
    dim = 64
    p = model.Conv('data', 'conv1_1', 3, dim, 3, pad=1, stride=1, no_bias=1, weight_init=weight_init)
    p = model.AffineChannel(p, 'conv1_1_bn', dim=dim, inplace=True)
    p = model.Relu(p, p)
    p = model.MaxPool(p, 'pool1_1', kernel=2, pad=0, stride=2)
    p = model.Conv( p, 'conv1_2', dim, dim, 3, pad=1, stride=1, no_bias=1, weight_init=weight_init)
    p = model.AffineChannel(p, 'conv1_2_bn', dim=dim, inplace=True)
    p = model.Relu(p, p)
    p = model.MaxPool(p, 'pool1_2', kernel=2, pad=0, stride=2)
    return p, dim

    weight_init = ("MSRAFill", {})
    dim = 64
    p = model.Conv(data, 'conv1', 3, dim, 7, pad=3, stride=2, no_bias=1, weight_init=weight_init)
    p = model.AffineChannel(p, 'res_conv1_bn', dim=dim, inplace=True)
    p = model.Relu(p, p)
    p = model.MaxPool(p, 'pool1', kernel=3, pad=1, stride=2)
    return p, dim


def basic_gn_stem_nouse(model, data, **kwargs):
    """Add a basic ResNet stem (using GN)"""

    weight_init = ("MSRAFill", {})
    dim = 64
    p = model.ConvGN('data', 'conv1_1', 3, dim, 3,group_gn=get_group_gn(dim), pad=1, stride=1, weight_init=weight_init)
    p = model.Relu(p, p)
    p = model.ConvGN(p, 'conv1_2', dim, dim, 3,group_gn=get_group_gn(dim), pad=1, stride=1, weight_init=weight_init)
    p = model.Relu(p, p)
    p = model.MaxPool(p, 'pool1', kernel=2, pad=0, stride=2)
    return p, dim

    dim = 64
    p = model.ConvGN(
        data, 'conv1', 3, dim, 7, group_gn=get_group_gn(dim), pad=3, stride=2
    )
    p = model.Relu(p, p)
    p = model.MaxPool(p, 'pool1', kernel=3, pad=1, stride=2)
    return p, dim


# ------------------------------------------------------------------------------
# various transformations (may expand and may consider a new helper)
# ------------------------------------------------------------------------------


def residual_transformation(
        model,
        blob_in,
        dim_in,
        dim_out,
        stride,
        prefix,
        dim_inner,
        dilation=1,
        group=1,
):
    """Add a bottleneck transformation to the model."""

    weight_init = ("MSRAFill", {})

    # conv 3x3 -> BN -> ReLU
    cur = model.ConvAffine(
        blob_in,
        prefix + '_branch2a',
        dim_in,
        dim_out,
        kernel=3,
        stride=stride,
        pad=1,
        inplace=True,
        weight_init=weight_init,
    )
    cur = model.Relu(cur, cur)

    # conv 3x3 -> BN -> ReLU
    cur = model.ConvAffine(
        cur,
        prefix + '_branch2b',
        dim_out,
        dim_out,
        kernel=3,
        stride=1,
        pad=1 * dilation,
        dilation=dilation,
        group=group,
        inplace=False,
        weight_init=weight_init,
    )

    return cur


def add_WSODNet_2fc_head(model, blob_in, dim_in, spatial_scale, dim_fc=4096):
    feats =[]
    print(spatial_scale)
    print(blob_in)
    print(dim_in)
    for i in range(len(blob_in)):
        k = int(32. * spatial_scale[i])
        if k == 1:
            cur = blob_in[i]
        else:
            cur = model.MaxPool(blob_in[i], blob_in[i] + '_pool', kernel=k, pad=0, stride=k)
        feats.append(cur)

    pool5, _ = model.net.Concat(feats, ['pool5', 'pool5_concat_dims'], axis=1)

    # pool5 = model.MaxPool(blob_in, 'pool5', kernel=2, pad=0, stride=2)

    weight_init = ('GaussianFill', {'std': 0.0005})
    # weight_init = ('XavierFill', {})
    # weight_init = ("MSRAFill", {})

    bias_init = ('ConstantFill', {'value': 0.01})

    fc6 = model.FC(
        pool5,
        'fc6',
        sum(dim_in) * 7 * 7,
        dim_fc,
        weight_init=weight_init,
        bias_init=bias_init)
    relu6 = model.Relu(fc6, 'fc6')
    drop6 = model.Dropout(relu6, 'drop6', ratio=0.5, is_test=not model.train)

    fc7 = model.FC(
        drop6,
        'fc7',
        dim_fc,
        dim_fc,
        weight_init=weight_init,
        bias_init=bias_init,
    )
    relu7 = model.Relu(fc7, 'fc7')
    drop7 = model.Dropout(relu7, 'drop7', ratio=0.5, is_test=not model.train)
    return drop7, dim_fc, spatial_scale[-1] / 2.
