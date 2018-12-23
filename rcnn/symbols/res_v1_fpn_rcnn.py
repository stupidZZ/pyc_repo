import cPickle
import mxnet as mx
import numpy as np
from common.lib.utils.symbol import Symbol
from common.backbone import resnet_v1
from common.operator_py.proposal_target import *
from common.operator_py.pad_like import *
from common.gpu_metric import *

class res_v1_fpn_rcnn(Symbol):
    def __init__(self, FP16=False):
        """
        Use __init__ to define parameter network needs
        """
        self.shared_param_list = ['rpn_conv_3x3', 'rpn_cls_score', 'rpn_bbox_pred']
        self.shared_param_dict = {}
        self.FP16 = FP16
        self.dtype = np.float16 if FP16 else np.float32

        for name in self.shared_param_list:
            self.shared_param_dict[name + '_weight'] = mx.sym.Variable(name + '_weight', dtype=self.dtype)
            self.shared_param_dict[name + '_bias'] = mx.sym.Variable(name + '_bias', dtype=self.dtype)
            self.shared_param_dict[name + '_offset_weight'] = mx.sym.Variable(name + '_offset_weight', dtype=self.dtype)
            self.shared_param_dict[name + '_offset_bias'] = mx.sym.Variable(name + '_offset_bias', dtype=self.dtype)
        # allocate roi offset
        self.shared_param_dict['roi_offset_weight'] = mx.sym.Variable('roi_offset_weight', dtype=self.dtype)
        self.shared_param_dict['roi_offset_bias'] = mx.sym.Variable('roi_offset_bias', dtype=self.dtype)


    def deformable_conv(self, data, name, num_filter, stride, lr_mult, param_dict=None, prefix=''):
        if param_dict is not None:
            conv_offset_weight = param_dict[name + '_offset_weight']
            conv_offset_bias = param_dict[name + '_offset_bias']
            conv_weight = param_dict[name + '_weight']
            conv_bias = param_dict[name + '_bias']
        else:
            conv_offset_weight = mx.sym.Variable(name=name + '_offset_weight', lr_mult=lr_mult, dtype=self.dtype)
            conv_offset_bias = mx.sym.Variable(name=name + '_offset_bias', lr_mult=lr_mult, dtype=self.dtype)
            conv_weight = mx.sym.Variable(name=name + '_weight', lr_mult=lr_mult, dtype=self.dtype)
            conv_bias = mx.sym.Variable(name=name + '_bias', lr_mult=lr_mult, dtype=self.dtype)

        conv_offset = mx.symbol.Convolution(name=prefix + name + '_offset', data=data, num_filter=18, pad=(1, 1),
                                            kernel=(3, 3), stride=stride,
                                            weight=conv_offset_weight, bias=conv_offset_bias)
        conv_new = mx.contrib.symbol.DeformableConvolution(name=prefix + name, data=data, offset=conv_offset,
                                                           num_filter=num_filter, pad=(1, 1), kernel=(3, 3),
                                                           stride=stride,
                                                           num_deformable_group=1, no_bias=False,
                                                           weight=conv_weight, bias=conv_bias,
                                                           max_compute_batchsize=1024)
        return conv_new

    def get_fpn_feature(self, c2, c3, c4, c5, feature_dim, use_dconv, dconv_lr_mult):
        # lateral connection
        fpn_p5_1x1 = mx.symbol.Convolution(data=c5, kernel=(1, 1), pad=(0, 0), stride=(1, 1), num_filter=feature_dim, name='fpn_p5_1x1')
        fpn_p4_1x1 = mx.symbol.Convolution(data=c4, kernel=(1, 1), pad=(0, 0), stride=(1, 1), num_filter=feature_dim, name='fpn_p4_1x1')
        fpn_p3_1x1 = mx.symbol.Convolution(data=c3, kernel=(1, 1), pad=(0, 0), stride=(1, 1), num_filter=feature_dim, name='fpn_p3_1x1')
        fpn_p2_1x1 = mx.symbol.Convolution(data=c2, kernel=(1, 1), pad=(0, 0), stride=(1, 1), num_filter=feature_dim, name='fpn_p2_1x1')
        # top-down connection
        fpn_p5_upsample = mx.symbol.UpSampling(fpn_p5_1x1, scale=2, sample_type='nearest', name='fpn_p5_upsample')
        fpn_p4_plus = mx.sym.ElementWiseSum(*[fpn_p5_upsample, fpn_p4_1x1], name='fpn_p4_sum')
        fpn_p4_upsample = mx.symbol.UpSampling(fpn_p4_plus, scale=2, sample_type='nearest', name='fpn_p4_upsample')
        fpn_p3_plus = mx.sym.ElementWiseSum(*[fpn_p4_upsample, fpn_p3_1x1], name='fpn_p3_sum')
        fpn_p3_upsample = mx.symbol.UpSampling(fpn_p3_plus, scale=2, sample_type='nearest', name='fpn_p3_upsample')
        fpn_p2_plus = mx.sym.ElementWiseSum(*[fpn_p3_upsample, fpn_p2_1x1], name='fpn_p2_sum')
        # FPN feature
        if use_dconv:
            fpn_p6 = self.deformable_conv(c5, 'fpn_p6', feature_dim, stride=(2, 2), lr_mult=dconv_lr_mult)
            fpn_p4 = self.deformable_conv(fpn_p4_plus, 'fpn_p4', feature_dim, stride=(1, 1), lr_mult=dconv_lr_mult)
            fpn_p3 = self.deformable_conv(fpn_p3_plus, 'fpn_p3', feature_dim, stride=(1, 1), lr_mult=dconv_lr_mult)
            fpn_p2 = self.deformable_conv(fpn_p2_plus, 'fpn_p2', feature_dim, stride=(1, 1), lr_mult=dconv_lr_mult)
        else:
            fpn_p6 = mx.symbol.Convolution(data=c5, kernel=(3, 3), pad=(1, 1), stride=(2, 2), num_filter=feature_dim, name='fpn_p6')
            fpn_p5 = mx.symbol.Convolution(data=fpn_p5_1x1, kernel=(3, 3), pad=(1, 1), stride=(1, 1), num_filter=feature_dim, name='fpn_p5')
            fpn_p4 = mx.symbol.Convolution(data=fpn_p4_plus, kernel=(3, 3), pad=(1, 1), stride=(1, 1), num_filter=feature_dim, name='fpn_p4')
            fpn_p3 = mx.symbol.Convolution(data=fpn_p3_plus, kernel=(3, 3), pad=(1, 1), stride=(1, 1), num_filter=feature_dim, name='fpn_p3')
            fpn_p2 = mx.symbol.Convolution(data=fpn_p2_plus, kernel=(3, 3), pad=(1, 1), stride=(1, 1), num_filter=feature_dim, name='fpn_p2')

        return fpn_p2, fpn_p3, fpn_p4, fpn_p5, fpn_p6

    def get_rpn_subnet(self, data, num_anchors, use_dconv, dconv_lr_mult, suffix):
        if use_dconv:
            rpn_conv = self.deformable_conv(data=data, num_filter=512, stride=1, lr_mult=dconv_lr_mult,
                                            param_dict=self.shared_param_dict, name='rpn_conv_3x3', prefix=suffix+'_')
        else:
            rpn_conv = mx.sym.Convolution(data=data, kernel=(3, 3), pad=(1, 1), num_filter=512, name='rpn_conv_'+suffix,
                                          weight=self.shared_param_dict['rpn_conv_3x3_weight'], bias=self.shared_param_dict['rpn_conv_3x3_bias'])

        rpn_relu = mx.sym.Activation(data=rpn_conv, act_type='relu', name='rpn_relu_' + suffix)
        rpn_cls_score = mx.sym.Convolution(data=rpn_relu, kernel=(1, 1), pad=(0, 0), num_filter=2 * num_anchors, name='rpn_cls_score_' + suffix,
                                           weight=self.shared_param_dict['rpn_cls_score_weight'], bias=self.shared_param_dict['rpn_cls_score_bias'])
        rpn_bbox_pred = mx.sym.Convolution(data=rpn_relu, kernel=(1, 1), pad=(0, 0), num_filter=4 * num_anchors, name='rpn_bbox_pred_' + suffix,
                                           weight=self.shared_param_dict['rpn_bbox_pred_weight'], bias=self.shared_param_dict['rpn_bbox_pred_bias'])

        # n x (2*A) x H x W => n x 2 x (A*H*W)
        rpn_cls_score_t1 = mx.sym.Reshape(data=rpn_cls_score, shape=(0, 2, -1, 0), name='rpn_cls_score_t1_' + suffix)
        rpn_cls_score_t2 = mx.sym.Reshape(data=rpn_cls_score_t1, shape=(0, 2, -1), name='rpn_cls_score_t2_' + suffix)
        if self.FP16:
            rpn_cls_score_t1 = mx.sym.Cast(rpn_cls_score_t1, dtype=np.float32)
        rpn_cls_prob = mx.sym.SoftmaxActivation(data=rpn_cls_score_t1, mode='channel', name='rpn_cls_prob_' + suffix)
        if self.FP16:
            rpn_cls_prob = mx.sym.Cast(rpn_cls_prob, dtype=np.float16)
        rpn_cls_prob_t = mx.sym.Reshape(data=rpn_cls_prob, shape=(0, 2 * num_anchors, -1, 0), name='rpn_cls_prob_t_' + suffix)
        rpn_bbox_pred_t = mx.sym.Reshape(data=rpn_bbox_pred, shape=(0, 0, -1), name='rpn_bbox_pred_t_' + suffix)
        return rpn_cls_score_t2, rpn_cls_prob_t, rpn_bbox_pred_t, rpn_bbox_pred

    def fpn_roi_pool_sym(self, rois, data_p2, data_p3, data_p4, data_p5, roi_size=7):
        rois_xmin = mx.sym.slice_axis(data=rois, axis=1, begin=1, end=2)
        rois_ymin = mx.sym.slice_axis(data=rois, axis=1, begin=2, end=3)
        rois_xmax = mx.sym.slice_axis(data=rois, axis=1, begin=3, end=4)
        rois_ymax = mx.sym.slice_axis(data=rois, axis=1, begin=4, end=5)
        rois_w = rois_xmax - rois_xmin + 1
        rois_h = rois_ymax - rois_ymin + 1

        stage_id = mx.sym.clip(mx.sym.BlockGrad(mx.sym.floor(2 + mx.sym.log2(mx.sym.sqrt(rois_w * rois_h) / 224.0))), 0, 3)
        rois_all = mx.sym.Concat(stage_id, rois)

        roi_pool = mx.contrib.sym.DeformablePSROIPoolingv2(*[data_p2, data_p3, data_p4, data_p5, rois_all],
                                                           name='roi_align_' + str(roi_size),
                                                           group_size=1,
                                                           pooled_size=roi_size, sample_per_part=2,
                                                           no_trans=True, part_size=roi_size, output_dim=256,
                                                           spatial_scale=(0.25, 0.125, 0.0625, 0.03125))
        return roi_pool

    def get_symbol(self, cfg, is_train=True):
        # config alias for convenient
        num_classes = cfg.dataset.num_classes
        num_reg_classes = (2 if cfg.class_agnostic else num_classes)
        num_anchors = cfg.network.NUM_ANCHORS
        grad_scale = cfg.TRAIN.FP16_GRAD_SCALE if self.FP16 else 1.0

        if is_train:
            data = mx.sym.Variable(name="data")
            im_info = mx.sym.Variable(name="im_info")
            gt_boxes = mx.sym.Variable(name="gt_boxes")
            rpn_label = mx.sym.Variable(name='label')
            rpn_bbox_target = mx.sym.Variable(name='bbox_target')
            rpn_bbox_weight = mx.sym.Variable(name='bbox_weight')
            if self.FP16:
                data = mx.sym.Cast(data, dtype=np.float16)
                im_info = mx.sym.Cast(im_info, dtype=np.float16)
                gt_boxes = mx.sym.Cast(gt_boxes, dtype=np.float16)
                rpn_bbox_target = mx.sym.Cast(rpn_bbox_target, dtype=np.float16)
                rpn_bbox_weight = mx.sym.Cast(rpn_bbox_weight, dtype=np.float16)
        else:
            data = mx.sym.Variable(name="data")
            im_info = mx.sym.Variable(name="im_info")


        assert cfg.network.use_dilation_on_c5 == False, "fpn should keep use_dilation_on_c5 to be False"
        _, c2, c3, c4, c5 = resnet_v1.get_resnet_backbone(data=data,
                                                          num_layers=cfg.network.num_layers,
                                                          use_dilation_on_c5=cfg.network.use_dilation_on_c5,
                                                          use_dconv=cfg.network.backbone_use_dconv,
                                                          dconv_lr_mult=cfg.network.backbone_dconv_lr_mult,
                                                          dconv_group=cfg.network.backbone_dconv_group,
                                                          dconv_start_channel=cfg.network.backbone_dconv_start_channel)

        fpn_p2, fpn_p3, fpn_p4, fpn_p5, fpn_p6 = self.get_fpn_feature(c2, c3, c4, c5,
                                                                      feature_dim=cfg.FPN.feature_dim,
                                                                      use_dconv=cfg.FPN.use_dconv,
                                                                      dconv_lr_mult=cfg.FPN.dconv_lr_mult)

        rpn_cls_score_p2, rpn_prob_p2, rpn_bbox_loss_p2, rpn_bbox_pred_p2 = self.get_rpn_subnet(data=fpn_p2,
                                                                                                num_anchors=num_anchors,
                                                                                                use_dconv=cfg.FPN.use_dconv,
                                                                                                dconv_lr_mult=cfg.FPN.dconv_lr_mult,
                                                                                                suffix='p2')

        rpn_cls_score_p3, rpn_prob_p3, rpn_bbox_loss_p3, rpn_bbox_pred_p3 = self.get_rpn_subnet(data=fpn_p3,
                                                                                                num_anchors=num_anchors,
                                                                                                use_dconv=cfg.FPN.use_dconv,
                                                                                                dconv_lr_mult=cfg.FPN.dconv_lr_mult,
                                                                                                suffix='p3')

        rpn_cls_score_p4, rpn_prob_p4, rpn_bbox_loss_p4, rpn_bbox_pred_p4 = self.get_rpn_subnet(data=fpn_p4,
                                                                                                num_anchors=num_anchors,
                                                                                                use_dconv=cfg.FPN.use_dconv,
                                                                                                dconv_lr_mult=cfg.FPN.dconv_lr_mult,
                                                                                                suffix='p4')

        rpn_cls_score_p5, rpn_prob_p5, rpn_bbox_loss_p5, rpn_bbox_pred_p5 = self.get_rpn_subnet(data=fpn_p5,
                                                                                                num_anchors=num_anchors,
                                                                                                use_dconv=cfg.FPN.use_dconv,
                                                                                                dconv_lr_mult=cfg.FPN.dconv_lr_mult,
                                                                                                suffix='p5')

        rpn_cls_score_p6, rpn_prob_p6, rpn_bbox_loss_p6, rpn_bbox_pred_p6 = self.get_rpn_subnet(data=fpn_p6,
                                                                                                num_anchors=num_anchors,
                                                                                                use_dconv=cfg.FPN.use_dconv,
                                                                                                dconv_lr_mult=cfg.FPN.dconv_lr_mult,
                                                                                                suffix='p6')

        rpn_cls_score = mx.sym.Concat(rpn_cls_score_p2, rpn_cls_score_p3, rpn_cls_score_p4, rpn_cls_score_p5, rpn_cls_score_p6, dim=2)
        rpn_bbox_pred = mx.sym.Concat(rpn_bbox_loss_p2, rpn_bbox_loss_p3, rpn_bbox_loss_p4, rpn_bbox_loss_p5, rpn_bbox_loss_p6, dim=2)
        if is_train:
            # classification
            if self.FP16:
                rpn_cls_score = mx.sym.Cast(rpn_cls_score, dtype=np.float32)
                rpn_cls_prob = mx.sym.SoftmaxOutput(data=rpn_cls_score, label=rpn_label, multi_output=True,
                                                   normalization='valid', use_ignore=True, ignore_label=-1, name="rpn_cls_prob",grad_scale=grad_scale)
            else:
                rpn_cls_prob = mx.sym.SoftmaxOutput(data=rpn_cls_score, label=rpn_label, multi_output=True,
                                                    normalization='valid', use_ignore=True, ignore_label=-1, name="rpn_cls_prob")
            # bounding box regression
            rpn_bbox_loss_ = rpn_bbox_weight * mx.sym.smooth_l1(name='rpn_bbox_loss_', scalar=3.0, data=(rpn_bbox_pred - rpn_bbox_target))
            rpn_bbox_loss = mx.sym.MakeLoss(name='rpn_bbox_loss', data=rpn_bbox_loss_, grad_scale=grad_scale * 1.0 / (cfg.TRAIN.RPN_BATCH_SIZE * cfg.TRAIN.BATCH_IMAGES))

            rois = mx.contrib.sym.MultiPyramidProposal(im_info=im_info,
                                                         rpn_cls_prob_stride0=rpn_prob_p2,
                                                         rpn_bbox_pred_stride0=rpn_bbox_pred_p2,
                                                         rpn_cls_prob_stride1=rpn_prob_p3,
                                                         rpn_bbox_pred_stride1=rpn_bbox_pred_p3,
                                                         rpn_cls_prob_stride2=rpn_prob_p4,
                                                         rpn_bbox_pred_stride2=rpn_bbox_pred_p4,
                                                         rpn_cls_prob_stride3=rpn_prob_p5,
                                                         rpn_bbox_pred_stride3=rpn_bbox_pred_p5,
                                                         rpn_cls_prob_stride4=rpn_prob_p6,
                                                         rpn_bbox_pred_stride4=rpn_bbox_pred_p6,
                                                         name='rois',
                                                         feature_stride=tuple(cfg.network.RPN_FEAT_STRIDE),
                                                         scales=tuple(cfg.network.ANCHOR_SCALES),
                                                         ratios=tuple(cfg.network.ANCHOR_RATIOS),
                                                         rpn_pre_nms_top_n=cfg.TRAIN.RPN_PRE_NMS_TOP_N,
                                                         rpn_post_nms_top_n=cfg.TRAIN.RPN_POST_NMS_TOP_N,
                                                         threshold=cfg.TRAIN.RPN_NMS_THRESH,
                                                         rpn_min_size=cfg.TRAIN.RPN_MIN_SIZE)

            gt_boxes_reshape = mx.sym.Reshape(data=gt_boxes, shape=(-1, 6), name='gt_boxes_reshape')

            rois, label, bbox_target, bbox_weight = mx.sym.Custom(rois=rois, gt_boxes=gt_boxes_reshape,
                                                                  op_type='proposal_target',
                                                                  num_classes=num_reg_classes,
                                                                  batch_images=cfg.TRAIN.BATCH_IMAGES,
                                                                  batch_rois=cfg.TRAIN.BATCH_ROIS,
                                                                  cfg=cPickle.dumps(cfg),
                                                                  fg_fraction=cfg.TRAIN.FG_FRACTION)
        else:
            rois, rpn_score = mx.contrib.sym.MultiPyramidProposal(im_info=im_info,
                                                                    rpn_cls_prob_stride0=rpn_prob_p2,
                                                                    rpn_bbox_pred_stride0=rpn_bbox_pred_p2,
                                                                    rpn_cls_prob_stride1=rpn_prob_p3,
                                                                    rpn_bbox_pred_stride1=rpn_bbox_pred_p3,
                                                                    rpn_cls_prob_stride2=rpn_prob_p4,
                                                                    rpn_bbox_pred_stride2=rpn_bbox_pred_p4,
                                                                    rpn_cls_prob_stride3=rpn_prob_p5,
                                                                    rpn_bbox_pred_stride3=rpn_bbox_pred_p5,
                                                                    rpn_cls_prob_stride4=rpn_prob_p6,
                                                                    rpn_bbox_pred_stride4=rpn_bbox_pred_p6,
                                                                    name='rois', output_score=True,
                                                                    feature_stride=tuple(cfg.network.RPN_FEAT_STRIDE),
                                                                    scales=tuple(cfg.network.ANCHOR_SCALES),
                                                                    ratios=tuple(cfg.network.ANCHOR_RATIOS),
                                                                    rpn_pre_nms_top_n=cfg.TEST.RPN_PRE_NMS_TOP_N,
                                                                    rpn_post_nms_top_n=cfg.TEST.RPN_POST_NMS_TOP_N,
                                                                    threshold=cfg.TEST.RPN_NMS_THRESH,
                                                                    rpn_min_size=cfg.TEST.RPN_MIN_SIZE)

        rois_feat = self.fpn_roi_pool_sym(rois, fpn_p2, fpn_p3, fpn_p4, fpn_p5, roi_size=7)

        fc_new_1 = mx.symbol.FullyConnected(name='fc_new_1', data=rois_feat, num_hidden=1024)
        fc_new_1_relu = mx.sym.Activation(data=fc_new_1, act_type='relu', name='fc_new_1_relu')

        fc_new_2 = mx.symbol.FullyConnected(name='fc_new_2', data=fc_new_1_relu, num_hidden=1024)
        fc_new_2_relu = mx.sym.Activation(data=fc_new_2, act_type='relu', name='fc_new_2_relu')

        # cls_score/bbox_pred
        cls_score = mx.symbol.FullyConnected(name='cls_score', data=fc_new_2_relu, num_hidden=num_classes)
        bbox_pred = mx.symbol.FullyConnected(name='bbox_pred', data=fc_new_2_relu, num_hidden=num_reg_classes * 4)

        if is_train:
            if cfg.TRAIN.ENABLE_OHEM:
                labels_ohem, bbox_weights_ohem = mx.sym.Custom(op_type='BoxAnnotatorOHEM', num_classes=num_classes,
                                                               num_reg_classes=num_reg_classes,
                                                               roi_per_img=cfg.TRAIN.BATCH_ROIS_OHEM,
                                                               cls_score=cls_score, bbox_pred=bbox_pred, labels=label,
                                                               bbox_targets=bbox_target, bbox_weights=bbox_weight)
                if self.FP16:
                    cls_score = mx.sym.Cast(cls_score, dtype=np.float32)
                    labels_ohem = mx.sym.Cast(labels_ohem, dtype=np.float32)
                    cls_prob = mx.sym.SoftmaxOutput(name='cls_prob', data=cls_score, label=labels_ohem,
                                                    normalization='valid', use_ignore=True, ignore_label=-1,
                                                    grad_scale=grad_scale * cfg.network.FRCNN_LOSS_GRAD_SCALE)
                else:
                    cls_prob = mx.sym.SoftmaxOutput(name='cls_prob', data=cls_score, label=labels_ohem,
                                                    normalization='valid', use_ignore=True, ignore_label=-1,
                                                    grad_scale=cfg.network.FRCNN_LOSS_GRAD_SCALE)
                bbox_loss_ = bbox_weights_ohem * mx.sym.smooth_l1(name='bbox_loss_', scalar=1.0,
                                                                  data=(bbox_pred - bbox_target))
                bbox_loss = mx.sym.MakeLoss(name='bbox_loss', data=bbox_loss_,
                                            grad_scale=grad_scale * cfg.network.FRCNN_LOSS_GRAD_SCALE * 2.0 / cfg.TRAIN.BATCH_ROIS_OHEM)
                rcnn_label = labels_ohem
            else:
                rcnn_label = label
                if self.FP16:
                    cls_score = mx.sym.Cast(cls_score, dtype=np.float32)
                    rcnn_label = mx.sym.Cast(rcnn_label, dtype=np.float32)
                    cls_prob = mx.sym.SoftmaxOutput(name='cls_prob', data=cls_score, label=rcnn_label, normalization='valid',
                                                    grad_scale = grad_scale)
                else:
                    cls_prob = mx.sym.SoftmaxOutput(name='cls_prob', data=cls_score, label=rcnn_label, normalization='valid')
                bbox_loss_ = bbox_weight * mx.sym.smooth_l1(name='bbox_loss_', scalar=1.0,
                                                            data=(bbox_pred - bbox_target))
                bbox_loss = mx.sym.MakeLoss(name='bbox_loss', data=bbox_loss_,
                                            grad_scale=grad_scale * 2.0 / (cfg.TRAIN.BATCH_IMAGES * cfg.TRAIN.BATCH_ROIS))

            # reshape output
            rcnn_label = mx.sym.Reshape(data=rcnn_label, shape=(cfg.TRAIN.BATCH_IMAGES, -1), name='label_reshape')
            cls_prob = mx.sym.Reshape(data=cls_prob, shape=(cfg.TRAIN.BATCH_IMAGES, -1, num_classes),
                                      name='cls_prob_reshape')
            bbox_loss = mx.sym.Reshape(data=bbox_loss, shape=(cfg.TRAIN.BATCH_IMAGES, -1, 4 * num_reg_classes),
                                           name='bbox_loss_reshape')

            output_list = [rpn_cls_prob, rpn_bbox_loss, cls_prob, bbox_loss, mx.sym.BlockGrad(rcnn_label)]

            # get metric
            if cfg.TRAIN.GPU_METRIC:
                output_list.extend(get_rpn_acc(rpn_cls_prob, rpn_label))
                output_list.extend(get_rcnn_acc(cls_prob, rcnn_label))
                output_list.extend(get_rcnn_fg_acc(cls_prob, rcnn_label))
                output_list.extend(get_rpn_fg_fraction(rcnn_label))
                output_list.extend(get_rpn_logloss(rpn_cls_prob, rpn_label))
                output_list.extend(get_rcnn_logloss(cls_prob, rcnn_label, cfg))
                output_list.extend(get_rpn_l1loss(rpn_bbox_loss, rpn_label))
                output_list.extend(get_rcnn_l1loss(bbox_loss, rcnn_label))

            group = mx.sym.Group(output_list)
        else:
            cls_prob_ = mx.sym.SoftmaxActivation(name='cls_prob', data=cls_score)
            cls_prob = mx.sym.Reshape(data=cls_prob_, shape=(cfg.TEST.BATCH_IMAGES, -1, num_classes),
                                      name='cls_prob_reshape')
            bbox_pred = mx.sym.Reshape(data=bbox_pred, shape=(cfg.TEST.BATCH_IMAGES, -1, 4 * num_reg_classes),
                                       name='bbox_pred_reshape')
            group = mx.sym.Group([rois, rpn_score, cls_prob, bbox_pred])

        self.sym = group
        return group

    def get_pred_names(self, is_train, gpu_metric=False):
        if is_train:
            pred_names = ['rpn_cls_prob', 'rpn_bbox_loss', 'rcnn_cls_prob', 'rcnn_bbox_loss', 'rcnn_label']
            if gpu_metric:
                pred_names.extend([
                    'RPNAcc', 'RPNAccInstNum',
                    'RCNNAcc', 'RCNNAccInstNum',
                    'RCNNFgAcc', 'RCNNFgAccInstNum',
                    'RPNFgFrac', 'RPNFgFracInstNum',
                    'RPNLogLoss', 'RPNLogLossInstNum',
                    'RCNNLogLoss', 'RCNNLogLossInstNum',
                    'RPNL1Loss', 'RPNL1LossInstNum',
                    'RCNNL1Loss', 'RCNNL1LossInstNum'])
            return pred_names
        else:
            return ['rois', 'rcnn_cls_prob', 'rcnn_bbox_pred']

    def get_label_names(self):
        return ['rpn_label', 'rpn_bbox_target', 'rpn_bbox_weight']

    def init_weight_rcnn(self, cfg, arg_params, aux_params):
        arg_params['fc_new_1_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['fc_new_1_weight'])
        arg_params['fc_new_1_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['fc_new_1_bias'])
        arg_params['fc_new_2_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['fc_new_2_weight'])
        arg_params['fc_new_2_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['fc_new_2_bias'])
        arg_params['cls_score_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['cls_score_weight'])
        arg_params['cls_score_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['cls_score_bias'])
        arg_params['bbox_pred_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['bbox_pred_weight'])
        arg_params['bbox_pred_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['bbox_pred_bias'])

    def init_weight_rpn(self, cfg, arg_params, aux_params):
        arg_params['rpn_conv_3x3_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['rpn_conv_3x3_weight'])
        arg_params['rpn_conv_3x3_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['rpn_conv_3x3_bias'])
        arg_params['rpn_cls_score_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['rpn_cls_score_weight'])
        arg_params['rpn_cls_score_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['rpn_cls_score_bias'])
        arg_params['rpn_bbox_pred_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['rpn_bbox_pred_weight'])
        arg_params['rpn_bbox_pred_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['rpn_bbox_pred_bias'])

    def init_weight_fpn(self, cfg, arg_params, aux_params):
        arg_params['fpn_p6_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['fpn_p6_weight'])
        arg_params['fpn_p6_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['fpn_p6_bias'])
        arg_params['fpn_p5_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['fpn_p5_weight'])
        arg_params['fpn_p5_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['fpn_p5_bias'])
        arg_params['fpn_p4_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['fpn_p4_weight'])
        arg_params['fpn_p4_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['fpn_p4_bias'])
        arg_params['fpn_p3_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['fpn_p3_weight'])
        arg_params['fpn_p3_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['fpn_p3_bias'])
        arg_params['fpn_p2_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['fpn_p2_weight'])
        arg_params['fpn_p2_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['fpn_p2_bias'])

        arg_params['fpn_p5_1x1_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['fpn_p5_1x1_weight'])
        arg_params['fpn_p5_1x1_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['fpn_p5_1x1_bias'])
        arg_params['fpn_p4_1x1_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['fpn_p4_1x1_weight'])
        arg_params['fpn_p4_1x1_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['fpn_p4_1x1_bias'])
        arg_params['fpn_p3_1x1_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['fpn_p3_1x1_weight'])
        arg_params['fpn_p3_1x1_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['fpn_p3_1x1_bias'])
        arg_params['fpn_p2_1x1_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['fpn_p2_1x1_weight'])
        arg_params['fpn_p2_1x1_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['fpn_p2_1x1_bias'])

    def init_weight_dcn_offset(self, cfg, arg_params, aux_params):
        for key in self.arg_shape_dict:
            if 'offset' in key and not key in arg_params:
                if 'reduce' in key and 'weight' in key:
                    arg_params[key] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict[key])
                else:
                    arg_params[key] = mx.nd.zeros(shape=self.arg_shape_dict[key])

    def init_weight(self, cfg, arg_params, aux_params):
        self.init_weight_rpn(cfg, arg_params, aux_params)
        self.init_weight_rcnn(cfg, arg_params, aux_params)
        self.init_weight_fpn(cfg, arg_params, aux_params)
        if cfg.network.backbone_use_dconv or cfg.FPN.use_dconv:
            self.init_weight_dcn_offset(cfg, arg_params, aux_params)

