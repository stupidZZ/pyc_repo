import cPickle
import mxnet as mx
import numpy as np
from common.lib.utils.symbol import Symbol
from common.backbone import resnet_v1
from common.operator_py.proposal import *
from common.operator_py.proposal_target import *
from common.operator_py.box_annotator_ohem import *
from common.operator_py.arange_like import *
from common.gpu_metric import *

class res_v1_rcnn(Symbol):
    def __init__(self, FP16=False):
        """
        Use __init__ to define parameter network needs
        """
        self.FP16 = FP16

    def get_rpn(self, conv_feat, num_anchors):
        rpn_conv = mx.sym.Convolution(data=conv_feat, kernel=(3, 3), pad=(1, 1), num_filter=1024, name='rpn_conv_3x3')
        rpn_relu = mx.sym.Activation(data=rpn_conv, act_type='relu', name='rpn_relu')
        rpn_cls_score = mx.sym.Convolution(data=rpn_relu, kernel=(1, 1), pad=(0, 0), num_filter=2 * num_anchors, name='rpn_cls_score')
        rpn_bbox_pred = mx.sym.Convolution(data=rpn_relu, kernel=(1, 1), pad=(0, 0), num_filter=4 * num_anchors, name='rpn_bbox_pred')
        return rpn_cls_score, rpn_bbox_pred

    def get_symbol(self, cfg, is_train=True):
        # config alias for convenient
        num_classes = cfg.dataset.num_classes
        num_reg_classes = (2 if cfg.class_agnostic else num_classes)
        num_anchors = cfg.network.NUM_ANCHORS
        grad_scale = cfg.TRAIN.FP16_GRAD_SCALE if self.FP16 else 1.0

        # input init
        if is_train:
            data = mx.sym.Variable(name="data")
            im_info = mx.sym.Variable(name="im_info")
            gt_boxes = mx.sym.Variable(name="gt_boxes")
            rpn_label = mx.sym.Variable(name='label')
            rpn_bbox_target = mx.sym.Variable(name='bbox_target')
            rpn_bbox_weight = mx.sym.Variable(name='bbox_weight')
            if self.FP16:
                data = mx.sym.Cast(data, dtype = np.float16)
                im_info = mx.sym.Cast(im_info, dtype = np.float16)
                gt_boxes = mx.sym.Cast(gt_boxes, dtype = np.float16)
                rpn_bbox_target = mx.sym.Cast(rpn_bbox_target, dtype = np.float16)
                rpn_bbox_weight = mx.sym.Cast(rpn_bbox_weight, dtype = np.float16)
        else:
            data = mx.sym.Variable(name="data")
            im_info = mx.sym.Variable(name="im_info")

        _, _, _, c4, c5 = resnet_v1.get_resnet_backbone(data=data,
                                                        num_layers=cfg.network.num_layers,
                                                        use_dilation_on_c5=cfg.network.use_dilation_on_c5,
                                                        use_dconv=cfg.network.backbone_use_dconv,
                                                        dconv_lr_mult=cfg.network.backbone_dconv_lr_mult,
                                                        dconv_group=cfg.network.backbone_dconv_group,
                                                        dconv_start_channel=cfg.network.backbone_dconv_start_channel)

        rpn_cls_score, rpn_bbox_pred = self.get_rpn(c4, num_anchors)

        if is_train:
            # prepare rpn data
            rpn_cls_score_reshape = mx.sym.Reshape(data=rpn_cls_score, shape=(0, 2, -1, 0), name="rpn_cls_score_reshape")

            # classification
            if self.FP16:
                rpn_cls_score_reshape = mx.sym.Cast(rpn_cls_score_reshape, dtype=np.float32)
                rpn_cls_prob = mx.sym.SoftmaxOutput(data=rpn_cls_score_reshape, label=rpn_label, multi_output=True,
                                                    normalization='valid', use_ignore=True, ignore_label=-1, name="rpn_cls_prob",
                                                    grad_scale=grad_scale)
            else:
                rpn_cls_prob = mx.sym.SoftmaxOutput(data=rpn_cls_score_reshape, label=rpn_label, multi_output=True,
                                                    normalization='valid', use_ignore=True, ignore_label=-1, name="rpn_cls_prob")

            # bounding box regression
            rpn_bbox_loss_ = rpn_bbox_weight * mx.sym.smooth_l1(name='rpn_bbox_loss_', scalar=3.0,
                                                                data=(rpn_bbox_pred - rpn_bbox_target))
            rpn_bbox_loss = mx.sym.MakeLoss(name='rpn_bbox_loss', data=rpn_bbox_loss_,
                                            grad_scale=grad_scale * 1.0 / (cfg.TRAIN.RPN_BATCH_SIZE * cfg.TRAIN.BATCH_IMAGES))
            # ROI proposal
            rpn_cls_act = mx.sym.SoftmaxActivation(data=rpn_cls_score_reshape, mode="channel", name="rpn_cls_act")
            if self.FP16:
                rpn_cls_act = mx.sym.Cast(rpn_cls_act, dtype=np.float16)
            rpn_cls_act_reshape = mx.sym.Reshape(data=rpn_cls_act, shape=(0, 2 * num_anchors, -1, 0), name='rpn_cls_act_reshape')
            if cfg.TRAIN.CXX_PROPOSAL:
                rois = mx.contrib.sym.MultiProposal(
                    cls_prob=rpn_cls_act_reshape, bbox_pred=rpn_bbox_pred, im_info=im_info, name='rois',
                    feature_stride=cfg.network.RPN_FEAT_STRIDE, scales=tuple(cfg.network.ANCHOR_SCALES),
                    ratios=tuple(cfg.network.ANCHOR_RATIOS),
                    rpn_pre_nms_top_n=cfg.TRAIN.RPN_PRE_NMS_TOP_N, rpn_post_nms_top_n=cfg.TRAIN.RPN_POST_NMS_TOP_N,
                    threshold=cfg.TRAIN.RPN_NMS_THRESH, rpn_min_size=cfg.TRAIN.RPN_MIN_SIZE)
            else:
                rois = mx.sym.Custom(
                    cls_prob=rpn_cls_act_reshape, bbox_pred=rpn_bbox_pred, im_info=im_info, name='rois',
                    op_type='proposal', feat_stride=cfg.network.RPN_FEAT_STRIDE,
                    scales=tuple(cfg.network.ANCHOR_SCALES), ratios=tuple(cfg.network.ANCHOR_RATIOS),
                    rpn_pre_nms_top_n=cfg.TRAIN.RPN_PRE_NMS_TOP_N, rpn_post_nms_top_n=cfg.TRAIN.RPN_POST_NMS_TOP_N,
                    threshold=cfg.TRAIN.RPN_NMS_THRESH, rpn_min_size=cfg.TRAIN.RPN_MIN_SIZE)
            # ROI proposal target
            gt_boxes_reshape = mx.sym.Reshape(data=gt_boxes, shape=(-1, 6), name='gt_boxes_reshape')

            rois_all = mx.sym.Custom(rois=rois, gt_boxes=gt_boxes_reshape,
                                    op_type='proposal_target',
                                    num_classes=num_reg_classes,
                                    batch_images=cfg.TRAIN.BATCH_IMAGES,
                                    batch_rois=cfg.TRAIN.BATCH_ROIS,
                                    cfg=cPickle.dumps(cfg),
                                    fg_fraction=cfg.TRAIN.FG_FRACTION)

            rois = rois_all[0]
            label = rois_all[1]
            bbox_target = rois_all[2]
            bbox_weight = rois_all[3]
        else:
            # ROI Proposal
            rpn_cls_score_reshape = mx.sym.Reshape(data=rpn_cls_score, shape=(0, 2, -1, 0), name="rpn_cls_score_reshape")
            rpn_cls_prob = mx.sym.SoftmaxActivation(data=rpn_cls_score_reshape, mode="channel", name="rpn_cls_prob")
            rpn_cls_prob_reshape = mx.sym.Reshape(data=rpn_cls_prob, shape=(0, 2 * num_anchors, -1, 0), name='rpn_cls_prob_reshape')

            if cfg.TEST.CXX_PROPOSAL:
                rois = mx.contrib.sym.MultiProposal(
                    cls_prob=rpn_cls_prob_reshape, bbox_pred=rpn_bbox_pred, im_info=im_info, name='rois',
                    feature_stride=cfg.network.RPN_FEAT_STRIDE, scales=tuple(cfg.network.ANCHOR_SCALES),
                    ratios=tuple(cfg.network.ANCHOR_RATIOS),
                    rpn_pre_nms_top_n=cfg.TEST.RPN_PRE_NMS_TOP_N, rpn_post_nms_top_n=cfg.TEST.RPN_POST_NMS_TOP_N,
                    threshold=cfg.TEST.RPN_NMS_THRESH, rpn_min_size=cfg.TEST.RPN_MIN_SIZE)
            else:
                rois = mx.sym.Custom(
                    cls_prob=rpn_cls_prob_reshape, bbox_pred=rpn_bbox_pred, im_info=im_info, name='rois',
                    op_type='proposal', feat_stride=cfg.network.RPN_FEAT_STRIDE,
                    scales=tuple(cfg.network.ANCHOR_SCALES), ratios=tuple(cfg.network.ANCHOR_RATIOS),
                    rpn_pre_nms_top_n=cfg.TEST.RPN_PRE_NMS_TOP_N, rpn_post_nms_top_n=cfg.TEST.RPN_POST_NMS_TOP_N,
                    threshold=cfg.TEST.RPN_NMS_THRESH, rpn_min_size=cfg.TEST.RPN_MIN_SIZE)

        conv_new_1 = mx.sym.Convolution(data=c5, kernel=(1, 1), num_filter=256, name="conv_new_1")
        conv_new_1_relu = mx.sym.Activation(data=conv_new_1, act_type='relu', name='conv_new_1_relu')

        if cfg.network.use_dilation_on_c5:
            spatial_scale = 0.0625
        else:
            spatial_scale = 0.03125

        if cfg.network.use_dpool:
            roi_pool_offset = mx.contrib.sym.DeformablePSROIPooling(name='roi_offset', data=conv_new_1_relu,
                                                                    rois=rois,
                                                                    group_size=1,
                                                                    pooled_size=7, sample_per_part=2,
                                                                    no_trans=True, part_size=7, output_dim=256,
                                                                    spatial_scale=spatial_scale)
            offset = mx.sym.FullyConnected(name='offset', data=roi_pool_offset, num_hidden=7 * 7 * 2, lr_mult=cfg.network.dpool_lr_mult)
            offset_reshape = mx.sym.Reshape(data=offset, shape=(-1, 2, 7, 7), name="offset_reshape")
            roi_pool = mx.contrib.sym.DeformablePSROIPooling(name='roi_deformable', data=conv_new_1_relu,
                                                            rois=rois,
                                                            trans=offset_reshape,
                                                            group_size=1,
                                                            pooled_size=7, sample_per_part=2,
                                                            no_trans=False, part_size=7, output_dim=256,
                                                            spatial_scale=spatial_scale)

        else:
            roi_pool = mx.contrib.sym.DeformablePSROIPooling(name='roi_align', data=conv_new_1_relu,
                                                             rois=rois,
                                                             group_size=1,
                                                             pooled_size=7, sample_per_part=2,
                                                             no_trans=True, part_size=7, output_dim=256,
                                                             spatial_scale=spatial_scale)

        # 2 fc
        fc_new_1 = mx.symbol.FullyConnected(name='fc_new_1', data=roi_pool, num_hidden=1024)
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
                    labels_ohem = mx.sym.Cast(labels_ohem, dtype=np.float32)
                    cls_score = mx.sym.Cast(cls_score, dtype=np.float32)
                    cls_prob = mx.sym.SoftmaxOutput(name='cls_prob', data=cls_score, label=labels_ohem,
                                                normalization='valid', use_ignore=True, ignore_label=-1, grad_scale = grad_scale)
                else:
                    cls_prob = mx.sym.SoftmaxOutput(name='cls_prob', data=cls_score, label=labels_ohem,
                                                    normalization='valid', use_ignore=True, ignore_label=-1)
                bbox_loss_ = bbox_weights_ohem * mx.sym.smooth_l1(name='bbox_loss_', scalar=1.0, data=(bbox_pred - bbox_target))
                bbox_loss = mx.sym.MakeLoss(name='bbox_loss', data=bbox_loss_, grad_scale=grad_scale * 2.0 / cfg.TRAIN.BATCH_ROIS_OHEM)
                rcnn_label = labels_ohem
            else:
                if self.FP16:
                    label = mx.sym.Cast(label, dtype=np.float32)
                    cls_score = mx.sym.Cast(cls_score, dtype=np.float32)
                    cls_prob = mx.sym.SoftmaxOutput(name='cls_prob', data=cls_score, label=label, normalization='valid',
                                                    grad_scale=grad_scale)
                else:
                    cls_prob = mx.sym.SoftmaxOutput(name='cls_prob', data=cls_score, label=label, normalization='valid')
                bbox_loss_ = bbox_weight * mx.sym.smooth_l1(name='bbox_loss_', scalar=1.0, data=(bbox_pred - bbox_target))
                bbox_loss = mx.sym.MakeLoss(name='bbox_loss', data=bbox_loss_,
                                            grad_scale=grad_scale * 2.0 / (cfg.TRAIN.BATCH_IMAGES * cfg.TRAIN.BATCH_ROIS))
                rcnn_label = label

            # reshape output
            rcnn_label = mx.sym.Reshape(data=rcnn_label, shape=(cfg.TRAIN.BATCH_IMAGES, -1), name='label_reshape')
            cls_prob = mx.sym.Reshape(data=cls_prob, shape=(cfg.TRAIN.BATCH_IMAGES, -1, num_classes),
                                      name='cls_prob_reshape')
            bbox_loss = mx.sym.Reshape(data=bbox_loss, shape=(cfg.TRAIN.BATCH_IMAGES, -1, 4 * num_reg_classes),
                                       name='bbox_loss_reshape')

            output_list = [rpn_cls_prob, rpn_bbox_loss, cls_prob, bbox_loss, mx.sym.BlockGrad(rcnn_label)]

            # get gpu metric
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
            cls_prob = mx.sym.SoftmaxActivation(name='cls_prob', data=cls_score)
            cls_prob = mx.sym.Reshape(data=cls_prob, shape=(cfg.TEST.BATCH_IMAGES, -1, num_classes), name='cls_prob_reshape')
            bbox_pred = mx.sym.Reshape(data=bbox_pred, shape=(cfg.TEST.BATCH_IMAGES, -1, 4 * num_reg_classes), name='bbox_pred_reshape')

            group = mx.sym.Group([rois, cls_prob, bbox_pred])

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
        arg_params['conv_new_1_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['conv_new_1_weight'])
        arg_params['conv_new_1_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['conv_new_1_bias'])
        arg_params['fc_new_1_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['fc_new_1_weight'])
        arg_params['fc_new_1_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['fc_new_1_bias'])
        arg_params['fc_new_2_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['fc_new_2_weight'])
        arg_params['fc_new_2_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['fc_new_2_bias'])
        arg_params['cls_score_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['cls_score_weight'])
        arg_params['cls_score_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['cls_score_bias'])
        arg_params['bbox_pred_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['bbox_pred_weight'])
        arg_params['bbox_pred_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['bbox_pred_bias'])

        if cfg.network.use_dpool:
            arg_params['offset_weight'] = mx.nd.zeros(shape=self.arg_shape_dict['offset_weight'])
            arg_params['offset_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['offset_bias'])

    def init_weight_rpn(self, cfg, arg_params, aux_params):
        arg_params['rpn_conv_3x3_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['rpn_conv_3x3_weight'])
        arg_params['rpn_conv_3x3_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['rpn_conv_3x3_bias'])
        arg_params['rpn_cls_score_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['rpn_cls_score_weight'])
        arg_params['rpn_cls_score_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['rpn_cls_score_bias'])
        arg_params['rpn_bbox_pred_weight'] = mx.random.normal(0, 0.01, shape=self.arg_shape_dict['rpn_bbox_pred_weight'])
        arg_params['rpn_bbox_pred_bias'] = mx.nd.zeros(shape=self.arg_shape_dict['rpn_bbox_pred_bias'])

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

        if cfg.network.backbone_use_dconv:
            self.init_weight_dcn_offset(cfg, arg_params, aux_params)

