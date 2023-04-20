import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as models
from torch.autograd import Variable
import numpy as np
from model.utils.config import cfg
from model.rpn.rpn import _RPN
from model.roi_pooling.modules.roi_pool import _RoIPooling
from model.roi_crop.modules.roi_crop import _RoICrop
from model.roi_align.modules.roi_align import RoIAlignAvg
from model.rpn.proposal_target_layer_cascade import _ProposalTargetLayer
import math
from model.da_faster_rcnn.DA import _ImageDA
from model.da_faster_rcnn.DA import _InstanceDA
import time
import pdb
from model.utils.net_utils import _smooth_l1_loss, _crop_pool_layer, _affine_grid_gen, _affine_theta

from model.da_faster_rcnn.UDA import get_source_common_weight,get_target_common_weight,normalize

#partial-set pascal voc to water

class _fasterRCNN(nn.Module):
    """ faster RCNN """

    def __init__(self, classes, class_agnostic):
        super(_fasterRCNN, self).__init__()
        self.classes = classes
        self.n_classes = len(classes)
        self.class_agnostic = class_agnostic
        # loss
        self.RCNN_loss_cls = 0
        self.RCNN_loss_bbox = 0

        # define rpn
        self.RCNN_rpn = _RPN(self.dout_base_model)
        self.RCNN_proposal_target = _ProposalTargetLayer(self.n_classes)
        self.RCNN_roi_pool = _RoIPooling(cfg.POOLING_SIZE, cfg.POOLING_SIZE, 1.0 / 16.0)
        self.RCNN_roi_align = RoIAlignAvg(cfg.POOLING_SIZE, cfg.POOLING_SIZE, 1.0 / 16.0)

        self.grid_size = cfg.POOLING_SIZE * 2 if cfg.CROP_RESIZE_WITH_MAX_POOL else cfg.POOLING_SIZE
        self.RCNN_roi_crop = _RoICrop()

        self.RCNN_imageDA = _ImageDA(self.dout_base_model)
        self.RCNN_instanceDA = _InstanceDA()
        self.consistency_loss = torch.nn.MSELoss(size_average=False)

    def forward(self, im_data, im_info, gt_boxes, num_boxes,
                tgt_im_data, tgt_im_info, tgt_gt_boxes, tgt_num_boxes,records,temperture_memory_filter):
        need_backprop=torch.tensor([1.0]).cuda()
        tgt_need_backprop = torch.tensor([0.0]).cuda()


        assert need_backprop.detach() == 1 and tgt_need_backprop.detach() == 0

        batch_size = im_data.size(0)
        im_info = im_info.data  # (size1,size2, image ratio(new image / source image) )
        gt_boxes = gt_boxes.data
        num_boxes = num_boxes.data
        need_backprop = need_backprop.data

        base_feat = self.RCNN_base(im_data)
        self.RCNN_rpn.train()
        rois, rpn_loss_cls, rpn_loss_bbox = self.RCNN_rpn(base_feat, im_info, gt_boxes,num_boxes)
        if self.training:
            roi_data = self.RCNN_proposal_target(rois, gt_boxes, num_boxes)
            rois, rois_label, rois_target, rois_inside_ws, rois_outside_ws = roi_data  # 这里的rois有256个

            rois_label = Variable(rois_label.view(-1).long())
            rois_target = Variable(rois_target.view(-1, rois_target.size(2)))
            rois_inside_ws = Variable(rois_inside_ws.view(-1, rois_inside_ws.size(2)))
            rois_outside_ws = Variable(rois_outside_ws.view(-1, rois_outside_ws.size(2)))
        else:
            rois_label = None
            rois_target = None
            rois_inside_ws = None
            rois_outside_ws = None
            rpn_loss_cls = 0
            rpn_loss_bbox = 0

        rois = Variable(rois)
        if cfg.POOLING_MODE == 'align':
            pooled_feat = self.RCNN_roi_align(base_feat, rois.view(-1, 5))


        pooled_feat = self._head_to_tail(pooled_feat)
        bbox_pred = self.RCNN_bbox_pred(pooled_feat)
        if self.training and not self.class_agnostic:
            bbox_pred_view = bbox_pred.view(bbox_pred.size(0), int(bbox_pred.size(1) / 4), 4)
            bbox_pred_select = torch.gather(bbox_pred_view, 1,rois_label.view(rois_label.size(0), 1, 1).expand(rois_label.size(0), 1, 4))
            bbox_pred = bbox_pred_select.squeeze(1)

        cls_score = self.RCNN_cls_score(pooled_feat)  # 用于分类#[128,21]
        cls_prob2 = F.softmax(cls_score, 1)#[128,21]
        RCNN_loss_cls = 0
        RCNN_loss_bbox = 0



        if self.training:
            RCNN_loss_cls = F.cross_entropy(cls_score, rois_label)
            RCNN_loss_bbox = _smooth_l1_loss(bbox_pred, rois_target, rois_inside_ws, rois_outside_ws)

        cls_prob = cls_prob2.view(batch_size, rois.size(1), -1)
        bbox_pred = bbox_pred.view(batch_size, rois.size(1), -1)


        """ =================== for target =========================="""

        tgt_batch_size = tgt_im_data.size(0)
        tgt_im_info = tgt_im_info.data  # (size1,size2, image ratio(new image / source image) )
        tgt_gt_boxes = tgt_gt_boxes.data
        tgt_num_boxes = tgt_num_boxes.data


        # feed image data to base model to obtain base feature map
        tgt_base_feat = self.RCNN_base(tgt_im_data)


        self.RCNN_rpn.eval()
        tgt_rois, tgt_rpn_loss_cls, tgt_rpn_loss_bbox = self.RCNN_rpn(tgt_base_feat, tgt_im_info, tgt_gt_boxes, tgt_num_boxes)
        tgt_rois = Variable(tgt_rois)

        if cfg.POOLING_MODE == 'align':
            tgt_pooled_feat = self.RCNN_roi_align(tgt_base_feat, tgt_rois.view(-1, 5))  # torch.Size([300, 512, 7, 7])

        tgt_pooled_feat = self._head_to_tail(tgt_pooled_feat)

        tgt_cls_score = self.RCNN_cls_score(tgt_pooled_feat)  # 用于分类#[300,21]
        tgt_cls_prob = F.softmax(tgt_cls_score, 1)  # [300,21]




        batch_size_ins = len(cls_prob2)
        tgt_batch_size_ins=len(tgt_cls_prob)



        common_source_weight = torch.zeros(batch_size_ins, 1).cuda()

        delta_margin = torch.zeros(self.n_classes, 1).cuda()
        target_pseudo_label = tgt_cls_prob.max(1)[1]

        num_per_class = torch.zeros(self.n_classes, 1).cuda()
        target_predict_order = torch.sort(tgt_cls_prob, dim=1, descending=True)[0]
        for index in range(tgt_batch_size_ins):
            if target_pseudo_label[index] != 0:
                delta_margin[target_pseudo_label[index],0] += target_predict_order[index,0] - target_predict_order[index,1:].mean(dim=0)
                num_per_class[target_pseudo_label[index],0] +=1
        if sum(num_per_class)==0:
            weight_vetory=temperture_memory_filter
        else:
            weight_vetory = ((temperture_memory_filter * records) + torch.div(delta_margin, num_per_class + 1e-6)) / (records + 1)

        norm_weight_vetory=normalize(weight_vetory, cut=0)


        for index in range(batch_size_ins):
            common_source_weight[index, 0] = norm_weight_vetory[rois_label[index]]  # - domain_prob_source[index]


        temperture_memory_filter=weight_vetory.detach()




        #源域图像级权重
        _, _, h1, w1 = im_data.size()
        _, _, h2, w2 = base_feat.size()
        weight_ad_img = torch.zeros(size=(h2, w2)).cuda()
        scale_w = w2 / w1
        scale_h = h2 / h1

        gt_boxes2 = gt_boxes.squeeze(0)


        for i in range(len(gt_boxes2)):
             if gt_boxes2[i][4] != 0:  #考虑到image-level中，直接不考虑背景类的信息，即默认为0
                x1 = torch.tensor(math.floor(gt_boxes2[i][0] * scale_w))
                y1 = torch.tensor(math.floor(gt_boxes2[i][1] * scale_h))
                x2 = torch.tensor(math.floor(gt_boxes2[i][2] * scale_w))
                y2 = torch.tensor(math.floor(gt_boxes2[i][3] * scale_h))
                weight_gt = torch.ones((y2 - y1 + 1, x2 - x1 + 1)).cuda() * norm_weight_vetory[int(gt_boxes2[i][4])]
                indice1 = np.arange(x1, x2 + 1, 1)
                indice2 = np.arange(y1, y2 + 1, 1)
                weight_ad_img[np.ix_(indice2, indice1)] = weight_gt


        # # 目标域图像级别的权重
        _, _, th1, tw1 = tgt_im_data.size()
        _, _, th2, tw2 = tgt_base_feat.size()
        tgt_weight_ad_img = torch.zeros(size=(th2, tw2)).cuda()
        t_scale_w = tw2 / tw1
        t_scale_h = th2 / th1

        # print("目标域边框", tgt_rois)
        tgt_gt_rois = tgt_rois.new(tgt_rois.size()).zero_()
        tgt_gt_rois[:, :, 1:5] = tgt_rois[:, :, 1:5]
        tgt_gt_rois[:,:, 0] = target_pseudo_label
        # print("目标域边框2", tgt_gt_rois)

        tgt_gt_rois2 = tgt_gt_rois.squeeze(0)
        for i in range(len(tgt_gt_rois2)):
            if tgt_gt_rois2[i][0] != 0:
                    t_x1 = torch.tensor(math.floor(tgt_gt_rois2[len(tgt_gt_rois2)-i-1][1] * t_scale_w))
                    t_y1 = torch.tensor(math.floor(tgt_gt_rois2[len(tgt_gt_rois2)-i-1][2] * t_scale_h))
                    t_x2 = torch.tensor(math.floor(tgt_gt_rois2[len(tgt_gt_rois2)-i-1][3] * t_scale_w))
                    t_y2 = torch.tensor(math.floor(tgt_gt_rois2[len(tgt_gt_rois2)-i-1][4] * t_scale_h))

                    tgt_weight_gt = torch.ones((t_y2 - t_y1 + 1, t_x2 - t_x1 + 1)).cuda() * norm_weight_vetory[int(tgt_gt_rois2[len(tgt_gt_rois2)-i-1][0])]
                    # print("真实gt的权重大小", weight_gt.shape)

                    t_indice1 = np.arange(t_x1, t_x2 + 1, 1)
                    t_indice2 = np.arange(t_y1, t_y2 + 1, 1)
                    # print("覆盖", len(indice1), len(indice2))
                    tgt_weight_ad_img[np.ix_(t_indice2, t_indice1)] = tgt_weight_gt


        # print("tgt_image_softmax_sort", tgt_image_softmax_sort)

        source_weight_ad_img=weight_ad_img
        source_common_weight_ins = common_source_weight

        target_weight_ad_img=tgt_weight_ad_img
        target_common_weight_ins=norm_weight_vetory[target_pseudo_label]


        """  DA loss   """
        # print("样本类别",self.n_classes)


        "源域对齐"
        base_score = self.RCNN_imageDA(base_feat)
        DA_img_loss_cls = nn.BCELoss(weight=1+source_weight_ad_img.view(1, 1, h2, w2).cuda().detach())(base_score, torch.ones_like(base_score))


        instance_sigmoid= self.RCNN_instanceDA(pooled_feat)
        DA_ins_loss_cls = nn.BCELoss(weight=1+source_common_weight_ins)(instance_sigmoid, torch.ones_like(instance_sigmoid))



        """  ************** taget loss ****************  """

        tgt_base_score = self.RCNN_imageDA(tgt_base_feat)
        tgt_DA_img_loss_cls = nn.BCELoss(weight=1+target_weight_ad_img.view(1, 1, th2, tw2))(tgt_base_score, torch.zeros_like(tgt_base_score))

        tgt_instance_sigmoid = self.RCNN_instanceDA(tgt_pooled_feat)
        tgt_DA_ins_loss_cls = nn.BCELoss(weight=1+target_common_weight_ins)(tgt_instance_sigmoid, torch.zeros_like(tgt_instance_sigmoid))




        return rois, cls_prob, bbox_pred, rpn_loss_cls, rpn_loss_bbox, RCNN_loss_cls, RCNN_loss_bbox,rois_label, \
               DA_img_loss_cls, DA_ins_loss_cls, tgt_DA_img_loss_cls, tgt_DA_ins_loss_cls,temperture_memory_filter

    def _init_weights(self):
        def normal_init(m, mean, stddev, truncated=False):
            """
            weight initalizer: truncated normal and random normal.
            """
            # x is a parameter
            if truncated:
                m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean)  # not a perfect approximation
            else:
                m.weight.data.normal_(mean, stddev)
                m.bias.data.zero_()

        normal_init(self.RCNN_rpn.RPN_Conv, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_rpn.RPN_cls_score, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_rpn.RPN_bbox_pred, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_cls_score, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_bbox_pred, 0, 0.001, cfg.TRAIN.TRUNCATED)

        normal_init(self.RCNN_cls_score_img, 0, 0.01, cfg.TRAIN.TRUNCATED)
    def create_architecture(self):
        self._init_modules()
        self._init_weights()
