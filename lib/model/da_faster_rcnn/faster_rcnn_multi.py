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

from model.da_faster_rcnn.UDA import normalize


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

        self.RCNN_imageDA3 = _ImageDA(256)
        self.RCNN_imageDA4 = _ImageDA(512)
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

        conv3_feat=self.conv3(im_data)
        conv4_feat = self.conv34(conv3_feat)
        base_feat=self.conv45(conv4_feat)

        self.RCNN_rpn.train()

        rois, rpn_loss_cls, rpn_loss_bbox = self.RCNN_rpn(base_feat, im_info, gt_boxes,num_boxes)
        if self.training:
            roi_data = self.RCNN_proposal_target(rois, gt_boxes, num_boxes)
            rois, rois_label, rois_target, rois_inside_ws, rois_outside_ws = roi_data

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

        rois = Variable(rois)  # [1, 256, 5]

        if cfg.POOLING_MODE == 'align':
            pooled_feat = self.RCNN_roi_align(base_feat, rois.view(-1, 5))


        pooled_feat = self._head_to_tail(pooled_feat)

        bbox_pred = self.RCNN_bbox_pred(pooled_feat)
        if self.training and not self.class_agnostic:

            bbox_pred_view = bbox_pred.view(bbox_pred.size(0), int(bbox_pred.size(1) / 4), 4)
            bbox_pred_select = torch.gather(bbox_pred_view, 1, rois_label.view(rois_label.size(0), 1, 1).expand(rois_label.size(0), 1, 4))
            bbox_pred = bbox_pred_select.squeeze(1)


        cls_score = self.RCNN_cls_score(pooled_feat)
        cls_prob2 = F.softmax(cls_score, 1)

        RCNN_loss_cls = 0
        RCNN_loss_bbox = 0



        if self.training:
            RCNN_loss_cls = F.cross_entropy(cls_score, rois_label)
            # bounding box regression L1 loss
            RCNN_loss_bbox = _smooth_l1_loss(bbox_pred, rois_target, rois_inside_ws, rois_outside_ws)

        cls_prob = cls_prob2.view(batch_size, rois.size(1), -1)
        bbox_pred = bbox_pred.view(batch_size, rois.size(1), -1)


        """ =================== for target =========================="""

        tgt_batch_size = tgt_im_data.size(0)
        tgt_im_info = tgt_im_info.data  # (size1,size2, image ratio(new image / source image) )
        tgt_gt_boxes = tgt_gt_boxes.data
        tgt_num_boxes = tgt_num_boxes.data



        tgt_conv3_feat = self.conv3(tgt_im_data)
        tgt_conv4_feat = self.conv34(tgt_conv3_feat)
        tgt_base_feat = self.conv45(tgt_conv4_feat)

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

        norm_weight_vetory=normalize(weight_vetory)

        for index in range(batch_size_ins):
             common_source_weight[index, 0] = norm_weight_vetory[rois_label[index]]  # - domain_prob_source[index]


        temperture_memory_filter=weight_vetory.detach()


        _, _, h1, w1 = im_data.size()

        _, _, h3, w3 = conv3_feat.size()
        _, _, h4, w4 = conv4_feat.size()
        _, _, h5, w5 = base_feat.size()

        weight_ad_img_3 = torch.zeros(size=(h3, w3)).cuda()
        weight_ad_img_4 = torch.zeros(size=(h4, w4)).cuda()
        weight_ad_img_5 = torch.zeros(size=(h5, w5)).cuda()

        scale_w_3 = w5 / w1
        scale_h_3 = h5 / h1

        scale_w_4 = w5 / w1
        scale_h_4 = h5 / h1

        scale_w_5 = w5 / w1
        scale_h_5 = h5 / h1
        gt_boxes2 = gt_boxes.squeeze(0)

        for i in range(len(gt_boxes2)):
            if gt_boxes2[i][4] != 0:

                x1_3 = torch.tensor(math.floor(gt_boxes2[i][0] * scale_w_3))
                y1_3 = torch.tensor(math.floor(gt_boxes2[i][1] * scale_h_3))
                x2_3 = torch.tensor(math.floor(gt_boxes2[i][2] * scale_w_3))
                y2_3 = torch.tensor(math.floor(gt_boxes2[i][3] * scale_h_3))
                weight_gt_3 = torch.ones((y2_3 - y1_3 + 1, x2_3 - x1_3 + 1)).cuda() * norm_weight_vetory[int(gt_boxes2[i][4])]

                indice1_3 = np.arange(x1_3, x2_3 + 1, 1)
                indice2_3 = np.arange(y1_3, y2_3 + 1, 1)
                weight_ad_img_3[np.ix_(indice2_3, indice1_3)] = weight_gt_3

                x1_4 = torch.tensor(math.floor(gt_boxes2[i][0] * scale_w_4))
                y1_4 = torch.tensor(math.floor(gt_boxes2[i][1] * scale_h_4))
                x2_4 = torch.tensor(math.floor(gt_boxes2[i][2] * scale_w_4))
                y2_4 = torch.tensor(math.floor(gt_boxes2[i][3] * scale_h_4))
                weight_gt_4 = torch.ones((y2_4 - y1_4 + 1, x2_4 - x1_4 + 1)).cuda() * norm_weight_vetory[int(gt_boxes2[i][4])]

                indice1_4 = np.arange(x1_4, x2_4 + 1, 1)
                indice2_4 = np.arange(y1_4, y2_4 + 1, 1)
                weight_ad_img_4[np.ix_(indice2_4, indice1_4)] = weight_gt_4

                x1_5 = torch.tensor(math.floor(gt_boxes2[i][0] * scale_w_5))
                y1_5 = torch.tensor(math.floor(gt_boxes2[i][1] * scale_h_5))
                x2_5 = torch.tensor(math.floor(gt_boxes2[i][2] * scale_w_5))
                y2_5 = torch.tensor(math.floor(gt_boxes2[i][3] * scale_h_5))
                weight_gt_5 = torch.ones((y2_5 - y1_5 + 1, x2_5 - x1_5 + 1)).cuda() * norm_weight_vetory[int(gt_boxes2[i][4])]

                indice1_5 = np.arange(x1_5, x2_5 + 1, 1)
                indice2_5 = np.arange(y1_5, y2_5 + 1, 1)
                weight_ad_img_5[np.ix_(indice2_5, indice1_5)] = weight_gt_5


        _, _, th1, tw1 = tgt_im_data.size()
        _, _, th3, tw3 = tgt_conv3_feat.size()
        _, _, th4, tw4 = tgt_conv4_feat.size()
        _, _, th5, tw5 = tgt_base_feat.size()

        tgt_weight_ad_img_3 = torch.zeros(size=(th3, tw3)).cuda()
        tgt_weight_ad_img_4 = torch.zeros(size=(th4, tw4)).cuda()
        tgt_weight_ad_img_5 = torch.zeros(size=(th5, tw5)).cuda()


        t_scale_w_3 = tw3 / tw1
        t_scale_h_3 = th3 / th1

        t_scale_w_4 = tw4 / tw1
        t_scale_h_4 = th4 / th1

        t_scale_w_5 = tw5 / tw1
        t_scale_h_5 = th5 / th1


        tgt_gt_rois = tgt_rois.new(tgt_rois.size()).zero_()
        tgt_gt_rois[:, :, 1:5] = tgt_rois[:, :, 1:5]
        tgt_gt_rois[:, :, 0] = target_pseudo_label


        tgt_gt_rois2 = tgt_gt_rois.squeeze(0)

        for i in range(len(tgt_gt_rois2)):
            if tgt_gt_rois2[i][0] != 0:
                t_x1_3 = torch.tensor(math.floor(tgt_gt_rois2[len(tgt_gt_rois2) - i - 1][1] * t_scale_w_3))
                t_y1_3 = torch.tensor(math.floor(tgt_gt_rois2[len(tgt_gt_rois2) - i - 1][2] * t_scale_h_3))
                t_x2_3 = torch.tensor(math.floor(tgt_gt_rois2[len(tgt_gt_rois2) - i - 1][3] * t_scale_w_3))
                t_y2_3 = torch.tensor(math.floor(tgt_gt_rois2[len(tgt_gt_rois2) - i - 1][4] * t_scale_h_3))

                tgt_weight_gt_3 = torch.ones((t_y2_3 - t_y1_3 + 1, t_x2_3 - t_x1_3 + 1)).cuda() * \
                                  norm_weight_vetory[int(tgt_gt_rois2[len(tgt_gt_rois2) - i - 1][0])]
                t_indice1_3 = np.arange(t_x1_3, t_x2_3 + 1, 1)
                t_indice2_3 = np.arange(t_y1_3, t_y2_3 + 1, 1)
                tgt_weight_ad_img_3[np.ix_(t_indice2_3, t_indice1_3)] = tgt_weight_gt_3

                t_x1_4 = torch.tensor(math.floor(tgt_gt_rois2[len(tgt_gt_rois2) - i - 1][1] * t_scale_w_4))
                t_y1_4 = torch.tensor(math.floor(tgt_gt_rois2[len(tgt_gt_rois2) - i - 1][2] * t_scale_h_4))
                t_x2_4 = torch.tensor(math.floor(tgt_gt_rois2[len(tgt_gt_rois2) - i - 1][3] * t_scale_w_4))
                t_y2_4 = torch.tensor(math.floor(tgt_gt_rois2[len(tgt_gt_rois2) - i - 1][4] * t_scale_h_4))

                tgt_weight_gt_4 = torch.ones((t_y2_4 - t_y1_4 + 1, t_x2_4 - t_x1_4 + 1)).cuda() * \
                                  norm_weight_vetory[int(tgt_gt_rois2[len(tgt_gt_rois2) - i - 1][0])]

                t_indice1_4 = np.arange(t_x1_4, t_x2_4 + 1, 1)
                t_indice2_4 = np.arange(t_y1_4, t_y2_4 + 1, 1)
                tgt_weight_ad_img_4[np.ix_(t_indice2_4, t_indice1_4)] = tgt_weight_gt_4

                t_x1_5 = torch.tensor(math.floor(tgt_gt_rois2[len(tgt_gt_rois2) - i - 1][1] * t_scale_w_5))
                t_y1_5 = torch.tensor(math.floor(tgt_gt_rois2[len(tgt_gt_rois2) - i - 1][2] * t_scale_h_5))
                t_x2_5 = torch.tensor(math.floor(tgt_gt_rois2[len(tgt_gt_rois2) - i - 1][3] * t_scale_w_5))
                t_y2_5 = torch.tensor(math.floor(tgt_gt_rois2[len(tgt_gt_rois2) - i - 1][4] * t_scale_h_5))

                tgt_weight_gt_5 = torch.ones((t_y2_5 - t_y1_5 + 1, t_x2_5 - t_x1_5 + 1)).cuda() * \
                                  norm_weight_vetory[int(tgt_gt_rois2[len(tgt_gt_rois2) - i - 1][0])]

                t_indice1_5 = np.arange(t_x1_5, t_x2_5 + 1, 1)
                t_indice2_5 = np.arange(t_y1_5, t_y2_5 + 1, 1)
                tgt_weight_ad_img_5[np.ix_(t_indice2_5, t_indice1_5)] = tgt_weight_gt_5

        # print("tgt_image_softmax_sort", tgt_image_softmax_sort)

        source_weight_ad_img_3 = weight_ad_img_3
        source_weight_ad_img_4 = weight_ad_img_4
        source_weight_ad_img_5 = weight_ad_img_5
        common_source_weight_ins = common_source_weight

        target_weight_ad_img_3 = tgt_weight_ad_img_3
        target_weight_ad_img_4 = tgt_weight_ad_img_4
        target_weight_ad_img_5 = tgt_weight_ad_img_5
        common_target_weight_ins = norm_weight_vetory[target_pseudo_label]



        """  DA loss   """




        base_score3 = self.RCNN_imageDA3(conv3_feat)
        DA_img_loss_cls3 = nn.BCELoss(weight=1+source_weight_ad_img_3.view(1, 1, h3, w3).cuda().detach())(base_score3, torch.ones_like(base_score3))

        base_score4 = self.RCNN_imageDA4(conv4_feat)
        DA_img_loss_cls4 = nn.BCELoss(weight=1 + source_weight_ad_img_4.view(1, 1, h4, w4).cuda().detach())(base_score4, torch.ones_like(base_score4))

        base_score = self.RCNN_imageDA(base_feat)
        DA_img_loss_cls = nn.BCELoss(weight=1 + source_weight_ad_img_5.view(1, 1, h5, w5).cuda().detach())(base_score, torch.ones_like(base_score))

        DA_img_loss_cls = (DA_img_loss_cls + DA_img_loss_cls4 + DA_img_loss_cls3) / 3


        instance_sigmoid= self.RCNN_instanceDA(pooled_feat)
        DA_ins_loss_cls = nn.BCELoss(weight=1+common_source_weight_ins)(instance_sigmoid, torch.ones_like(instance_sigmoid))



        """  ************** taget loss ****************  """

        tgt_base_score3 = self.RCNN_imageDA3(tgt_conv3_feat)
        tgt_DA_img_loss_cls3 = nn.BCELoss(weight=1+target_weight_ad_img_3.view(1, 1, th3, tw3))(tgt_base_score3, torch.zeros_like(tgt_base_score3))

        tgt_base_score4 = self.RCNN_imageDA4(tgt_conv4_feat)
        tgt_DA_img_loss_cls4 = nn.BCELoss(weight=1 + target_weight_ad_img_4.view(1, 1, th4, tw4))(tgt_base_score4,torch.zeros_like(tgt_base_score4))

        tgt_base_score = self.RCNN_imageDA(tgt_base_feat)
        tgt_DA_img_loss_cls = nn.BCELoss(weight=1 + target_weight_ad_img_5.view(1, 1, th5, tw5))(tgt_base_score,torch.zeros_like(tgt_base_score))

        tgt_DA_img_loss_cls = (tgt_DA_img_loss_cls + tgt_DA_img_loss_cls4 + tgt_DA_img_loss_cls3) / 3

        tgt_instance_sigmoid = self.RCNN_instanceDA(tgt_pooled_feat)
        tgt_DA_ins_loss_cls = nn.BCELoss(weight=1+common_target_weight_ins)(tgt_instance_sigmoid, torch.zeros_like(tgt_instance_sigmoid))






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
