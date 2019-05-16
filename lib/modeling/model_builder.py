from functools import wraps
import importlib
import logging
import numpy as np
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import faiss

from core.config import cfg
from model.roi_pooling.functions.roi_pool import RoIPoolFunction
from model.roi_crop.functions.roi_crop import RoICropFunction
from modeling.roi_xfrom.roi_align.functions.roi_align import RoIAlignFunction
import modeling.rpn_heads as rpn_heads
import modeling.fast_rcnn_heads as fast_rcnn_heads
import modeling.mask_rcnn_heads as mask_rcnn_heads
import modeling.keypoint_rcnn_heads as keypoint_rcnn_heads
import utils.blob as blob_utils
import utils.net as net_utils
import utils.resnet_weights_helper as resnet_utils
import utils.fpn as fpn_utils
import utils.boxes as box_utils
logger = logging.getLogger(__name__)
OBJECTNESS_THRESHOLD = 0.5


def get_func(func_name):
    """Helper to return a function object by name. func_name must identify a
    function in this module or the path to a function relative to the base
    'modeling' module.
    """
    if func_name == '':
        return None
    try:
        parts = func_name.split('.')
        # Refers to a function in this module
        if len(parts) == 1:
            return globals()[parts[0]]
        # Otherwise, assume we're referencing a module under modeling
        module_name = 'modeling.' + '.'.join(parts[:-1])
        module = importlib.import_module(module_name)
        return getattr(module, parts[-1])
    except Exception:
        logger.error('Failed to find function: %s', func_name)
        raise


def compare_state_dict(sa, sb):
    if sa.keys() != sb.keys():
        return False
    for k, va in sa.items():
        if not torch.equal(va, sb[k]):
            return False
    return True


def check_inference(net_func):
    @wraps(net_func)
    def wrapper(self, *args, **kwargs):
        if not self.training:
            if cfg.PYTORCH_VERSION_LESS_THAN_040:
                return net_func(self, *args, **kwargs)
            else:
                with torch.no_grad():
                    return net_func(self, *args, **kwargs)
        else:
            raise ValueError('You should call this function only on inference.'
                              'Set the network in inference mode by net.eval().')

    return wrapper


class Generalized_RCNN(nn.Module):
    def __init__(self):
        super().__init__()

        # For cache
        self.mapping_to_detectron = None
        self.orphans_in_detectron = None

        # Backbone for feature extraction
        self.Conv_Body = get_func(cfg.MODEL.CONV_BODY)()

        # Region Proposal Network
        if cfg.RPN.RPN_ON:
            self.RPN = rpn_heads.generic_rpn_outputs(
                self.Conv_Body.dim_out, self.Conv_Body.spatial_scale)

        if cfg.FPN.FPN_ON:
            # Only supports case when RPN and ROI min levels are the same
            assert cfg.FPN.RPN_MIN_LEVEL == cfg.FPN.ROI_MIN_LEVEL
            # RPN max level can be >= to ROI max level
            assert cfg.FPN.RPN_MAX_LEVEL >= cfg.FPN.ROI_MAX_LEVEL
            # FPN RPN max level might be > FPN ROI max level in which case we
            # need to discard some leading conv blobs (blobs are ordered from
            # max/coarsest level to min/finest level)
            self.num_roi_levels = cfg.FPN.ROI_MAX_LEVEL - cfg.FPN.ROI_MIN_LEVEL + 1

            # Retain only the spatial scales that will be used for RoI heads. `Conv_Body.spatial_scale`
            # may include extra scales that are used for RPN proposals, but not for RoI heads.
            self.Conv_Body.spatial_scale = self.Conv_Body.spatial_scale[-self.num_roi_levels:]

        # BBOX Branch
        if not cfg.MODEL.RPN_ONLY:
            self.Box_Head = get_func(cfg.FAST_RCNN.ROI_BOX_HEAD)(
                self.RPN.dim_out, self.roi_feature_transform, self.Conv_Body.spatial_scale)
            self.Box_Outs = fast_rcnn_heads.fast_rcnn_outputs(
                self.Box_Head.dim_out)

        # Mask Branch
        if cfg.MODEL.MASK_ON:
            self.Mask_Head = get_func(cfg.MRCNN.ROI_MASK_HEAD)(
                self.RPN.dim_out, self.roi_feature_transform, self.Conv_Body.spatial_scale)
            if getattr(self.Mask_Head, 'SHARE_RES5', False):
                self.Mask_Head.share_res5_module(self.Box_Head.res5)
            self.Mask_Outs = mask_rcnn_heads.mask_rcnn_outputs(self.Mask_Head.dim_out)

        # Keypoints Branch
        if cfg.MODEL.KEYPOINTS_ON:
            self.Keypoint_Head = get_func(cfg.KRCNN.ROI_KEYPOINTS_HEAD)(
                self.RPN.dim_out, self.roi_feature_transform, self.Conv_Body.spatial_scale)
            if getattr(self.Keypoint_Head, 'SHARE_RES5', False):
                self.Keypoint_Head.share_res5_module(self.Box_Head.res5)
            self.Keypoint_Outs = keypoint_rcnn_heads.keypoint_outputs(self.Keypoint_Head.dim_out)

        self._init_modules()

    def _init_modules(self):
        if cfg.MODEL.LOAD_IMAGENET_PRETRAINED_WEIGHTS:
            resnet_utils.load_pretrained_imagenet_weights(self)
            # Check if shared weights are equaled
            if cfg.MODEL.MASK_ON and getattr(self.Mask_Head, 'SHARE_RES5', False):
                assert compare_state_dict(self.Mask_Head.res5.state_dict(), self.Box_Head.res5.state_dict())
            if cfg.MODEL.KEYPOINTS_ON and getattr(self.Keypoint_Head, 'SHARE_RES5', False):
                assert compare_state_dict(self.Keypoint_Head.res5.state_dict(), self.Box_Head.res5.state_dict())

        if cfg.TRAIN.FREEZE_CONV_BODY:
            for p in self.Conv_Body.parameters():
                p.requires_grad = False

    def forward(self, data, im_info,
                 roidb=None,
                 only_bbox=None,
                 image_to_idx=None,
                 bbbp=False,
                 dataset_to_classes= {},
                 C = set(),
                 classes_faiss = None,
                 dataset_idx_to_classes = None,
                 median_distance_class = None,
                **rpn_kwargs):
        if cfg.PYTORCH_VERSION_LESS_THAN_040:
            return self._forward(data, im_info,
                  roidb,
                 only_bbox,
                 image_to_idx,
                 bbbp,
                 dataset_to_classes,
                 C,
                 classes_faiss,
                 dataset_idx_to_classes,
                 median_distance_class,
                **rpn_kwargs)
        else:
            with torch.set_grad_enabled(self.training):
                return self._forward(
                 data,
                 im_info,
                 roidb,
                 only_bbox,
                 image_to_idx,
                 bbbp,
                 dataset_to_classes,
                 C,
                 classes_faiss,
                 dataset_idx_to_classes,
                 median_distance_class,
                **rpn_kwargs)


    def _forward(self, data,
                 im_info,
                 roidb=None,
                 only_bbox=None,
                 image_to_idx=None,
                 bbbp=False,
                 dataset_to_classes= {},
                 C = set(),
                 classes_faiss = None,
                 dataset_idx_to_classes = None,
                 median_distance_class = None,

                 **rpn_kwargs):
        im_data = data
        if self.training or only_bbox:
            roidb = list(map(lambda x: blob_utils.deserialize(x)[0], roidb))

        device_id = im_data.get_device()

        return_dict = {}  # A dict to collect return variables
        blob_conv = self.Conv_Body(im_data)
        if not only_bbox:
            rpn_ret = self.RPN(blob_conv, im_info, roidb)
        else:
            lvl_min = cfg.FPN.ROI_MIN_LEVEL
            lvl_max = cfg.FPN.ROI_MAX_LEVEL
            if cfg.FPN.FPN_ON:
                # Retain only the blobs that will be used for RoI heads. `blob_conv` may include
                # extra blobs that are used for RPN proposals, but not for RoI heads.
                blob_conv = blob_conv[-self.num_roi_levels:]
            return_dict["ground_truth"] = []
            return_dict["bboxes"] = []
            for i in range(len(roidb)):
                rpn_ret = {}
                for lvl in range(lvl_min, lvl_max + 1):
                    rpn_ret["rois_fpn" + str(lvl)] = []
                target_lvls = fpn_utils.map_rois_to_fpn_levels(roidb[i]["boxes"],lvl_min, lvl_max)
                boxes = np.array(list(map(lambda x: np.append([i], x), roidb[i]["boxes"])))
                fpn_utils.add_multilevel_roi_blobs(rpn_ret, 'rois', boxes, target_lvls, lvl_min, lvl_max)
                for key in rpn_ret.keys():
                    rpn_ret[key] = np.array(rpn_ret[key])

                box_feat = self.Box_Head(blob_conv, rpn_ret)
                return_dict["ground_truth"].append(box_feat)
                return_dict["bboxes"].append(roidb[i]["boxes"])
            return return_dict



        # if self.training:
        #     # can be used to infer fg/bg ratio
        #     return_dict['rois_label'] = rpn_ret['labels_int32']

        if cfg.FPN.FPN_ON:
            # Retain only the blobs that will be used for RoI heads. `blob_conv` may include
            # extra blobs that are used for RPN proposals, but not for RoI heads.
            blob_conv = blob_conv[-self.num_roi_levels:]

        if not self.training:
            return_dict['blob_conv'] = blob_conv
        if not cfg.MODEL.RPN_ONLY:
            if cfg.MODEL.SHARE_RES5 and self.training:
                box_feat, res5_feat = self.Box_Head(blob_conv, rpn_ret)
            else:
                box_feat = self.Box_Head(blob_conv, rpn_ret)
            cls_score, bbox_pred = self.Box_Outs(box_feat)
        else:
            # TODO: complete the returns for RPN only situation
            pass

        if bbbp and self.training:
            cls_score_np = F.softmax(cls_score).detach().cpu().numpy()

            preidcted_classes = np.argmax(cls_score_np, axis=1)
            objective_scores = rpn_ret['objective_scores']



            preidcted_features = box_feat
            indecies_to_drop = []
            dataset_idx = roidb[0]["dataset_idx"]

            c_plus = dataset_to_classes[dataset_idx]
            c_minus = set(C) - set(c_plus)
            for proposal_idx, proposal_predicted_class in enumerate(preidcted_classes):
                if proposal_predicted_class != 0:
                    if proposal_predicted_class in c_minus and objective_scores[proposal_idx] > OBJECTNESS_THRESHOLD:
                        feature = preidcted_features[proposal_idx]
                        nn_distance, nn_idx, chosen_dataset = np.inf, None, None

                        for other_dataset_idx in classes_faiss.keys():
                            if other_dataset_idx!=dataset_idx:
                                nn_distance_other, nn_idx_other = classes_faiss[other_dataset_idx].search(np.array([feature.detach().cpu().numpy().astype(np.float32)]), 1)
                                if nn_distance_other < nn_distance:
                                    nn_distance = nn_distance_other
                                    nn_idx = nn_idx_other
                                    chosen_dataset = other_dataset_idx

                        if proposal_predicted_class == dataset_idx_to_classes[chosen_dataset][nn_idx] \
                                and nn_distance < median_distance_class[proposal_predicted_class]:
                            rpn_ret['labels_int32'][proposal_idx] = -1
                            indecies_to_drop.append(proposal_idx)

            if len(indecies_to_drop)>0:
                indecies_to_drop_cumalitve = rpn_ret["rois_idx_restore_int32"][indecies_to_drop]
                lens = []
                levels_to_idx = {}
                for i, lvl in enumerate(range(cfg.FPN.RPN_MIN_LEVEL, cfg.FPN.RPN_MAX_LEVEL + 1)):
                    if "rois_fpn" + str(lvl) in rpn_ret.keys():
                        lens.append(len(rpn_ret["rois_fpn" + str(lvl)]))
                        levels_to_idx[lvl] = []
                cumulative_lens = [np.sum(lens[:i + 1]) for i in range(len(lens))]
                for idx_roi, idx_cum in zip(indecies_to_drop, indecies_to_drop_cumalitve):
                    for idx_lvl in range(len(cumulative_lens)):
                        if idx_cum < cumulative_lens[idx_lvl]:
                            levels_to_idx[cfg.FPN.RPN_MIN_LEVEL + idx_lvl].append(idx_roi)
                            break;
                for lvl in levels_to_idx.keys():
                    _, A, H, W = rpn_ret["rpn_cls_logits_fpn" + str(lvl)].shape
                    if len(levels_to_idx[lvl]) > 0:
                        for roi_idx in levels_to_idx[lvl]:
                            idx = rpn_ret["indecies_anchors"][roi_idx]
                            if idx != -1:
                                h, w, a = get_hwa(idx, A, W)
                                rpn_kwargs["rpn_labels_int32_wide_fpn" + str(lvl)][0, a, h, w] = -1
                            #distance, idx = classes_faiss[predicted_class].search(
                            #    np.array([feature.detach().cpu().numpy().astype(np.float32)]), 1)
                            #if predicted_class == dataset_idx_to_classes[dataset_idx][idx] and distance < median_distance_class[predicted_class]:
                            #    rpn_ret['labels_int32'][idx] = -1

        if self.training:
            return_dict['losses'] = {}
            return_dict['metrics'] = {}
            # rpn loss
            rpn_kwargs.update(dict(
                (k, rpn_ret[k]) for k in rpn_ret.keys()
                if (k.startswith('rpn_cls_logits') or k.startswith('rpn_bbox_pred'))
            ))
            loss_rpn_cls, loss_rpn_bbox = rpn_heads.generic_rpn_losses(**rpn_kwargs)
            if cfg.FPN.FPN_ON:
                for i, lvl in enumerate(range(cfg.FPN.RPN_MIN_LEVEL, cfg.FPN.RPN_MAX_LEVEL + 1)):
                    return_dict['losses']['loss_rpn_cls_fpn%d' % lvl] = loss_rpn_cls[i]
                    return_dict['losses']['loss_rpn_bbox_fpn%d' % lvl] = loss_rpn_bbox[i]
            else:
                return_dict['losses']['loss_rpn_cls'] = loss_rpn_cls
                return_dict['losses']['loss_rpn_bbox'] = loss_rpn_bbox

            # bbox loss
            loss_cls, loss_bbox, accuracy_cls = fast_rcnn_heads.fast_rcnn_losses(
                cls_score, bbox_pred, rpn_ret['labels_int32'], rpn_ret['bbox_targets'],
                rpn_ret['bbox_inside_weights'], rpn_ret['bbox_outside_weights'])
            return_dict['losses']['loss_cls'] = loss_cls
            return_dict['losses']['loss_bbox'] = loss_bbox
            return_dict['metrics']['accuracy_cls'] = accuracy_cls

            if cfg.MODEL.MASK_ON:
                if getattr(self.Mask_Head, 'SHARE_RES5', False):
                    mask_feat = self.Mask_Head(res5_feat, rpn_ret,
                                               roi_has_mask_int32=rpn_ret['roi_has_mask_int32'])
                else:
                    mask_feat = self.Mask_Head(blob_conv, rpn_ret)
                mask_pred = self.Mask_Outs(mask_feat)
                # return_dict['mask_pred'] = mask_pred
                # mask loss
                loss_mask = mask_rcnn_heads.mask_rcnn_losses(mask_pred, rpn_ret['masks_int32'])
                return_dict['losses']['loss_mask'] = loss_mask

            if cfg.MODEL.KEYPOINTS_ON:
                if getattr(self.Keypoint_Head, 'SHARE_RES5', False):
                    # No corresponding keypoint head implemented yet (Neither in Detectron)
                    # Also, rpn need to generate the label 'roi_has_keypoints_int32'
                    kps_feat = self.Keypoint_Head(res5_feat, rpn_ret,
                                                  roi_has_keypoints_int32=rpn_ret['roi_has_keypoint_int32'])
                else:
                    kps_feat = self.Keypoint_Head(blob_conv, rpn_ret)
                kps_pred = self.Keypoint_Outs(kps_feat)
                # return_dict['keypoints_pred'] = kps_pred
                # keypoints loss
                if cfg.KRCNN.NORMALIZE_BY_VISIBLE_KEYPOINTS:
                    loss_keypoints = keypoint_rcnn_heads.keypoint_losses(
                        kps_pred, rpn_ret['keypoint_locations_int32'], rpn_ret['keypoint_weights'])
                else:
                    loss_keypoints = keypoint_rcnn_heads.keypoint_losses(
                        kps_pred, rpn_ret['keypoint_locations_int32'], rpn_ret['keypoint_weights'],
                        rpn_ret['keypoint_loss_normalizer'])
                return_dict['losses']['loss_kps'] = loss_keypoints

            # pytorch0.4 bug on gathering scalar(0-dim) tensors
            for k, v in return_dict['losses'].items():
                return_dict['losses'][k] = v.unsqueeze(0)
            for k, v in return_dict['metrics'].items():
                return_dict['metrics'][k] = v.unsqueeze(0)

        else:
            # Testing
            return_dict['rois'] = rpn_ret['rois']
            return_dict['cls_score'] = cls_score
            return_dict['bbox_pred'] = bbox_pred
            return_dict['box_feat'] = box_feat

        return return_dict

    def roi_feature_transform(self, blobs_in, rpn_ret, blob_rois='rois', method='RoIPoolF',
                              resolution=7, spatial_scale=1. / 16., sampling_ratio=0):
        """Add the specified RoI pooling method. The sampling_ratio argument
        is supported for some, but not all, RoI transform methods.

        RoIFeatureTransform abstracts away:
          - Use of FPN or not
          - Specifics of the transform method
        """
        assert method in {'RoIPoolF', 'RoICrop', 'RoIAlign'}, \
            'Unknown pooling method: {}'.format(method)

        if isinstance(blobs_in, list):
            # FPN case: add RoIFeatureTransform to each FPN level
            device_id = blobs_in[0].get_device()
            k_max = cfg.FPN.ROI_MAX_LEVEL  # coarsest level of pyramid
            k_min = cfg.FPN.ROI_MIN_LEVEL  # finest level of pyramid
            assert len(blobs_in) == k_max - k_min + 1
            bl_out_list = []
            for lvl in range(k_min, k_max + 1):
                bl_in = blobs_in[k_max - lvl]  # blobs_in is in reversed order
                sc = spatial_scale[k_max - lvl]  # in reversed order
                bl_rois = blob_rois + '_fpn' + str(lvl)
                if len(rpn_ret[bl_rois]):
                    rois = Variable(torch.from_numpy(rpn_ret[bl_rois])).cuda(device_id)
                    if method == 'RoIPoolF':
                        # Warning!: Not check if implementation matches Detectron
                        xform_out = RoIPoolFunction(resolution, resolution, sc)(bl_in, rois)
                    elif method == 'RoICrop':
                        # Warning!: Not check if implementation matches Detectron
                        grid_xy = net_utils.affine_grid_gen(
                            rois, bl_in.size()[2:], self.grid_size)
                        grid_yx = torch.stack(
                            [grid_xy.data[:, :, :, 1], grid_xy.data[:, :, :, 0]], 3).contiguous()
                        xform_out = RoICropFunction()(bl_in, Variable(grid_yx).detach())
                        if cfg.CROP_RESIZE_WITH_MAX_POOL:
                            xform_out = F.max_pool2d(xform_out, 2, 2)
                    elif method == 'RoIAlign':
                        if rois.type()== 'torch.cuda.DoubleTensor':
                            rois = rois.type(torch.cuda.FloatTensor)
                        xform_out = RoIAlignFunction(
                            resolution, resolution, sc, sampling_ratio)(bl_in, rois)
                    bl_out_list.append(xform_out)
            # The pooled features from all levels are concatenated along the
            # batch dimension into a single 4D tensor.
            xform_shuffled = torch.cat(bl_out_list, dim=0)

            # Unshuffle to match rois from dataloader
            device_id = xform_shuffled.get_device()
            restore_bl = rpn_ret[blob_rois + '_idx_restore_int32']
            restore_bl = Variable(
                torch.from_numpy(restore_bl.astype('int64', copy=False))).cuda(device_id)
            xform_out = xform_shuffled[restore_bl]
        else:
            # Single feature level
            # rois: holds R regions of interest, each is a 5-tuple
            # (batch_idx, x1, y1, x2, y2) specifying an image batch index and a
            # rectangle (x1, y1, x2, y2)
            device_id = blobs_in.get_device()
            rois = Variable(torch.from_numpy(rpn_ret[blob_rois])).cuda(device_id)
            if method == 'RoIPoolF':
                xform_out = RoIPoolFunction(resolution, resolution, spatial_scale)(blobs_in, rois)
            elif method == 'RoICrop':
                grid_xy = net_utils.affine_grid_gen(rois, blobs_in.size()[2:], self.grid_size)
                grid_yx = torch.stack(
                    [grid_xy.data[:, :, :, 1], grid_xy.data[:, :, :, 0]], 3).contiguous()
                xform_out = RoICropFunction()(blobs_in, Variable(grid_yx).detach())
                if cfg.CROP_RESIZE_WITH_MAX_POOL:
                    xform_out = F.max_pool2d(xform_out, 2, 2)
            elif method == 'RoIAlign':
                xform_out = RoIAlignFunction(
                    resolution, resolution, spatial_scale, sampling_ratio)(blobs_in, rois)

        return xform_out

    @check_inference
    def convbody_net(self, data):
        """For inference. Run Conv Body only"""
        blob_conv = self.Conv_Body(data)
        if cfg.FPN.FPN_ON:
            # Retain only the blobs that will be used for RoI heads. `blob_conv` may include
            # extra blobs that are used for RPN proposals, but not for RoI heads.
            blob_conv = blob_conv[-self.num_roi_levels:]
        return blob_conv

    @check_inference
    def mask_net(self, blob_conv, rpn_blob):
        """For inference"""
        mask_feat = self.Mask_Head(blob_conv, rpn_blob)
        mask_pred = self.Mask_Outs(mask_feat)
        return mask_pred

    @check_inference
    def keypoint_net(self, blob_conv, rpn_blob):
        """For inference"""
        kps_feat = self.Keypoint_Head(blob_conv, rpn_blob)
        kps_pred = self.Keypoint_Outs(kps_feat)
        return kps_pred

    @property
    def detectron_weight_mapping(self):
        if self.mapping_to_detectron is None:
            d_wmap = {}  # detectron_weight_mapping
            d_orphan = []  # detectron orphan weight list
            for name, m_child in self.named_children():
                if list(m_child.parameters()):  # if module has any parameter
                    child_map, child_orphan = m_child.detectron_weight_mapping()
                    d_orphan.extend(child_orphan)
                    for key, value in child_map.items():
                        new_key = name + '.' + key
                        d_wmap[new_key] = value
            self.mapping_to_detectron = d_wmap
            self.orphans_in_detectron = d_orphan

        return self.mapping_to_detectron, self.orphans_in_detectron

    def _add_loss(self, return_dict, key, value):
        """Add loss tensor to returned dictionary"""
        return_dict['losses'][key] = value



def find_threhold_for_each_class(db, classes,  k_neighbours=10):
    index = create_db(db)
    distance, indecies = index.search(db, k_neighbours)
    classes_idx = classes[indecies]
    average_distance_sample = []
    for idx, neighbours in enumerate(classes_idx):
        myself = neighbours[0]
        not_class_neighbours = np.where(neighbours!=myself)[0]
        first_not_class_neighbours = not_class_neighbours[0]
        average_distance_sample.append(distance[idx, first_not_class_neighbours])
    average_distance_sample = np.array(average_distance_sample)
    average_distance_class = {}
    for class_idx in set(classes):
        average_distance_class[class_idx] = np.median(average_distance_sample[np.where(classes==class_idx)])
    return average_distance_class


def create_db(db):
    dimension = 1024
    db = db.astype('float32')
    faiss_db = faiss.IndexFlatL2(dimension)
    faiss_db.add(db)
    return faiss_db


def find_closest_class_for_background(faiss_db, db,  looked_features, threholds,  k_neighbours=10):
    distance, indecies = faiss_db.search(looked_features, k_neighbours)




def get_hwa(idx, A,W):
    h = idx // (A*W)
    w = (idx- h*A*W)//A
    a = idx - h*A*W - w*A
    return h,w,a

