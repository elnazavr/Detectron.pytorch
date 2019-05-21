# Copyright (c) 2017-present, Facebook, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##############################################################################

"""Collection of available datasets."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os

from core.config import cfg

# Path to data dir
_DATA_DIR = cfg.DATA_DIR

# Required dataset entry keys
IM_DIR = 'image_directory'
ANN_FN = 'annotation_file'

# Optional dataset entry keys
IM_PREFIX = 'image_prefix'
DEVKIT_DIR = 'devkit_directory'
RAW_DIR = 'raw_dir'

# Available datasets
DATASETS = {
    'cityscapes_fine_instanceonly_seg_train': {
        IM_DIR:
            _DATA_DIR + '/cityscapes/images',
        ANN_FN:
            _DATA_DIR + '/cityscapes/annotations/instancesonly_gtFine_train.json',
        RAW_DIR:
            _DATA_DIR + '/cityscapes/raw'
    },
    'cityscapes_fine_instanceonly_seg_val': {
        IM_DIR:
            _DATA_DIR + '/cityscapes/images',
        # use filtered validation as there is an issue converting contours
        ANN_FN:
            _DATA_DIR + '/cityscapes/annotations/instancesonly_filtered_gtFine_val.json',
        RAW_DIR:
            _DATA_DIR + '/cityscapes/raw'
    },
    'cityscapes_fine_instanceonly_seg_test': {
        IM_DIR:
            _DATA_DIR + '/cityscapes/images',
        ANN_FN:
            _DATA_DIR + '/cityscapes/annotations/instancesonly_gtFine_test.json',
        RAW_DIR:
            _DATA_DIR + '/cityscapes/raw'
    },
    'coco_2014_train': {
        IM_DIR:
            _DATA_DIR + '/coco/images/train2014',
        ANN_FN:
            _DATA_DIR + '/coco/annotations/instances_train2014.json'
    },
    'coco_2014_val': {
        IM_DIR:
            _DATA_DIR + '/coco/images/val2014',
        ANN_FN:
            _DATA_DIR + '/coco/annotations/instances_val2014.json'
    },
    'coco_2014_minival': {
        IM_DIR:
            _DATA_DIR + '/coco/images/val2014',
        ANN_FN:
            _DATA_DIR + '/coco/annotations/instances_minival2014.json'
    },
    'coco_2014_valminusminival': {
        IM_DIR:
            _DATA_DIR + '/coco/images/val2014',
        ANN_FN:
            _DATA_DIR + '/coco/annotations/instances_valminusminival2014.json'
    },
    'coco_2015_test': {
        IM_DIR:
            _DATA_DIR + '/coco/images/test2015',
        ANN_FN:
            _DATA_DIR + '/coco/annotations/image_info_test2015.json'
    },
    'coco_2015_test-dev': {
        IM_DIR:
            _DATA_DIR + '/coco/images/test2015',
        ANN_FN:
            _DATA_DIR + '/coco/annotations/image_info_test-dev2015.json'
    },
    'coco_2017_train': {
        IM_DIR:
            _DATA_DIR + '/coco/train2017',
        ANN_FN:
            _DATA_DIR + '/coco/annotations/instances_train2017.json',
    },
    'coco_2017_val': {
        IM_DIR:
            _DATA_DIR + '/coco/val2017',
        ANN_FN:
            _DATA_DIR + '/coco/annotations/instances_val2017.json',
    },
    'coco_2017_val_debug':{
        IM_DIR:
            _DATA_DIR + '/coco/val2017',
        ANN_FN:
            _DATA_DIR + '/coco/annotations/debug_instances_val2017.json',
    },
    'coco_2017_test': {  # 2017 test uses 2015 test images
        IM_DIR:
            _DATA_DIR + '/coco/images/test2015',
        ANN_FN:
            _DATA_DIR + '/coco/annotations/image_info_test2017.json',
        IM_PREFIX:
            'COCO_test2015_'
    },
    'coco_2017_test-dev': {  # 2017 test-dev uses 2015 test images
        IM_DIR:
            _DATA_DIR + '/coco/images/test2015',
        ANN_FN:
            _DATA_DIR + '/coco/annotations/image_info_test-dev2017.json',
        IM_PREFIX:
            'COCO_test2015_'
    },
    'coco_stuff_train': {
        IM_DIR:
            _DATA_DIR + '/coco/images/train2014',
        ANN_FN:
            _DATA_DIR + '/coco/annotations/stuff_train.json'
    },
    'coco_stuff_val': {
        IM_DIR:
            _DATA_DIR + '/coco/images/val2014',
        ANN_FN:
            _DATA_DIR + '/coco/annotations/stuff_val.json'
    },
    'keypoints_coco_2014_train': {
        IM_DIR:
            _DATA_DIR + '/coco/images/train2014',
        ANN_FN:
            _DATA_DIR + '/coco/annotations/person_keypoints_train2014.json'
    },
    'keypoints_coco_2014_val': {
        IM_DIR:
            _DATA_DIR + '/coco/images/val2014',
        ANN_FN:
            _DATA_DIR + '/coco/annotations/person_keypoints_val2014.json'
    },
    'keypoints_coco_2014_minival': {
        IM_DIR:
            _DATA_DIR + '/coco/images/val2014',
        ANN_FN:
            _DATA_DIR + '/coco/annotations/person_keypoints_minival2014.json'
    },
    'keypoints_coco_2014_valminusminival': {
        IM_DIR:
            _DATA_DIR + '/coco/images/val2014',
        ANN_FN:
            _DATA_DIR + '/coco/annotations/person_keypoints_valminusminival2014.json'
    },
    'keypoints_coco_2015_test': {
        IM_DIR:
            _DATA_DIR + '/coco/images/test2015',
        ANN_FN:
            _DATA_DIR + '/coco/annotations/image_info_test2015.json'
    },
    'keypoints_coco_2015_test-dev': {
        IM_DIR:
            _DATA_DIR + '/coco/images/test2015',
        ANN_FN:
            _DATA_DIR + '/coco/annotations/image_info_test-dev2015.json'
    },
    'keypoints_coco_2017_train': {
        IM_DIR:
            _DATA_DIR + '/coco/images/train2017',
        ANN_FN:
            _DATA_DIR + '/coco/annotations/person_keypoints_train2017.json'
    },
    'keypoints_coco_2017_val': {
        IM_DIR:
            _DATA_DIR + '/coco/images/val2017',
        ANN_FN:
            _DATA_DIR + '/coco/annotations/person_keypoints_val2017.json'
    },
    'keypoints_coco_2017_test': {
        IM_DIR:
            _DATA_DIR + '/coco/images/test2017',
        ANN_FN:
            _DATA_DIR + '/coco/annotations/image_info_test2017.json'
    },
    'voc_2007_trainval': {
        IM_DIR:
            _DATA_DIR + '/VOC2007/JPEGImages',
        ANN_FN:
            _DATA_DIR + '/VOC2007/annotations/voc_2007_trainval.json',
        DEVKIT_DIR:
            _DATA_DIR + '/VOC2007/VOCdevkit2007'
    },
    'voc_2007_test': {
        IM_DIR:
            _DATA_DIR + '/VOC2007/JPEGImages',
        ANN_FN:
            _DATA_DIR + '/VOC2007/annotations/voc_2007_test.json',
        DEVKIT_DIR:
            _DATA_DIR + '/VOC2007/VOCdevkit2007'
    },
    'voc_2012_val': {
        IM_DIR:
            _DATA_DIR + '/voc/VOCdevkit/VOC2012/JPEGImages',
        ANN_FN:
            _DATA_DIR + '/voc/VOCdevkit/VOC2012/annotations_coco_style/pascal_val2012.json',
        DEVKIT_DIR:
            _DATA_DIR + '/voc/VOCdevkit/VOC2012/'
    },
    'voc_2012_train': {
        IM_DIR:
            _DATA_DIR + '/voc/VOCdevkit/VOC2012/JPEGImages',
        ANN_FN:
            _DATA_DIR + '/voc/VOCdevkit/VOC2012/annotations_coco_style/pascal_val2012.json',
        DEVKIT_DIR:
            _DATA_DIR + '/voc/VOCdevkit/VOC2012/'
    },
    'coco2017voc2012_train' : {
        IM_DIR:
            _DATA_DIR + '/cocovoc/images',
        ANN_FN:
            _DATA_DIR + '/cocovoc/annotations/coco2017voc2012_train.json',
        DEVKIT_DIR:
            _DATA_DIR + '/voc/VOCdevkit/VOC2012/'
    },
    'coco2017voc2012_val': {
        IM_DIR:
            _DATA_DIR + '/cocovoc/images',
        ANN_FN:
            _DATA_DIR + '/cocovoc/annotations/coco2017voc2012_train.json',
        DEVKIT_DIR:
            _DATA_DIR + '/voc/VOCdevkit/VOC2012/'
    },
    'coco2017_part0_val': {
        IM_DIR:
            _DATA_DIR + '/coco/val2017',
        ANN_FN:
            _DATA_DIR + '/coco/parts/annotations/0_val.json'
    },
    'coco2017_part0_train': {
        IM_DIR:
            _DATA_DIR + '/coco/train2017',
        ANN_FN:
            _DATA_DIR + '/coco/parts/annotations/0_train.json'
    },
    'coco2017_part1_val': {
        IM_DIR:
            _DATA_DIR + '/coco/val2017',
        ANN_FN:
            _DATA_DIR + '/coco/parts/annotations/1_val.json'
    },
    'coco2017_part1_train': {
        IM_DIR:
            _DATA_DIR + '/coco/train2017',
        ANN_FN:
            _DATA_DIR + '/coco/parts/annotations/1_train.json'
    },
    'coco2017_part2_val': {
        IM_DIR:
            _DATA_DIR + '/coco/val2017',
        ANN_FN:
            _DATA_DIR + '/coco/parts/annotations/2_val.json'
    },
    'coco2017_part2_train': {
        IM_DIR:
            _DATA_DIR + '/coco/train2017',
        ANN_FN:
            _DATA_DIR + '/coco/parts/annotations/2_train.json'
    },
    'coco2017_part2_debug': {
        IM_DIR:
            _DATA_DIR + '/coco/train2017',
        ANN_FN:
            _DATA_DIR + '/coco/parts/annotations/2_train_debug.json'
    },
    'coco2017_part1_debug': {
        IM_DIR:
            _DATA_DIR + '/coco/train2017',
        ANN_FN:
            _DATA_DIR + '/coco/parts/annotations/1_train_debug.json'
    },
    'coco2017_part0_debug': {
        IM_DIR:
            _DATA_DIR + '/coco/train2017',
        ANN_FN:
            _DATA_DIR + '/coco/parts/annotations/0_train_debug.json'
    },
    'coco2017_part0_val_coco':{
        IM_DIR:
            _DATA_DIR + '/coco/val2017',
        ANN_FN:
            _DATA_DIR + '/coco/parts/annotations/coco0_val.json'
    },
    'coco2017_part1_val_coco':{
        IM_DIR:
            _DATA_DIR + '/coco/val2017',
        ANN_FN:
            _DATA_DIR + '/coco/parts/annotations/coco1_val.json'
    },
    'coco2017_part2_val_coco':{
        IM_DIR:
            _DATA_DIR + '/coco/val2017',
        ANN_FN:
            _DATA_DIR + '/coco/parts/annotations/coco2_val.json'
    },
    'coco2017_part0_train_intersected':{
        IM_DIR:
            _DATA_DIR + '/coco/train2017',
        ANN_FN:
            _DATA_DIR + '/coco/intresected_parts/annotations/0_train.json'
    },
    'coco2017_part1_train_intersected':{
        IM_DIR:
            _DATA_DIR + '/coco/train2017',
        ANN_FN:
            _DATA_DIR + '/coco/intresected_parts/annotations/1_train.json'
    },
    'coco2017_part2_train_intersected':{
        IM_DIR:
            _DATA_DIR + '/coco/train2017',
        ANN_FN:
            _DATA_DIR + '/coco/intresected_parts/annotations/2_train.json'
    },
    'coco2017_part0_val_intersected':{
        IM_DIR:
            _DATA_DIR + '/coco/val2017',
        ANN_FN:
            _DATA_DIR + '/coco/intresected_parts/annotations/coco0_val.json'
    },
    'coco2017_part1_val_intersected':{
        IM_DIR:
            _DATA_DIR + '/coco/val2017',
        ANN_FN:
            _DATA_DIR + '/coco/intresected_parts/annotations/coco1_val.json'
    },
    'coco2017_part2_val_intersected':{
        IM_DIR:
            _DATA_DIR + '/coco/val2017',
        ANN_FN:
            _DATA_DIR + '/coco/intresected_parts/annotations/coco2_val.json'
    }
}
