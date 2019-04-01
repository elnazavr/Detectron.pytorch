""" Training Script """



import argparse
import distutils.util
import os
import sys
import pickle
import resource
import traceback
import logging
from collections import defaultdict
os.environ["CUDA_VISIBLE_DEVICES"]="1"


import numpy as np
import yaml
import torch
from torch.autograd import Variable
import torch.nn as nn
import cv2
cv2.setNumThreads(0)  # pytorch issue 1355: possible deadlock in dataloader

import _init_paths  # pylint: disable=unused-import
sys.path.append("/home.nfs/babayeln/doc/utils")
sys.path.append("/home.nfs/babayeln/doc/lib/datasets")


import nn as mynn
import utils.net as net_utils
import utils.misc as misc_utils
from core.config import cfg, cfg_from_file, cfg_from_list, assert_and_infer_cfg
from datasets.roidb import combined_roidb_for_training
from modeling.model_builder import Generalized_RCNN
from roi_data.loader import RoiDataLoader, MinibatchSampler, collate_minibatch, collate_minibatch2
from utils.detectron_weight_helper import load_detectron_weight
from utils.logging import log_stats
from utils.timer import Timer
from utils.training_stats import TrainingStats
import utils.blob as blob_utils
import faiss

# OpenCL may be enabled by default in OpenCV3; disable it because it's not
# thread safe and causes unwanted GPU memory allocations.
cv2.ocl.setUseOpenCL(False)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# RuntimeError: received 0 items of ancdata. Issue: pytorch/pytorch#973
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))


def parse_args():
    """Parse input arguments"""
    parser = argparse.ArgumentParser(description='Train a X-RCNN network')

    parser.add_argument(
        '--dataset', dest='dataset', required=False,
        help='Dataset to use')
    parser.add_argument(
        '--cfg', dest='cfg_file', required=True,
        help='Config file for training (and optionally testing)')
    parser.add_argument(
        '--set', dest='set_cfgs',
        help='Set config keys. Key value sequence seperate by whitespace.'
             'e.g. [key] [value] [key] [value]',
        default=[], nargs='+')

    parser.add_argument(
        '--disp_interval',
        help='Display training info every N iterations',
        default=100, type=int)
    parser.add_argument(
        '--no_cuda', dest='cuda', help='Do not use CUDA device', action='store_false')

    # Optimization
    # These options has the highest prioity and can overwrite the values in config file
    # or values set by set_cfgs. `None` means do not overwrite.
    parser.add_argument(
        '--bs', dest='batch_size',
        help='Explicitly specify to overwrite the value comed from cfg_file.',
        type=int)
    parser.add_argument(
        '--nw', dest='num_workers',
        help='Explicitly specify to overwrite number of workers to load data. Defaults to 4',
        type=int)

    parser.add_argument(
        '--o', dest='optimizer', help='Training optimizer.',
        default=None)
    parser.add_argument(
        '--lr', help='Base learning rate.',
        default=None, type=float)
    parser.add_argument(
        '--lr_decay_gamma',
        help='Learning rate decay rate.',
        default=None, type=float)
    parser.add_argument(
        '--lr_decay_epochs',
        help='Epochs to decay the learning rate on. '
             'Decay happens on the beginning of a epoch. '
             'Epoch is 0-indexed.',
        default=[4, 5], nargs='+', type=int)

    # Epoch
    parser.add_argument(
        '--start_iter',
        help='Starting iteration for first training epoch. 0-indexed.',
        default=0, type=int)
    parser.add_argument(
        '--start_epoch',
        help='Starting epoch count. Epoch is 0-indexed.',
        default=0, type=int)
    parser.add_argument(
        '--epochs', dest='num_epochs',
        help='Number of epochs to train',
        default=6, type=int)

    # Resume training: requires same iterations per epoch
    parser.add_argument(
        '--resume',
        help='resume to training on a checkpoint',
        action='store_true')

    parser.add_argument(
        '--no_save', help='do not save anything', action='store_true')

    parser.add_argument(
        '--ckpt_num_per_epoch',
        help='number of checkpoints to save in each epoch. '
             'Not include the one at the end of an epoch.',
        default=3, type=int)

    parser.add_argument(
        '--load_ckpt', help='checkpoint path to load')
    parser.add_argument(
        '--load_detectron', help='path to the detectron weight pickle file')

    parser.add_argument(
        '--use_tfboard', help='Use tensorflow tensorboard to log training info',
        action='store_true')

    parser.add_argument(
        '--bbbp', help='Use tensorflow tensorboard to log training info',
        action='store_true')

    return parser.parse_args()



def main():
    """Main function"""

    args = parse_args()
    #XXXXXXXXXXXXXXXXXXXXXXXXXXX
    #args.cfg_filename = "configs/baselines/e2e_faster_rcnn_R-50-FPN_2x_COCO_part123.yaml"
    #XXXXXXXXXXXXXXXXXXXXXXXXXXX

    print('Called with args:')
    print(args)

    if not torch.cuda.is_available():
        sys.exit("Need a CUDA device to run the code.")

    if args.cuda or cfg.NUM_GPUS > 0:
        cfg.CUDA = True
    else:
        raise ValueError("Need Cuda device to run !")


    cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)


    ### Adaptively adjust some configs ##
    original_batch_size = cfg.NUM_GPUS * cfg.TRAIN.IMS_PER_BATCH
    if args.batch_size is None:
        args.batch_size = original_batch_size
    cfg.NUM_GPUS = torch.cuda.device_count()
    assert (args.batch_size % cfg.NUM_GPUS) == 0, \
        'batch_size: %d, NUM_GPUS: %d' % (args.batch_size, cfg.NUM_GPUS)
    cfg.TRAIN.IMS_PER_BATCH = args.batch_size // cfg.NUM_GPUS
    print('Batch size change from {} (in config file) to {}'.format(
        original_batch_size, args.batch_size))
    print('NUM_GPUs: %d, TRAIN.IMS_PER_BATCH: %d' % (cfg.NUM_GPUS, cfg.TRAIN.IMS_PER_BATCH))

    if args.num_workers is not None:
        cfg.DATA_LOADER.NUM_THREADS = args.num_workers
    print('Number of data loading threads: %d' % cfg.DATA_LOADER.NUM_THREADS)

    ### Adjust learning based on batch size change linearly
    old_base_lr = cfg.SOLVER.BASE_LR
    cfg.SOLVER.BASE_LR *= args.batch_size / original_batch_size
    print('Adjust BASE_LR linearly according to batch size change: {} --> {}'.format(
        old_base_lr, cfg.SOLVER.BASE_LR))

    if len(cfg.TRAIN.DATASETS)==0 or cfg.MODEL.NUM_CLASSES is None:
        raise ValueError("There is no dataset in cfg train")

    ### Overwrite some solver settings from command line arguments
    if args.optimizer is not None:
        cfg.SOLVER.TYPE = args.optimizer
    if args.lr is not None:
        cfg.SOLVER.BASE_LR = args.lr
    if args.lr_decay_gamma is not None:
        cfg.SOLVER.GAMMA = args.lr_decay_gamma

    output_dir = cfg.OUTPUT_DIR
    args.run_name = misc_utils.get_run_name()
    if output_dir is None:
        raise ValueError("Output dir is not detected in cfg")

    timers = defaultdict(Timer)

    ### Dataset ###
    timers['roidb'].tic()
    feature_db = np.empty((860001,1031), dtype=np.float32)
    image_to_idx = {}
    """
    labels: Image_id 0 , Dataset 1, Class 2, Bbox 3, feature
    dim: 1, 1, 1, 4, 1024 = 1031
    """
    ground_truth_roidb =[]
    roidb, ratio_list, ratio_index, feature_db, dataset_to_classes = combined_roidb_for_training(
        cfg.TRAIN.DATASETS, cfg.TRAIN.PROPOSAL_FILES, feature_db=feature_db,
        ground_truth_roidb = ground_truth_roidb, image_to_idx = image_to_idx)

    # roidb_new = []
    # import copy
    # for roi in roidb:
    #     roi_new = copy.deepcopy(roi)
    #     roi_new["dataset"] = -1
    #     roidb_new.append(roi_new)

    #np.save(os.path.join(output_dir, "roidb_initial" + ".pkl"), roidb_new)
    #np.save(os.path.join(output_dir, "feature_db_val_initial" + ".pkl"), feature_db)

    timers['roidb'].toc()
    train_size = len(roidb)
    logger.info('{:d} roidb entries'.format(train_size))
    logger.info('Takes %.2f sec(s) to construct roidb', timers['roidb'].average_time)


    sampler = MinibatchSampler(ratio_list, ratio_index)

    dataset = RoiDataLoader(
        roidb,
        cfg.MODEL.NUM_CLASSES,
        training=True)


    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        #num_workers=cfg.DATA_LOADER.NUM_THREADS,
        collate_fn=collate_minibatch)


    dataset_groundtruth = RoiDataLoader(
        roidb= roidb,
        num_classes=cfg.MODEL.NUM_CLASSES,
        training=False
    )
    dataloader_groundtruth = torch.utils.data.DataLoader(
        dataset_groundtruth,
        batch_size=args.batch_size,
        sampler=sampler,
        #num_workers=cfg.DATA_LOADER.NUM_THREADS,
        collate_fn=collate_minibatch)

    assert_and_infer_cfg()

    ### Model ###
    maskRCNN = Generalized_RCNN()

    if cfg.CUDA:
        maskRCNN.cuda()

    ### Optimizer ###
    bias_params = []
    nonbias_params = []
    for key, value in dict(maskRCNN.named_parameters()).items():
        if value.requires_grad:
            if 'bias' in key:
                bias_params.append(value)
            else:
                nonbias_params.append(value)
    params = [
        {'params': nonbias_params,
         'lr': cfg.SOLVER.BASE_LR,
         'weight_decay': cfg.SOLVER.WEIGHT_DECAY},
        {'params': bias_params,
         'lr': cfg.SOLVER.BASE_LR * (cfg.SOLVER.BIAS_DOUBLE_LR + 1),
         'weight_decay': cfg.SOLVER.WEIGHT_DECAY if cfg.SOLVER.BIAS_WEIGHT_DECAY else 0}
    ]

    if cfg.SOLVER.TYPE == "SGD":
        optimizer = torch.optim.SGD(params, momentum=cfg.SOLVER.MOMENTUM)
    elif cfg.SOLVER.TYPE == "Adam":
        optimizer = torch.optim.Adam(params)

    ### Load checkpoint
    if args.load_ckpt:
        load_name = args.load_ckpt
        logging.info("loading checkpoint %s", load_name)
        checkpoint = torch.load(load_name, map_location=lambda storage, loc: storage)
        net_utils.load_ckpt(maskRCNN, checkpoint['model'])
        if args.resume:
            print(checkpoint['iters_per_epoch'], train_size // args.batch_size)
            #assert checkpoint['iters_per_epoch'] == train_size // args.batch_size, \
            #v    "iters_per_epoch should match for resume"
            # There is a bug in optimizer.load_state_dict on Pytorch 0.3.1.
            # However it's fixed on master.
            # optimizer.load_state_dict(checkpoint['optimizer'])
            misc_utils.load_optimizer_state_dict(optimizer, checkpoint['optimizer'])
            if checkpoint['step'] == (checkpoint['iters_per_epoch'] - 1):
                # Resume from end of an epoch
                args.start_epoch = checkpoint['epoch'] + 1
                args.start_iter = 0
            else:
                # Resume from the middle of an epoch.
                # NOTE: dataloader is not synced with previous state
                args.start_epoch = checkpoint['epoch']
                args.start_iter = checkpoint['step'] + 1
        del checkpoint
        torch.cuda.empty_cache()

    if args.load_detectron:  #TODO resume for detectron weights (load sgd momentum values)
        logging.info("loading Detectron weights %s", args.load_detectron)
        load_detectron_weight(maskRCNN, args.load_detectron)

    lr = optimizer.param_groups[0]['lr']  # lr of non-bias parameters, for commmand line outputs.

    maskRCNN = mynn.DataParallel(maskRCNN, cpu_keywords=['im_info', 'roidb'],
                                 minibatch=True)

    print(maskRCNN)
    ### Training Setups ###
    #args.run_name = misc_utils.get_run_name()
    #output_dir = misc_utils.get_output_dir(args, args.run_name)

    args.cfg_filename = os.path.basename(args.cfg_file)

    if not args.no_save:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        blob = {'cfg': yaml.dump(cfg), 'args': args}
        with open(os.path.join(output_dir, 'config_and_args.pkl'), 'wb') as f:
            pickle.dump(blob, f, pickle.HIGHEST_PROTOCOL)

        if args.use_tfboard:
            from tensorboardX import SummaryWriter
            # Set the Tensorboard logger
            tblogger = SummaryWriter(output_dir)

    ### Training Loop ###
    maskRCNN.train()

    training_stats = TrainingStats(
        args,
        args.disp_interval,
        tblogger if args.use_tfboard and not args.no_save else None)

    iters_per_epoch = int(train_size / args.batch_size)  # drop last
    args.iters_per_epoch = iters_per_epoch
    ckpt_interval_per_epoch = iters_per_epoch // args.ckpt_num_per_epoch
    if args.num_epochs==-1:
        number_epochs = (cfg.SOLVER.MAX_ITER // iters_per_epoch)
    else:
        number_epochs = args.num_epochs

    datasets_dictionary = create_dbs_for_classes(feature_db)
    median_distance_class = [np.inf] * cfg.MODEL.NUM_CLASSES


    print("Total number of epochs: ", number_epochs)

    try:
        logger.info('Training starts !')
        args.step = args.start_iter
        global_step = iters_per_epoch * args.start_epoch + args.step
        x, y = 0, 0
        for args.epoch in range(args.start_epoch, args.start_epoch + number_epochs):
            # ---- Start of epoch ----

            # adjust learning rate
            if args.lr_decay_epochs and args.epoch == args.lr_decay_epochs[0] and args.start_iter == 0:
                args.lr_decay_epochs.pop(0)
                net_utils.decay_learning_rate(optimizer, lr, cfg.SOLVER.GAMMA)
                lr *= cfg.SOLVER.GAMMA
            for args.step, input_data in zip(range(args.start_iter, iters_per_epoch), dataloader):
                x = x + 512
                for key in input_data:
                    if key != 'roidb': # roidb is a list of ndarrays with inconsistent length
                        input_data[key] = list(map(Variable, input_data[key]))
                training_stats.IterTic()
                input_data['only_bbox'] = [False]
                net_outputs = maskRCNN(**input_data)

                preidcted_classes = net_outputs["faiss_db"]["class"].detach().cpu().numpy()
                preidcted_classes_score = net_outputs["faiss_db"]["class_score"]
                roidb_batch = list(map(lambda x: blob_utils.deserialize(x)[0], input_data["roidb"][0]))
                print("Image" , [(os.path.basename(roi["image"]),  roi["dataset_idx"]) for roi in roidb_batch])
                print(len(preidcted_classes), [cl for cl in preidcted_classes if cl!=0], [roi["gt_classes"] for roi in roidb_batch])
                y += sum([gt_class for roi in roidb_batch for gt_class in roi["gt_classes"]if gt_class!=0])
                preidcted_bbox = net_outputs["faiss_db"]["bbox_pred"]
                preidcted_features = net_outputs["faiss_db"]["bbox_feat"].detach().cpu().numpy().astype(np.float32)
                foreground = net_outputs['faiss_db']["foreground"]
                if args.bbbp:
                    for roi in roidb_batch:
                        dataset_idx = roi["dataset_idx"]
                        if dataset_idx==1:
                            pass

                    #detect dataset
                    #detect if the predicted clas is not from the dataset
                    #detect probabability of being the object

                training_stats.UpdateIterStats(net_outputs)
                loss = net_outputs['total_loss']
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                training_stats.IterToc()
                print("Finishing training part")
                images = []

                print("Acc", (x - y) / x)

                if (args.step+1) % ckpt_interval_per_epoch == 0:
                    net_utils.save_ckpt(output_dir, args, maskRCNN, optimizer)

                if args.step % args.disp_interval == 0 and args.step!=0 :
                    log_training_stats(training_stats, global_step, lr)


                global_step += 1



            # ---- End of epoch ----
            # save checkpoint
            net_utils.save_ckpt(output_dir, args, maskRCNN, optimizer)
            # reset starting iter number after first epoch
            args.start_iter = 0
            if args.bbbp:
                dimension = 1024
                feature_db = update_db(args, dataloader_groundtruth, maskRCNN, images, image_to_idx, feature_db, output_dir)
                features = feature_db[:, 7:]
                features = features.astype('float32')
                faiss_db = faiss.IndexFlatL2(dimension)
                faiss_db.add(features)
                median_distance_class = find_threhold_for_each_class(faiss_db, features, feature_db[:, 2], k_neighbours=5)


        # ---- Training ends ----
        if iters_per_epoch % args.disp_interval != 0:
            # log last stats at the end
            log_training_stats(training_stats, global_step, lr)


    except (RuntimeError, KeyboardInterrupt):
        logger.info('Save ckpt on exception ...')
        net_utils.save_ckpt(output_dir, args, maskRCNN, optimizer)
        logger.info('Save ckpt done.')
        stack_trace = traceback.format_exc()
        print(stack_trace)

    finally:
        if args.use_tfboard and not args.no_save:
            tblogger.close()

def update_db(args, dataloader_groundtruth, maskRCNN, images, image_to_idx, feature_db, output_dir ):
    k = 0
    for val_data in zip(dataloader_groundtruth):
        output_path = os.path.join(output_dir, "feature_db_train" + str(args.step))
        print("Iteration", k)
        # mport ipdb; ipdb.set_trace()
        val_data = val_data[0]
        for key in val_data:
            if key != 'roidb':  # roidb is a list of ndarrays with inconsistent length
                val_data[key] = list(map(Variable, val_data[key]))
        roidb = list(map(lambda x: blob_utils.deserialize(x)[0], val_data["roidb"][0]))
        val_data["only_bbox"] = [True]
        val_data["image_to_idx"] = [image_to_idx]
        net_val_outputs = maskRCNN(**val_data)
        ground_truth_outputs = net_val_outputs['ground_truth']
        for i in ground_truth_outputs.keys():
            image_idx = ground_truth_outputs[i]["image"][0].item()
            if image_idx in images:
                continue
            images.append(image_idx)
            bboxes_gt = ground_truth_outputs[i]["bbox"].numpy()
            N_instances = len(bboxes_gt)
            features_gt = ground_truth_outputs[i]["features"].data.cpu().numpy().astype("float32")
            classes = ground_truth_outputs[i]["classes"].numpy()
            classes = [[cl] for cl in classes]
            db_idx = image_to_idx[image_idx]
            if not (feature_db[db_idx: db_idx + N_instances, 7:] == np.zeros((N_instances, 1024))).all():
                import ipdb;
                ipdb.set_trace()
            if not int(feature_db[db_idx, 0]) == image_idx:
                import ipdb;
                ipdb.set_trace()
            feature_db[db_idx: db_idx + N_instances, 2:] = np.concatenate([classes, bboxes_gt, features_gt], axis=1)
        k += len(ground_truth_outputs)
        if k % 500 == 0:
            print("Dumping it to pickle file ", output_path)
            np.save(output_path, feature_db)
    print("Dumping it to pickle file ", output_path)
    np.save(output_path, feature_db)
    with open(output_path + ".pkl", "wb") as f:
        pickle.dump(feature_db, f)
    print("Done working with database")
    return feature_db


def find_threhold_for_each_class(index, db, classes, k_neighbours=10):
    print("Doing search")
    distance, indecies = index.search(db, k_neighbours)
    print("Finishing search")
    classes_idx = classes[indecies]
    distance_class = {}
    counts = {}
    for idx, neighbours in enumerate(classes_idx):
        myself = int(classes[idx])
        not_class_neighbours = np.where(neighbours != myself)[0]
        if len(not_class_neighbours) == 0:
            not_class_neighbours = [k_neighbours - 1]
        first_not_class_neighbours = not_class_neighbours[0]
        # if first_not_class_neighbours==0:
        #    import ipdb; ipdb.set_trace()
        if myself not in counts.keys():
            counts[myself] = []
            distance_class[myself] = []
        counts[myself].append(first_not_class_neighbours)
        distance_class[myself].append(distance[idx, first_not_class_neighbours])

    median_distance_class = {}
    for class_idx in distance_class.keys():
        median_distance_class[class_idx] = np.median(distance_class[class_idx])
    return median_distance_class


def make_knn(index, db, classes, average_distance):
    distance, indecies = index.search(db, 1)
    neighbours = classes[indecies]
    drop_loss = []
    for neighbour, neighbour_distance in neighbours:
        if neighbour_distance < average_distance[neighbour]:
            drop_loss.append(True)
        else:
            drop_loss.append(False)
    return drop_loss


def create_dbs_for_classes(feature_db):
    set_datasets = set(feature_db[:,1])
    dimension = 1024
    datasets_dictionary = {}
    for dataset_id in set_datasets:
        indecies = np.where(feature_db[:,1]!=dataset_id)[0]
        features = feature_db[indecies, 7: ]
        features = features.astype('float32')
        faiss_db = faiss.IndexFlatL2(dimension)
        faiss_db.add(features)
        datasets_dictionary[dataset_id] = faiss_db
    return datasets_dictionary








def log_training_stats(training_stats, global_step, lr):
    stats = training_stats.GetStats(global_step, lr)
    log_stats(stats, training_stats.misc_args)
    if training_stats.tblogger:
        training_stats.tb_log_stats(stats, global_step)


if __name__ == '__main__':
    main()
