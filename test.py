from __future__ import  absolute_import
import os

import ipdb
import argparse
from data.util import  read_image
import matplotlib
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

from utils.config import opt
from data.dataset import Dataset, TestDataset, inverse_normalize
from model import FasterRCNNVGG16, FasterRCNNResnet101
from torch.utils import data as data_
from trainer import FasterRCNNTrainer
from utils import array_tool as at
from utils.vis_tool import visdom_bbox
from utils.eval_tool import eval_detection_voc

# fix for ulimit
# https://github.com/pytorch/pytorch/issues/973#issuecomment-346405667
import resource

rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (20480, rlimit[1]))

matplotlib.use('agg')

def eval(dataloader, faster_rcnn, test_num=10000):
    pred_bboxes, pred_labels, pred_scores = list(), list(), list()
    gt_bboxes, gt_labels, gt_difficults = list(), list(), list()
    for ii, (imgs, sizes, gt_bboxes_, gt_labels_, gt_difficults_) in tqdm(enumerate(dataloader)):
        sizes = [sizes[0][0].item(), sizes[1][0].item()]
        pred_bboxes_, pred_labels_, pred_scores_ = faster_rcnn.predict(imgs, [sizes])
        gt_bboxes += list(gt_bboxes_.numpy())
        gt_labels += list(gt_labels_.numpy())
        gt_difficults += list(gt_difficults_.numpy())
        pred_bboxes += pred_bboxes_
        pred_labels += pred_labels_
        pred_scores += pred_scores_
        if ii == test_num: break

    result = eval_detection_voc(
        pred_bboxes, pred_labels, pred_scores,
        gt_bboxes, gt_labels, gt_difficults,
        use_07_metric=True)
    return result

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='lanta10_fasterrcnn')
    parser.add_argument('--pretrained_model', type=str, default='vgg16')
    parser.add_argument('--voc_data_dir', type=list, default=['Dataset/VOCdevkit2007/VOC2007/'])
    parser.add_argument('--plot_every', type=int, default=100)
    parser.add_argument('--caffe_pretrain', type=bool, default=True)
    parser.add_argument('--caffe_pretrain_path', type=str, default='pretrained/vgg16_caffe.pth')
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--lanta', type=float, default=1)
    parser.add_argument('--use_adam', type=bool, default=False)
    parser.add_argument('--use_drop', type=bool, default=False)
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    opt._parse(vars(args))
    testset = TestDataset(opt)
    if opt.pretrained_model == 'vgg16':
        faster_rcnn = FasterRCNNVGG16()
    elif opt.pretrained_model == 'resnet101':
        faster_rcnn = FasterRCNNResnet101()
    trainer = FasterRCNNTrainer(faster_rcnn).cuda()

    trainer.load('save/fasterrcnn_09071339_0.6984935332483329')

    test_dataloader = data_.DataLoader(testset,
                                       batch_size=1,
                                       num_workers=opt.test_num_workers,
                                       shuffle=False, \
                                       pin_memory=True
                                       )

    eval_result = eval(test_dataloader, faster_rcnn, test_num=opt.test_num)
    log_info = 'map:{}'.format(str(eval_result['map']))
    print(log_info)