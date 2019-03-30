# -*- coding: utf-8 -*-

# Copyright UCL Business plc 2017. Patent Pending. All rights reserved.
#
# The MonoDepth Software is licensed under the terms of the UCLB ACP-A licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.
#
# For any other use of the software not covered by the UCLB ACP-A Licence,
# please contact info@uclb.com

from __future__ import absolute_import, division, print_function

# only keep warnings and errors
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='1'
os.environ['CUDA_VISIBLE_DEVICES']='0'

import numpy as np
import argparse
import re
import time
import tensorflow as tf
import tensorflow.contrib.slim as slim

# 从同一个目录下的×××.py文件中导入相应的函数
from dataloader import *


# 命令行参数
parser = argparse.ArgumentParser(description='Monodepth TensorFlow implementation.')
# python monodepth_main.py --mode train 
# --model_name my_model 
# --data_path ../dataStereo/ 
# --filenames_file ../dataStereo/trainfilenames_stereo.txt 
# --log_directory ./tmp/
parser.add_argument('--mode',                      type=str,   help='train or test', default='train')
parser.add_argument('--dataset',                   type=str,   help='dataset to train on, kitti, or cityscapes', default='kitti')
parser.add_argument('--input_height',              type=int,   help='input height', default=256)
parser.add_argument('--input_width',               type=int,   help='input width', default=256)
parser.add_argument('--batch_size',                type=int,   help='batch size', default=2)
parser.add_argument('--num_epochs',                type=int,   help='number of epochs', default=50)
parser.add_argument('--learning_rate',             type=float, help='initial learning rate', default=1e-4)
parser.add_argument('--num_gpus',                  type=int,   help='number of GPUs to use for training', default=1)
parser.add_argument('--num_threads',               type=int,   help='number of threads to use for data loading', default=8)
parser.add_argument('--output_directory',          type=str,   help='output directory for test disparities, if empty outputs to checkpoint folder', default='')
parser.add_argument('--log_directory',             type=str,   help='directory to save checkpoints and summaries', default='')

parser.add_argument('--do_stereo',                             help='if set, will train the stereo model', action='store_true')


parser.add_argument('--data_path',                 type=str,   help='path to the data', required=True)
parser.add_argument('--filenames_file',            type=str,   help='path to the filenames text file', required=True)

args = parser.parse_args()

def post_process_disparity(disp):
    _, h, w = disp.shape
    l_disp = disp[0,:,:]
    r_disp = np.fliplr(disp[1,:,:])
    m_disp = 0.5 * (l_disp + r_disp)
    l, _ = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    l_mask = 1.0 - np.clip(20 * (l - 0.05), 0, 1)
    r_mask = np.fliplr(l_mask)
    return r_mask * l_disp + l_mask * r_disp + (1.0 - l_mask - r_mask) * m_disp

# 文件列表，返回行个数
def count_text_lines(file_path):
    f = open(file_path, 'r')
    lines = f.readlines()
    f.close()
    return len(lines)

def train(params):
    """Training loop."""

    # ??????
    with tf.Graph().as_default(), tf.device('/cpu:0'):

        # ??????
        global_step = tf.Variable(0, trainable=False)

        # OPTIMIZER
        # 导入训练文件的路径名，按照行进行读取；返回：文件中的行数，即双目训练集的数目
        num_training_samples = count_text_lines(args.filenames_file)
                
        steps_per_epoch = np.ceil(num_training_samples / params.batch_size).astype(np.int32)
        num_total_steps = params.num_epochs * steps_per_epoch
        start_learning_rate = args.learning_rate

        print("total number of samples: {}".format(num_training_samples))
        print("total number of epochs: {}".format(params.num_epochs))
        print("total number of num_total_steps: {}".format(num_total_steps))
        print("total number of batch_size: {}".format(params.batch_size))

        # --data_path ../dataStereo/ 
        # --filenames_file ../dataStereo/trainfilenames_stereo.txt
        # args.dataset  default='kitti'
        # 数据处理类：包含数据增强等操作，根据Epoch得到Tensorflow的数据
        dataloader = MonodepthDataloader(args.data_path, args.filenames_file, params, args.dataset, args.mode)
        left  = dataloader.left_image_batch
        right = dataloader.right_image_batch

        # SESSION
        config = tf.ConfigProto(allow_soft_placement=True)
        sess = tf.Session(config=config)

        # INIT
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        coordinator = tf.train.Coordinator()
        # 创建样例队列！！！
        threads = tf.train.start_queue_runners(sess=sess, coord=coordinator)

        # GO!
        start_step = global_step.eval(session=sess)
        start_time = time.time()
        for step in range(start_step, num_total_steps):
            print('step: ', step, '; shape: ', sess.run(tf.shape(left)))
            left_, right_ = sess.run([left, right])
            print(left_)


# def test(params):
#     """Test function."""

#     dataloader = MonodepthDataloader(args.data_path, args.filenames_file, params, args.dataset, args.mode)
#     left  = dataloader.left_image_batch
#     right = dataloader.right_image_batch

#     model = MonodepthModel(params, args.mode, left, right)

#     # SESSION
#     config = tf.ConfigProto(allow_soft_placement=True)
#     sess = tf.Session(config=config)

#     # SAVER
#     train_saver = tf.train.Saver()

#     # INIT
#     sess.run(tf.global_variables_initializer())
#     sess.run(tf.local_variables_initializer())
#     coordinator = tf.train.Coordinator()
#     threads = tf.train.start_queue_runners(sess=sess, coord=coordinator)

#     # RESTORE
#     if args.checkpoint_path == '':
#         restore_path = tf.train.latest_checkpoint(args.log_directory + '/' + args.model_name)
#     else:
#         restore_path = args.checkpoint_path.split(".")[0]
#     train_saver.restore(sess, restore_path)

#     num_test_samples = count_text_lines(args.filenames_file)

#     print('now testing {} files'.format(num_test_samples))
#     disparities    = np.zeros((num_test_samples, params.height, params.width), dtype=np.float32)
#     disparities_pp = np.zeros((num_test_samples, params.height, params.width), dtype=np.float32)
#     for step in range(num_test_samples):
#         disp = sess.run(model.disp_left_est[0])
#         disparities[step] = disp[0].squeeze()
#         disparities_pp[step] = post_process_disparity(disp.squeeze())

#     print('done.')

#     print('writing disparities.')
#     if args.output_directory == '':
#         output_directory = os.path.dirname(args.checkpoint_path)
#     else:
#         output_directory = args.output_directory
#     np.save(output_directory + '/disparities.npy',    disparities)
#     np.save(output_directory + '/disparities_pp.npy', disparities_pp)

#     print('done.')

def main(_):

    # monodepth参数信息，通过namedtuple()来定义
    params = dataloader_parameters(
        height=args.input_height,
        width=args.input_width,
        batch_size=args.batch_size,
        num_threads=args.num_threads,
        num_epochs=args.num_epochs,
        do_stereo=args.do_stereo)

    # print(params)
    
    if args.mode == 'train':
        train(params)
    elif args.mode == 'test':
        test(params)

if __name__ == '__main__':
    tf.app.run()
