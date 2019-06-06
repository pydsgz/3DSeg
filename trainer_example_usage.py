import argparse
import logging
import multiprocessing as mp

import keras
import tensorflow as tf

import trainer


def train(args):
    logging.info(keras.backend.image_data_format())

    # Change unique string to prepend to all outputs.
    vol_size = args.volume_dims

    # Segmentation trainer which loads data and handles augmentation and
    # training as well as testing.
    seg_trainer = trainer.Segmentation3DTrainer(args)

    # Train model using traning set and validation set.
    if args.train_model:

        # Multi-processing procedure
        mp.get_context('spawn')
        q = mp.Queue(args.volume_queue)
        p_augment1 = mp.Process(target=seg_trainer.augment_procedure,
                                args=(q, 1))
        p_augment2 = mp.Process(target=seg_trainer.augment_procedure,
                                args=(q, 100))
        p_dl = mp.Process(target=seg_trainer.train, args=(q, 999))
        p_augment1.start()
        p_augment2.start()
        p_dl.start()
        p_dl.join()
        p_augment1.terminate()
        p_augment2.terminate()
        p_augment1.join()
        p_augment2.join()

    # Evaluate model on test set
    else:
        seg_trainer.test()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Parse arguments
    parser = argparse.ArgumentParser(description='3-D Segementation Applet.')
    parser.add_argument('--train_model', type=int, default=0,
                        help='Set 1 to train the model 0 to test a given model')
    parser.add_argument('--composite_transform', type=int, default=1,
                        help='Set 1 to use composite transforms in '
                             'augmentation pipeline')
    parser.add_argument('--multi_channel', type=int, default=0,
                        help='Set 1 to use multi-channel input (T1, T2, SWI), '
                             'otherwise will use a single channel input.')
    parser.add_argument('--train_split_r', type=float, default=.9,
                        help='Ratio of training set to use.')
    parser.add_argument('--test_gt', type=int, default=0,
                        help='Set which segmentation volumes will be used to '
                             'during testing. Set 0 for multi-rater, '
                             'set 1 for multi-atlas, and 2 for single-rater.')
    parser.add_argument('--volume_dims', type=list, default=[128, 128, 128],
                        help='Dimensions of training volume. Resampling to '
                             'this dimension will be done during augmentation.')
    parser.add_argument('--epochs', type=int, default=10000,
                        help='Number of training epochs.')
    parser.add_argument('--volume_queue', type=int, default=2,
                        help='Number of volumes to put in multiprocessing '
                             'queue during on-the-fly augmentation.')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size during model training.')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Leanring rate to use.')

    parser.add_argument('--debug_dump_image', type=int, default=1,
                        help='Set to 1 to save first 10 augmented volumes to '
                             'disk (./imgs_composite_transform/).')
    parser.add_argument('--random_seed', type=int, default=1,
                        help='Random seed number to use for Tensorflow and '
                             'Numpy.')
    parser.add_argument('--base_dir', type=str,
                        default='./data/',
                        help='Location of all volumes and its '
                             'segmentations.')
    parser.add_argument('--base_dir_t1_swi', type=str,
                    default='./data_multi/',
                        help='Location of all T1 and SWI MRI volumes and its '
                             'segmentations.')
    parser.add_argument('--uniq_str', type=str,
                        default='vnet',
                        help='Unique string to prepend to prediction volumes,'
                             'model output and tensorboard events.')
    parser.add_argument('--fp_vols', type=str,
                        default='Case%02d.mhd',
                        help='File pattern of volumes.')
    parser.add_argument('--fp_segs', type=str,
                        default='Case%02d_segmentation.mhd',
                        help='File pattern of segmentation volumes.')

    args = parser.parse_args()

    # Entry-point to train or test model.
    train(args)
