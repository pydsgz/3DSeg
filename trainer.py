import abc
import time
import os
import keras
import numpy as np
import pandas as pd
import logging
import multiprocessing as mp

import SimpleITK as sitk
import tensorflow as tf
from keras.utils import to_categorical
from keras.backend.tensorflow_backend import set_session

import utils
from model_provider import build_vnet_network
from volume_augment import VolumeDataGenerator
from vnet_utils import saveImages, maskLargestConnectedComponent, \
    computeQualityMeasures, resampleToTargetResolution, keep_largest_cc

# Set logging configuration.
logging.basicConfig(level=logging.INFO)

# Tensorflow weight initialization reproducible.
tf.set_random_seed(0)
logging.info(keras.backend.image_data_format())


class DLTrainer(abc.ABC):
    """ Base class for deep learning."""
    def __init__(self):
        pass

    @abc.abstractmethod
    def load_train_set(self):
        raise NotImplementedError

    @abc.abstractmethod
    def load_val_set(self):
        raise NotImplementedError

    @abc.abstractmethod
    def load_test_set(self):
        raise NotImplementedError

    @abc.abstractmethod
    def build_model(self):
        raise NotImplementedError

    @abc.abstractmethod
    def augment_procedure(self):
        raise NotImplementedError

    @abc.abstractmethod
    def train(self):
        raise NotImplementedError

    @abc.abstractmethod
    def test(self):
        raise NotImplementedError


class Segmentation3DTrainer(DLTrainer):
    """ Data loader and trainer for 3D segmentation using deep learning. """
    def __init__(self, args):
        super(Segmentation3DTrainer, self).__init__()
        self.args = args

        # All input volumes should be located here at base_dir
        self.base_dir = args.base_dir
        self.base_dir_t1_swi = args.base_dir_t1_swi

        # Parameters for dataloader
        self.COMPOSITE_TRANSFORM = args.composite_transform
        self.MULTI_CHANNEL = args.multi_channel
        self.VOLUME_SIZE = args.volume_dims
        self.TRAIN_TEST_SPLIT = args.train_split_r
        self.N_BATCHSIZE = args.batch_size
        self.DEBUG_SAVE_AUGMENTATIONS = args.debug_dump_image
        self.RANDOM_SEED = args.random_seed
        self.N_VOLQUEUE = args.volume_queue
        self.L_RATE = args.lr
        self.N_EPOCHS = args.epochs

        # Train and test index
        self.train_idx = np.arange(1, 21)
        self.test_idx = [9, 12, 13, 14, 17]
        self.train_idx = list(set(self.train_idx) - set(self.test_idx))

        # Region indices to include during training
        self.select_idx = list(np.arange(23))
        # Total number of regions to segment.
        self.n_class = len(self.select_idx)
        # Total number of segmentable regions.
        self.total_class = 23

        # Unique string to save model, output predictions, and augmented
        # volumnes.
        self.UNIQ_STR = args.uniq_str

        # When composite transform set to True, load non-resampled volumes of
        # mri-vols and seg-labels and perform resampling once at the end to have
        # input volumes have self.args.volume_dims dimensions.
        if self.COMPOSITE_TRANSFORM:
            logging.info('Will use composite transform')
            self.fp_mri = 'S%02d_SWI.nii.gz'
            self.fp_mri_t1 = 'S%02d_T1.nii.gz'
            self.fp_mri_swi = 'S%02d_SWI.nii.gz'
            self.fp_seg = 'S%02d_SEGjoint_from_gold_set1.nii.gz'
            self.fp_seg_multi_rater = 'S%02d_SEGjoint_from_gold_set2.nii.gz'
            self.fp_seg_single_rater = 'S%02d_SEGgold.nii.gz'
        else:
            self.fp_mri = 'S%02d_SWI_resampled_152iso.nii.gz'
            self.fp_seg = 'S%02d_SEGjoint_from_gold_set1_resampled_152iso.nii' \
                          '.gz'
            self.fp_seg_multi_rater = \
                'S%02d_SEGjoint_from_gold_set2_resampled_152iso' \
                '.nii.gz'
            self.fp_seg_single_rater = 'S%02d_SEGgold_resampled_152iso.nii.gz'

        # Create four folders to put all outputs [model, pred_output, logs,
        # augmented_vols]
        self._init_folders()

        # Load data splits.
        if args.train_model:
            self.train_set = self.load_train_set()
            self.val_set, self.train_set = self.load_val_set()
            # self.test_set = self.load_test_set()
        else:
            self.test_set = self.load_test_set()

    def init_folders(self):
        """
        Create four folders which will be used to save all outputs.

        Returns
        -------
        None
        """
        for i in ['model', 'pred_output', 'logs', 'augmented_vols']:
            if not os.path.exists(i):
                os.makedirs(i)

    def load_train_set(self):
        """
        Load volumes into memory to be used for on-the-fly data augmentation.

        Returns
        -------
        imgs: list of sitk.Images
            list of sitk images.
        """
        # Always use multi-rater train indices. Segmentations with *gold_set1*
        fp_seg = self.fp_seg

        # List of MRI and segmentation volume paths
        mri_vols_path = [self.fp_mri % (x) for x in self.train_idx]
        mri_vols_path = np.sort(
            [os.path.join(self.base_dir, x) for x in mri_vols_path])
        seg_vols_path = [fp_seg % (x) for x in self.train_idx]
        seg_vols_path = np.sort(
            [os.path.join(self.base_dir, x) for x in seg_vols_path])

        # Multi-modal, T1, SWI, and SWI
        if self.MULTI_CHANNEL:
            mri_vols_path_t1 = [self.fp_mri_t1 % (x) for x in self.train_idx]
            mri_vols_path_t1 = np.sort(
                [os.path.join(self.base_dir_t1_swi, x) for x in
                 mri_vols_path_t1])

            mri_vols_path_swi = [self.fp_mri_swi % (x) for x in self.train_idx]
            mri_vols_path_swi = np.sort(
                [os.path.join(self.base_dir_t1_swi, x) for x in
                 mri_vols_path_swi])

        # Read mri and segmentation volumes into memory
        imgs = []
        for k, v in enumerate(zip(mri_vols_path, seg_vols_path)):
            itkVol = sitk.ReadImage(v[0])
            itkSeg = sitk.ReadImage(v[1])
            if self.MULTI_CHANNEL:
                itkVol_t1 = sitk.ReadImage(mri_vols_path_t1[k])
                itkVol_swi = sitk.ReadImage(mri_vols_path_swi[k])
                # SWI, T1, SWI volumes
                itkVol = [itkVol, itkVol_t1, itkVol_swi]
            imgs.append((itkVol, itkSeg))
        return imgs

    def load_val_set(self):
        """
        Load validation set volumes but will resample to specified dimensions (
        args.volume_dims). Since no augmentation will be applied to these
        volumes, we need to resample it to desired dimension.

        Returns
        -------
        imgs: list of sitk.Images, list of sitk.Images
            Validation set, and train set excluding validation set.
        """
        train_split = int(self.TRAIN_TEST_SPLIT * len(self.train_set))
        val_set = self.train_set[train_split:]
        train_set = self.train_set[:train_split]
        mri_vols = [x[0] for x in val_set]
        seg_vols = [x[1] for x in val_set]

        # Read mri and segmentation volumes into memory
        imgs = []
        for k, v in enumerate(zip(mri_vols, seg_vols)):
            itkVol = v[0]
            itkSeg = v[1]

            # Resample to target resolution and spacing
            if isinstance(itkVol, list):
                new_itk_vol = []
                for vol_cur in itkVol:
                    size = self.VOLUME_SIZE[::-1]  # Dimension in z, y, x
                    origin_vol = vol_cur.GetOrigin()
                    origin_seg = itkSeg.GetOrigin()
                    spacing = [1.0, 1.0, 1.0]
                    direction = vol_cur.GetDirection()
                    transform = sitk.Transform(len(size), sitk.sitkIdentity)
                    vol_cur = sitk.Resample(vol_cur, size, transform,
                                            sitk.sitkLinear, origin_vol,
                                            spacing, direction)
                    new_itk_vol.append(vol_cur)
                itkVol = new_itk_vol
                itkSeg = sitk.Resample(itkSeg, size, transform,
                                       sitk.sitkNearestNeighbor, origin_seg,
                                       spacing, direction)
            else:
                size = self.VOLUME_SIZE[::-1]  # Dimension in z, y, x
                origin_vol = itkVol.GetOrigin()
                origin_seg = itkSeg.GetOrigin()
                spacing = [1.0, 1.0, 1.0]
                direction = itkVol.GetDirection()
                transform = sitk.Transform(len(size), sitk.sitkIdentity)
                itkVol = sitk.Resample(itkVol, size, transform,
                                       sitk.sitkLinear, origin_vol,
                                       spacing, direction)
                itkSeg = sitk.Resample(itkSeg, size, transform,
                                       sitk.sitkNearestNeighbor, origin_seg,
                                       spacing, direction)
            imgs.append((itkVol, itkSeg))
        return imgs, train_set

    def load_test_set(self):
        """
        Load test set volumes as sitk.Images.
        Returns
        -------
        imgs: list of sitk.Images
        """
        if self.MULTI_CHANNEL:
            self.test_idx.remove(12)  # T1 is missing in Subject 12
        test_mri_vols_path = [self.fp_mri % (x) for x in self.test_idx]
        test_mri_vols_path = np.sort([os.path.join(self.base_dir, x) for x in \
                                      test_mri_vols_path])
        test_seg_vols_path = [self.fp_seg % (x) for x in self.test_idx]
        test_seg_vols_path = np.sort([os.path.join(self.base_dir, x) for x in
                                      test_seg_vols_path])

        # Multi-modal, T1, SWI, and SWI
        if self.MULTI_CHANNEL:
            test_mri_vols_path_t1 = [self.fp_mri_t1 % (x) for x in
                                     self.test_idx]
            test_mri_vols_path_t1 = np.sort(
                [os.path.join(self.base_dir_t1_swi, x) for x in
                 test_mri_vols_path_t1])

            test_mri_vols_path_swi = [self.fp_mri_swi % (x) for x in
                                      self.test_idx]
            test_mri_vols_path_swi = np.sort(
                [os.path.join(self.base_dir_t1_swi, x) for x in
                 test_mri_vols_path_swi])

        # Multi-rater volumes
        test_seg_vols_path_multi_rater = [self.fp_seg_multi_rater % (x) for x in
                                          self.test_idx]
        test_seg_vols_path_multi_rater = np.sort(
            [os.path.join(self.base_dir, x) for x in
             test_seg_vols_path_multi_rater])

        # Single-rater volumes
        test_seg_vols_path_single_rater = [self.fp_seg_single_rater % (x) for
                                           x in self.test_idx]
        test_seg_vols_path_single_rater = np.sort([os.path.join(self.base_dir,
                                                                x) for x in
                                               test_seg_vols_path_single_rater])

        # Which ground truth test set to evaluate. 0 for multi-rater,
        # 1 for multi-atlas, and 2 for single-rater.
        if self.args.test_gt == 0:
            test_seg_vols_path = test_seg_vols_path_multi_rater
        elif self.args.test_gt == 1:
            test_seg_vols_path = test_seg_vols_path
        elif self.args.test_gt == 2:
            test_seg_vols_path = test_seg_vols_path_single_rater

        else:
            raise Exception('Which ground truth segmentation to use. Select 0 '
                            'for multi-rater, 1 for multi-atlas, '
                            '2 for single-rater.')

        # Read mri and segmentation volumes into memory
        imgs = []
        for k, v in enumerate(zip(test_mri_vols_path, test_seg_vols_path)):
            itkVol = sitk.ReadImage(v[0])
            itkSeg = sitk.ReadImage(v[1])

            # Resample data to desired size and spacing
            if self.MULTI_CHANNEL:
                itkVol_t1 = sitk.ReadImage(test_mri_vols_path_t1[k])
                itkVol_swi = sitk.ReadImage(test_mri_vols_path_swi[k])

                # Resample volumes
                vol_list = [itkVol, itkVol_t1, itkVol_swi]
                new_vol_list = []
                for vol in vol_list:
                    size = self.VOLUME_SIZE[::-1]  # Dimension in z, y, x
                    origin_vol = vol.GetOrigin()
                    spacing = [1.0, 1.0, 1.0]
                    direction = vol.GetDirection()
                    transform = sitk.Transform(len(size), sitk.sitkIdentity)
                    vol_cur = sitk.Resample(vol, size, transform,
                                            sitk.sitkLinear, origin_vol,
                                            spacing, direction)
                    new_vol_list.append(vol_cur)
                # Convert itk vols to numpy array
                np_vol = [utils.itk_to_numpy(x) for x in new_vol_list]
                np_vol = np.stack(np_vol, -1).astype(float)
            else:
                # Resample single volume
                size = self.VOLUME_SIZE[::-1]  # Dimension in z, y, x
                origin_vol = itkVol.GetOrigin()
                spacing = [1.0, 1.0, 1.0]
                direction = itkVol.GetDirection()
                transform = sitk.Transform(len(size), sitk.sitkIdentity)
                vol_cur = sitk.Resample(itkVol, size, transform,
                                        sitk.sitkLinear, origin_vol,
                                        spacing, direction)
                np_vol = utils.itk_to_numpy(vol_cur)

            # Resample segmentation
            origin_seg = itkSeg.GetOrigin()
            size = self.VOLUME_SIZE[::-1]
            spacing = [1.0, 1.0, 1.0]
            direction = itkSeg.GetDirection()
            transform = sitk.Transform(len(size), sitk.sitkIdentity)
            vol_seg = sitk.Resample(itkSeg, size, transform,
                                    sitk.sitkNearestNeighbor, origin_seg,
                                    spacing, direction)
            np_seg = utils.itk_to_np_segmentation(vol_seg, self.total_class,
                                                  self.select_idx)
            logging.info(v[1], np_seg.shape)
            imgs.append((np_vol, np_seg))
        return imgs

    def build_model(self):
        """
        Build a depth 3 VNET model with [16, 32, 64, 128] intermediate layers.
        Returns
        -------
        model: keras.Model
            Keras compiled model.
        """
        if self.MULTI_CHANNEL:
            vol_size = tuple(self.VOLUME_SIZE) + (3,)
        else:
            vol_size = tuple(self.VOLUME_SIZE) + (1,)
        model = build_vnet_network(inputs=keras.Input(shape=vol_size),
                                        depth=3, filters=[16, 32, 64, 128],
                                        dropoutAt=[None, None, None, None],
                                        multidice=False, n_class=self.n_class,
                                        group_normalize=True,
                                   l_rate=self.L_RATE)
        # model.load_weights('./model/model_vnet_NN_sigmoid_w_bg_run2'
        #                    '_152x152x152_23class_0.001.hdf5')
        return model

    def augment_procedure(self, q, random_seed):
        """
        On-the-fly augmentation procedure during training. Augmented volumes
        will be put to queue.
        Parameters
        ----------
        q: mp.Queue
        random_seed int

        Returns
        -------
        None
        """
        # Allows reproducible data augmentation.
        np.random.seed(random_seed)

        # Training set data.
        volumes = self.train_set
        vols_train = [(0, [vol, seg]) for vol, seg in volumes]

        # Setup augmentation framework.
        degree2euler = lambda d: d / 180.0 * np.pi

        vaug = VolumeDataGenerator(
            enableDeformation=True,
            enableRotation=True,
            enableTranslation=True,
            enableFlips=False,
            enableCropZoom=True,

            rotationAngleRanges=[degree2euler(15), degree2euler(15),
                                 degree2euler(15)],
            translationRanges=[25.0, 25.0, 25.0],
            FlipProb=[0.5, 0.5, 0.5],
            deformationSigma=(0.0, 0.05),
            cropZoomLevelRange=[(0.9, 1.1), (0.9, 1.1), (0.9, 1.1)],
            size=self.VOLUME_SIZE[::-1]  # Dimension should be in z, y, x
        )

        # Some parameters use to convert sitk.Image to numpy arrays.
        if self.MULTI_CHANNEL:
            vol_size = tuple(vaug.size[::-1]) + (3,)
        else:
            vol_size = tuple(vaug.size[::-1]) + (1,)
        seg_size = tuple(vaug.size[::-1]) + (self.n_class,)

        # Augmented volumes
        # [ (label, [image1,image2,image3,...]),...]
        vaug_gen = vaug.getNewImageFromBatch(vols_train,
                                 composite_transform=self.COMPOSITE_TRANSFORM)
        batch_img = np.empty(shape=(self.N_BATCHSIZE,) + vol_size, dtype=float)
        batch_seg = np.empty(shape=(self.N_BATCHSIZE,) + seg_size, dtype=bool)
        batch_index = 0

        # Save to augemented volumes to disk and/or put to multi-processing
        # queue.
        if self.DEBUG_SAVE_AUGMENTATIONS:
            img_ctr = 0
        for _, augm_vols, train_idx in vaug_gen:
            # Write data into batch array.
            if self.MULTI_CHANNEL:
                np_vol = [utils.itk_to_numpy(x) for x in augm_vols[0]]
                np_vol = np.stack(np_vol)
                image = np_vol.reshape(vol_size)
                batch_img[batch_index, :, :, :, :] = image
            else:
                image = utils.itk_to_numpy(augm_vols[0]).reshape(vol_size)
                batch_img[batch_index, :, :, :, :] = image

            # Seg label should not normalized globally
            seg_label = utils.itk_to_np_segmentation(augm_vols[1],
                                                     self.total_class,
                                                     self.select_idx)
            seg_label = seg_label.reshape(seg_size)

            # Save first 10 augmented volumes to disk.
            if self.DEBUG_SAVE_AUGMENTATIONS and img_ctr <= 10:
                # Save augmented volumes
                debug_vols_path = 'augmented_vols/%s' % self.UNIQ_STR
                if not os.path.exists(debug_vols_path):
                    os.mkdir(debug_vols_path)

                # Multiple modality input
                if isinstance(augm_vols[0], list):
                    for k, v in enumerate(augm_vols[0]):
                        sitk.WriteImage(v, "{}/SUBJ_{}_modal{}_dump_vol{}x{}x{"
                                           "}_debug_{}.nii.gz".format(
                            debug_vols_path,
                            train_idx,
                            k,
                            self.VOLUME_SIZE[0],
                            self.VOLUME_SIZE[1],
                            self.VOLUME_SIZE[2],
                            img_ctr))
                else:
                    sitk.WriteImage(augm_vols[0], "{}/SUBJ_{}_dump_vol{}x{}x{"
                                                  "}_debug_{}.nii.gz".format(
                        debug_vols_path,
                        train_idx,
                        self.VOLUME_SIZE[0],
                        self.VOLUME_SIZE[1],
                        self.VOLUME_SIZE[2],
                        img_ctr))
                sitk.WriteImage(augm_vols[1],
                                '{}/SUBJ_{}_dump_seg{}x{}x{}'
                                '_debug_{}.nii.gz'.format(debug_vols_path,
                                                          train_idx,
                                                          self.VOLUME_SIZE[0],
                                                          self.VOLUME_SIZE[1],
                                                          self.VOLUME_SIZE[2],
                                                          img_ctr))
                img_ctr += 1

            # Put volumes to queue.
            batch_seg[batch_index, :, :, :, :] = seg_label
            # update counter
            batch_index += 1
            if batch_index >= self.N_BATCHSIZE:
                # push batch to queue
                q.put((batch_img, batch_seg))

                # reset for new batch
                batch_img = np.empty(shape=(self.N_BATCHSIZE,) + vol_size,
                                     dtype=float)
                batch_seg = np.empty(shape=(self.N_BATCHSIZE,) + seg_size,
                                     dtype=bool)
                batch_index = 0

    def train(self, q, random_seed):
        """
        Model training procedure. The following will be saved:
            1. Keras callbacks will store tensorboard events at
            ./logs/
            2. Best model will be saved at ./model/
            3. Training will stop when validation loss does not improve after
            100 epochs.

        Parameters
        ----------
        q: mp.Queue
            Queue with augmented images from self.augment_procedure.
        random_seed: int
        Returns
        -------
        None
        """
        tf.set_random_seed(random_seed)
        np.random.seed(random_seed)
        start = time.time()

        # Wait till queue is filled
        while not q.full():
            logging.info(
                "Waiting for queue to reach {}; currently {}".format(
                    self.N_VOLQUEUE, q.qsize()))
            time.sleep(3)

        if self.MULTI_CHANNEL:
            vol_size = tuple(self.VOLUME_SIZE) + (3,)
        else:
            vol_size = tuple(self.VOLUME_SIZE) + (1,)
        seg_size = tuple(self.VOLUME_SIZE) + (self.n_class,)

        # Load validation dataset as numpy arrays.
        val_vols = self.val_set
        if self.MULTI_CHANNEL:
            # ([MRI vol modalities], vol_seg)
            cur_mri_arr = []
            for cur_vol, seg in val_vols:
                cur_vols_val = [utils.itk_to_numpy(modal) for modal in
                                cur_vol]
                cur_vols_val = np.stack(cur_vols_val, -1)
                cur_mri_arr.append(cur_vols_val)
            vols_val = np.array(cur_mri_arr).astype(float)
        else:
            vols_val = np.array(
                [utils.itk_to_numpy(vol).reshape(vol_size) for vol,
                                                               seg in
                 val_vols]).astype(float)

        segs_val = [utils.itk_to_np_segmentation(seg, self.total_class, \
                    self.select_idx).reshape(seg_size) for vol, seg in val_vols]
        segs_val = np.array(segs_val).astype(bool)

        # Keras VNET model
        model = self.build_model()

        # Queue generator for kera.fit_generator
        def keras_queue_generator_adapter():
            while True:
                vols, segm = q.get()
                yield vols, segm

        # Keras callbacks. Will log tensorboard event at ./logs/. Save
        # best model at ./model/. Stop training when validation loss does not
        # improve for 100 times.
        cb = keras.callbacks.TensorBoard(
            log_dir='./logs/model_%s_%sclass_%s' % (self.UNIQ_STR, self.n_class,
                                                    self.L_RATE),
            histogram_freq=2,
            batch_size=self.N_BATCHSIZE,
            write_graph=False, write_grads=True,
            write_images=False)
        early_stp_cb = keras.callbacks.EarlyStopping(monitor='val_loss',
                                                     min_delta=0, patience=100,
                                                     verbose=0, mode='auto')
        cpoint_cb = keras.callbacks.ModelCheckpoint(
            './model/model_%s_%sclass_%s.hdf5' % (self.UNIQ_STR, self.n_class,
                                                  self.L_RATE),
            monitor='val_loss',
            verbose=0,
            save_best_only=True,
            save_weights_only=True, mode='auto',
            period=1)

        # Train model
        logging.info('Training model...')
        model.fit_generator(generator=keras_queue_generator_adapter(),
                            steps_per_epoch=self.N_BATCHSIZE,
                            epochs=self.N_EPOCHS,
                            verbose=1,
                            validation_data=(vols_val, segs_val),
                            callbacks=[cb, early_stp_cb, cpoint_cb])
        end = time.time()
        logging.info('Train time in seconds: %s' % (end - start))

    def test(self):
        """
        Given a trained model at ./model/ evaluate test set.

        Returns
        -------
        None
        """
        # Load best model.
        model = self.build_model()
        model.load_weights('./model/model_%s_%sclass_%s.hdf5' %
                           (self.UNIQ_STR, self.n_class, self.L_RATE))

        # Load test set as numpy array.
        imgs = self.test_set
        if self.MULTI_CHANNEL:
            test_mri_vols_arr = [x[0] for x in imgs]
            test_mri_vols_arr = np.stack(test_mri_vols_arr)
        else:
            test_mri_vols_arr = [np.expand_dims(x[0], axis=-1) for x in imgs]
            test_mri_vols_arr = np.stack(test_mri_vols_arr)
        test_seg_vols_arr = [x[1] for x in imgs]
        test_seg_vols_arr = np.stack(test_seg_vols_arr)

        # Evaluate test set using model.
        start_prediction = time.time()
        test_seg_pred = model.predict(test_mri_vols_arr, batch_size=1)
        end = time.time()
        logging.info('Inference time in seconds: %s' % (end - start_prediction))

        # Filenames use to save output predictions as nifti file and
        # evaluation metrics as xlsx file.
        if self.args.test_gt == 0:
            test_pred_arr_fname = '%s_%sclass_test_metrics_multirater.xlsx' % \
                                  (self.UNIQ_STR, self.n_class)
            fp_pred_out = self.fp_seg_multi_rater.replace('.nii.gz',
                                                     'testpred.nii.gz')
        elif self.args.test_gt == 1:
            test_pred_arr_fname = '%s_%sclass_test_metrics_multiatlas.xlsx' % (
                self.UNIQ_STR, \
                self.n_class)
            fp_pred_out = self.fp_seg.replace('.nii.gz', 'testpred.nii.gz')
        elif self.args.test_gt == 2:
            test_pred_arr_fname = '%s_%sclass_test_metrics_singlerater.xlsx' % \
                                  (self.UNIQ_STR, self.n_class)
            fp_pred_out = self.fp_seg_single_rater.replace('.nii.gz',
                                                      'testpred.nii.gz')
        else:
            raise Exception('Please select which test ground truth to use. '
                            'Select 0, 1, and 2 only.')

        if self.n_class > 1:
            res_pred = np.argmax(test_seg_pred, -1)
        else:
            res_pred = np.zeros_like(test_seg_pred, dtype=np.uint8)
            res_pred[test_seg_pred > 0.5] = 1

        # Evaluate Dice score and Hausdorff distance of test set.
        n_test_set = test_seg_pred.shape[0]
        main_dice_list = []
        main_hdf_list = []
        one_h_argmax_ts_segpred = to_categorical(res_pred, self.n_class)
        test_seg_pred_clean = np.zeros_like(one_h_argmax_ts_segpred)
        for i_seg in range(n_test_set):
            cur_gt_seg = test_seg_vols_arr[i_seg]
            cur_pred_seg = one_h_argmax_ts_segpred[i_seg]
            overlap_measures_filter = sitk.LabelOverlapMeasuresImageFilter()
            cur_dice_list = []
            cur_hdf_list = []
            cur_pred_seg_clean = np.zeros_like(cur_pred_seg)
            for j_seg in range(self.n_class):

                # Current region binary label volume
                _seg_gt_j = cur_gt_seg[..., j_seg].astype(np.uint8)
                _seg_gt_j = sitk.GetImageFromArray(_seg_gt_j)

                # Take large component only
                _seg_pred_j = cur_pred_seg[..., j_seg].astype(np.uint8)
                _seg_pred_j_clean = keep_largest_cc(_seg_pred_j)
                _seg_pred_j_clean = sitk.GetImageFromArray(_seg_pred_j_clean)

                # Dice overlap measure
                overlap_measures_filter.Execute(_seg_gt_j, _seg_pred_j_clean)
                cur_dice = overlap_measures_filter.GetDiceCoefficient()
                cur_dice_list.append(cur_dice)

                # Hausdorff distance
                try:
                    hausdorff_distance_filter = \
                        sitk.HausdorffDistanceImageFilter()
                    hausdorff_distance_filter.Execute(_seg_gt_j,
                                                      _seg_pred_j_clean)
                    cur_hdf = hausdorff_distance_filter.GetHausdorffDistance()
                    cur_hdf_list.append(cur_hdf)
                    cur_pred_seg_clean[..., j_seg] = sitk.GetArrayFromImage(
                        _seg_pred_j_clean)
                except Exception:
                    cur_hdf_list.append(0.0)

            # Insert current cleaned volume
            test_seg_pred_clean[i_seg, ...] = cur_pred_seg_clean

            main_dice_list.append(pd.DataFrame(cur_dice_list))
            main_hdf_list.append(pd.DataFrame(cur_hdf_list))

        # Convert evaluation metrics as pandas dataframes.
        res_df_dice = pd.concat(main_dice_list, 1)
        res_df_hdf = pd.concat(main_hdf_list, 1)
        res_df_dice.columns = ['Dice_score_subj_%s' % str(x) for x in
                               self.test_idx]
        res_df_hdf.columns = ['Hausdorff_distance_subj_%s' % str(x) for x in
                              self.test_idx]

        # Calculate region-wise mean
        res_df_dice['mean_dice'] = res_df_dice.mean(1)
        res_df_hdf['mean'] = res_df_hdf.mean(1)

        # Combine dice and hausdorff dataframes
        res_df = pd.concat([res_df_dice, res_df_hdf], 1)

        # Row/region names inside dataframe
        # region_names = pd.read_csv('../output/region_names.csv')
        # res_df.index = list(region_names.names)[:self.n_class]

        # Save predicted segmentations and dataframe metrics in one location
        pred_seg_out_path = np.sort([fp_pred_out % (x) for x in self.test_idx])
        dum_counter = 0
        pred_out_base_dir = './pred_output/%s_%sclass/' % (self.UNIQ_STR,
                                                           str(self.n_class))
        if not os.path.exists(pred_out_base_dir):
            os.makedirs(pred_out_base_dir)
        test_pred_arr_fname = os.path.join(pred_out_base_dir,
                                           test_pred_arr_fname)

        # Save per volume dice scores and hausdorff distance
        res_df.to_excel(test_pred_arr_fname)

        test_mri_vols_path = [self.fp_mri % (x) for x in self.test_idx]
        test_mri_vols_path = np.sort([os.path.join(self.base_dir, x) for x in \
                                      test_mri_vols_path])
        test_seg_vols_path = [self.fp_seg % (x) for x in self.test_idx]
        test_seg_vols_path = np.sort([os.path.join(self.base_dir, x) for x in
                                      test_seg_vols_path])
        res_pred = np.argmax(test_seg_pred_clean, -1)

        for i in zip(test_mri_vols_path, pred_seg_out_path):
            logging.info(i)
            cur_img = sitk.ReadImage(i[0])
            cur_arr = res_pred[dum_counter, ...].astype(np.uint8)
            cur_arr = np.squeeze(cur_arr)
            imgLAB = sitk.GetImageFromArray(cur_arr)

            # Resample current meta information to get proper vols
            size = self.VOLUME_SIZE[::-1]  # Dimension in z, y, x
            origin_vol = cur_img.GetOrigin()
            spacing = [1.0, 1.0, 1.0]
            direction = cur_img.GetDirection()
            transform = sitk.Transform(len(size), sitk.sitkIdentity)
            vol_cur = sitk.Resample(cur_img, size, transform,
                                    sitk.sitkLinear, origin_vol,
                                    spacing, direction)
            imgLAB.CopyInformation(vol_cur)
            save_path = i[1].replace('.nii.gz', '%s_class.nii.gz' %
                                     self.n_class)
            save_path = os.path.join(pred_out_base_dir, save_path)

            # Saving predicted segmentation
            sitk.WriteImage(imgLAB, save_path)
            dum_counter += 1

        logging.info('======== Test set evaluation done. ======')
