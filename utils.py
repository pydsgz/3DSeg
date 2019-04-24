import os
import sys
from os.path import join as osj
import numpy as np
import pandas as pd
from time import time
from shutil import copyfile
import subprocess
import re

from joblib import Parallel, delayed
import SimpleITK as sitk
from nipype.interfaces import ants
from nipype.interfaces import c3
from nipype.interfaces import fsl
import nilearn.image as ni
from keras.utils import to_categorical


###
### Helper functions to preprocess MRI volumes and segmentation maps
###
def reg_ants(ffFixed, ffMoving, output_prefix='output_',
             transforms=['Affine', 'SyN'],
             transform_parameters=[(2.0,), (0.25, 3.0, 0.0)],
             number_of_iterations=[[1500, 200], [30, 50, 20]],
             metric=['Mattes', 'CC'],
             metric_weight=[1, 1],
             num_threads=4,
             radius_or_number_of_bins=[32, 2],
             sampling_strategy=['Random', None],
             sampling_percentage=[0.05, None],
             convergence_threshold=[1.e-8, 1.e-9],
             convergence_window_size=[20, 20],
             smoothing_sigmas=[[2, 1], [3, 2, 1]],
             sigma_units=['vox', 'vox'],
             shrink_factors=[[3, 2], [4, 3, 2]],
             use_histogram_matching=[True, True],
             output_warped_image='output_warped_image.nii.gz'):
    """
    Sequential affine and deformable registration using ANTs.

    Parameters
    ----------
    ffFixed: str
    ffMoving: str
    output_prefix: str
    transforms: list
    transform_parameters: list
    number_of_iterations: list
    metric: list
    metric_weight: list
    num_threads: int
    radius_or_number_of_bins:list
    sampling_strategy: list
    sampling_percentage: list
    convergence_threshold: list
    convergence_window_size: list
    smoothing_sigmas: list
    sigma_units: list
    shrink_factors: list
    use_histogram_matching: list
    output_warped_image: str

    Returns
    -------
    reg: ants.Registration

    """
    reg = ants.Registration()
    reg.inputs.fixed_image = ffFixed
    reg.inputs.moving_image = ffMoving
    reg.inputs.output_transform_prefix = output_prefix
    # reg.inputs.initial_moving_transform = 'trans.mat'
    # reg.inputs.invert_initial_moving_transform = True
    reg.inputs.transforms = transforms
    reg.inputs.transform_parameters = transform_parameters
    reg.inputs.number_of_iterations = number_of_iterations
    reg.inputs.dimension = 3
    reg.inputs.write_composite_transform = False
    reg.inputs.collapse_output_transforms = False
    reg.inputs.metric = metric
    # Default (value ignored currently by ANTs)
    reg.inputs.metric_weight = metric_weight
    reg.inputs.num_threads = num_threads
    reg.inputs.radius_or_number_of_bins = radius_or_number_of_bins
    reg.inputs.sampling_strategy = sampling_strategy
    reg.inputs.sampling_percentage = sampling_percentage
    reg.inputs.convergence_threshold = convergence_threshold
    reg.inputs.convergence_window_size = convergence_window_size
    reg.inputs.smoothing_sigmas = smoothing_sigmas
    reg.inputs.sigma_units = sigma_units
    reg.inputs.shrink_factors = shrink_factors
    reg.inputs.use_histogram_matching = use_histogram_matching
    reg.inputs.output_warped_image = output_warped_image
    print(reg.cmdline)
    return reg


def reg_ants_at_speed(ffFixed, ffMoving, ffAFF, ffDEF, ffOUT, ffAFFInv=None,
                      ffDEFInv=None, speed='normal'):
    """
    Sequential affine and deformable registration using ANTs.
    Convenience wrapper to reg_ants that sets registration parameters to run
    at a certain "speed".

    Parameters
    ----------
    ffFixed: str
    ffMoving: str
    ffAFF: str
    ffDEF: str
    ffOUT: str
    ffAFFInv: str
    ffDEFInv: str
    speed: str
        Pick one from this list ["normal", "fast_inaccurate", "slow_accurate",
        "debug"]. Debug is just for debug purpose.

    Returns
    -------
    reg: nipype antsRegistration object
        Files will also be written to disk at specified target location.
        Output at ffOUT.
    """
    output_prefix = ffFixed.replace('.nii.gz', '_trf_')

    list_of_speed = ['normal', 'fast_inaccurate', 'slow_accurate', 'debug']
    if speed == 'normal':
        reg = reg_ants(ffFixed, ffMoving, output_prefix=output_prefix,
                       output_warped_image=output_prefix + 'deformed.nii.gz',
                       number_of_iterations=[[2000, 1000], [30, 50, 20]])
    elif speed == 'fast_inaccurate':
        reg = reg_ants(ffFixed, ffMoving, output_prefix=output_prefix,
                       output_warped_image=output_prefix + 'deformed.nii.gz',
                       number_of_iterations=[[50, 10], [10, 5, 2]])
    elif speed == 'slow_accurate':
        reg = reg_ants(ffFixed, ffMoving, output_prefix=output_prefix,
                       output_warped_image=output_prefix + 'deformed.nii.gz',
                       number_of_iterations=[[2000, 1000], [70, 100, 50]])
    elif speed == 'debug':
        reg = reg_ants(ffFixed, ffMoving, output_prefix=output_prefix,
                       output_warped_image=output_prefix + 'deformed.nii.gz',
                       number_of_iterations=[[1, 1], [1, 0, 0]])
    else:
        raise NotImplementedError
    reg.run()

    # rename file output_prefix0Affine.mat to ffAFF
    # etc
    os.rename('%s0Affine.mat' % output_prefix, ffAFF)
    os.rename('%s1Warp.nii.gz' % output_prefix, ffDEF)
    if ffDEFInv is not None:
        os.rename('%s1InverseWarp.nii.gz' % output_prefix, ffDEFInv)
    # create inverse affine transform
    if ffAFFInv is not None:
        sitk_inverse_affine(ffAFF, ffAFFInv)
        # rename the deformed output volume
    os.rename(output_prefix + 'deformed.nii.gz', ffOUT)
    return reg


def ants_apply_transform(ffMOV, ffREF, ffOUT, transformations,
                         interpolation='Linear', run=False):
    """
    Apply transformations to a volume using given transformations.

    Parameters
    ----------
    ffMOV: str
    ffREF: str
    ffOUT: str
    transformations: list of str
        full file paths to all transformation that should be applied
        Note: transformations are applied from right to left (like matrix
        multiplication)

    interpolation: str
        'Linear' or 'NearestNeighbor' or 'CosineWindowedSinc'
             or 'WelchWindowedSinc' or 'HammingWindowedSinc' or
             'LanczosWindowedSinc' or 'MultiLabel' or 'Gaussian' or 'BSpline',
             nipype default value: Linear
    run: bool,
        If True, files will be written to disk at specified target location.

    Returns
    -------
    at: ants
        output at ffOUT

    """
    at = ants.ApplyTransforms()  # FILTER / nipype module
    at.inputs.dimension = 3
    at.inputs.input_image = ffMOV
    at.inputs.reference_image = ffREF
    at.inputs.output_image = ffOUT
    at.inputs.interpolation = 'MultiLabel'
    # at.inputs.interpolation_parameters = (5,)
    # at.inputs.default_value = 0
    at.inputs.num_threads = 4
    at.inputs.transforms = transformations
    # at.inputs.invert_transform_flags = [False]
    if run:
        at.run()
    return at


def ants_n4_bf_correction(ffVOL, ffOUT, run=False,
                          bspline_fitting_distance=300,
                          shrink_factor=3,
                          n_iterations=[50, 50, 30, 20]):
    """
    Run N4 bias field correction on given MRI volume.
    Parameters
    ----------
    ffVOL: str
    ffOUT: str
    run: bool
    bspline_fitting_distance: int
    shrink_factor: int
    n_iterations: list

    Returns
    -------
    n4: nipype.interfaces.ants.N4BiasFieldCorrection
        Will write output to desire ffout path.
    """
    n4 = ants.N4BiasFieldCorrection() # FILTER/nipype module, here:wrapping ants
    n4.inputs.dimension = 3
    n4.inputs.bspline_fitting_distance = bspline_fitting_distance
    n4.inputs.shrink_factor = shrink_factor
    n4.inputs.n_iterations = n_iterations
    n4.inputs.num_threads = 4
    # N4 Bias Field correction for all volumes
    n4.inputs.input_image = ffVOL
    n4.inputs.output_image = ffOUT
    if run:
        n4.run()
    return n4


def robust_cropping_fsl(ffVOL, ffOUT, run=False):
    """
    Robust cropping using FSL to remove lower head and neck.


    Parameters
    ----------
    ffVOL: str
    ffOUT: str
    run: bool
        if True, will write output to desired ffout path.

    Returns
    -------
    fov: fsl.RobustFOV
    """
    # robustfov -i SUBJ01_FS30_T2_MRI.nii.gz -r
    # SUBJ01_FS30_T2_MRI_cropped.nii.gz
    fov = fsl.RobustFOV()
    fov.inputs.in_file = ffVOL
    fov.inputs.out_roi = ffOUT
    fov.inputs.output_type = 'NIFTI_GZ'
    if run:
        fov.run()
    return fov


def nibet(ffVOL, ffOUT, run=False):
    """
    Skull stripping using FSL.


    Parameters
    ----------
    ffVOL: str
    ffOUT: str
    run: bool
        if True, will write output to desired ffout path.
    Returns
    -------
    btr: fsl.BET
    """
    btr = fsl.BET()
    btr.inputs.in_file = ffVOL
    btr.inputs.out_file = ffOUT
    #  Overall segmented brain will become larger (<0.5) or smaller (>0.5)
    btr.inputs.frac = 0.25
    if run:
        btr.run()
    return btr


def crop_blackspace(ffVOL, ffOUT):
    """
    Crop blackspace in MRI volume.
    Will retain 1 voxel grid of zeros before the nonzero part. Will write
    output at ffOUT.


    Parameters
    ----------
    ffVOL: str
    ffOUT: str

    Returns
    -------
    None
    """
    img = ni.load_img(ffVOL)
    img = ni.crop_img(img, copy=True)
    img.to_filename(ffOUT)


def resample_to_target_resolution(ffVOL, ffOUT, targetResolution,
                                  interpolation='Linear', run=False):
    """
    Resample MRI volume to a target resolution. Will write output at ffOUT.


    Parameters
    ----------
    ffVOL: str
    ffOUT: str
    targetResolution
    interpolation: str
        Pick one from this list ["NearestNeighbor", "Linear", "Cubic", "Sinc",
        "Gaussian"]
    run: bool
        if True, will write output to desired ffout path.

    Returns
    -------
    c3d: c3.C3d
    Notes
    -----
    c3d documentation at:
    (https://sourceforge.net/p/c3d/git/ci/master/tree/doc/c3d.md)

    """
    if isinstance(targetResolution, float):
        targetResolution = [targetResolution] * 3
    c3d = c3.C3d()
    c3d.inputs.in_file = ffVOL
    c3d.inputs.out_file = ffOUT
    c3d.inputs.args = '-resample-mm %sx%sx%smm -interpolation %s' % (
    str(targetResolution[0]),
    str(targetResolution[1]),
    str(targetResolution[2]),
    interpolation)
    # print(c3.cmdline)
    if run:
        c3d.run()  # runnning the resmapling on the command line via nipype
    return c3d


def resample_to_reference_volume(ffREF, ffSRC, ffOUT, interpolation='Linear',
                                 save=True):
    """
    Resample volume to given reference volume.


    Parameters
    ----------
    ffREF: str
    ffSRC: str
    ffOUT: str
    interpolation
        Can be <NearestNeighbor|Linear|Cubic|Sinc|Gaussian|MultiLabel|Lanczos>
    save: bool,
        If True, will write output at ffOUT.
    Returns
    -------
        Will write output to desired ffout path.

    """
    if interpolation == 'Linear':
        inter = sitk.sitkLinear
    elif interpolation == 'Cubic':
        inter = sitk.sitkBSpline
    elif interpolation == 'NearestNeighbor':
        inter = sitk.sitkNearestNeighbor
    elif interpolation == 'Cubic':
        inter = sitk.sitkBSpline
    elif interpolation == 'Sinc':
        inter = sitk.sitkHammingWindowedSinc
    elif interpolation == 'MultiLabel':
        inter = sitk.sitkLabelGaussian
    elif interpolation == 'Lanczos':
        inter = sitk.sitkLanczosWindowedSinc
    else:
        raise NotImplementedError

    imgREF = sitk.ReadImage(ffREF)
    imgSRC = sitk.ReadImage(ffSRC)

    res = sitk.ResampleImageFilter()
    res.SetReferenceImage(imgREF)
    res.SetInterpolator(inter)
    imgOUT = res.Execute(imgSRC)
    if save:
        sitk.WriteImage(imgOUT, ffOUT)
    else:
        return imgOUT


def sitk_inverse_affine(ffAFF, ffOUT):
    """
    Given affine transform apply its inverse transform. Will write output at
    ffOUT.


    Parameters
    ----------
    ffAFF: str
    ffOUT: str

    Returns
    -------
    None
    """
    trf = sitk.ReadTransform(ffAFF)
    trfinv = trf.GetInverse()
    trfinv.WriteTransform(ffOUT)


def ants_register(idx_fix, idx_mov, pnBase, fptnVOL, speed='normal'):
    """
    Call to ANTs deformable registration using reg_ants_at_speed function.
    Will write output at ffOUT.

    Parameters
    ----------
    idx_fix: int
    idx_mov: int
    pnBase: str
    fptnVOL: str
    speed: str

    Returns
    -------
    None
    """
    # register!
    ffFIX = osj(pnBase, fptnVOL % idx_fix)
    ffMOV = osj(pnBase, fptnVOL % idx_mov)
    ffAFF = osj(pnBase, 'T_%02d_to_%02d_AFF.mat' % (idx_mov, idx_fix))
    ffDEF = osj(pnBase, 'T_%02d_to_%02d_DEF.nii.gz' % (idx_mov, idx_fix))
    ffOUT = osj(pnBase, 'T_%02d_to_%02d_deformed.nii.gz' % (idx_mov, idx_fix))
    # ffAFFInv = osj(pnBase,'T_%02d_to_%02d_AFF.mat'%(idx_fix,idx_mov))
    # ffDEFInv = osj(pnBase,'T_%02d_to_%02d_DEF.nii.gz'%(idx_fix,idx_mov))
    # if not os.path.exists(ffOUT):
    time0 = time()
    reg_ants_at_speed(ffFIX, ffMOV, ffAFF, ffDEF, ffOUT, speed=speed)
    # ffOUTInv = osj(pnBase,'T_%02d_to_%02d_deformed.nii.gz'%(idx_fix,idx_mov))
    # ants_apply_transform(ffFIX,ffMOV,ffOUTInv,[ffAFFInv,ffDEFInv],
    # interpolation='Linear',run=True)
    print('\n\n\nTransformation done: %d to %d (Elapsed time: %0.2f)\n\n\n' % \
          (idx_mov, idx_fix, time() - time0))


def create_silver_seg(idx_fix, idx_mov, pnBase, fptnVOL, fptnSEG):
    """
    Given gold standard segmentation create a silver
    standard segmentation by transforming the segmentation
    volume to the target volume. Will write output at ffOUT.


    Parameters
    ----------
    idx_fix: int
    idx_mov: int
    pnBase: str
    fptnVOL: str
    fptnSEG: str

    Returns
    -------
    None
    """
    # File names to use
    ffFIX = osj(pnBase, fptnVOL % idx_fix)
    ffMOV = osj(pnBase, fptnSEG % idx_mov)
    ffAFF = osj(pnBase, 'T_%02d_to_%02d_AFF.mat' % (idx_mov, idx_fix))
    ffDEF = osj(pnBase, 'T_%02d_to_%02d_DEF.nii.gz' % (idx_mov, idx_fix))
    ffOUT = osj(pnBase, 'S%02d_on_%02d_SEG.nii.gz' % (idx_mov, idx_fix))

    # Create silver standard segmentations using
    # transformed gold standard segmentation
    if idx_mov == idx_fix:
        # copy segmentation with ffOUT fname
        copyfile(ffMOV, ffOUT)
    else:
        # Apply correct transformations to current segmentation
        ants_apply_transform(ffMOV, ffFIX, ffOUT, [ffDEF, ffAFF], \
                             interpolation='Linear', run=True)


def parallel_create_silver_seg(gold_idx, idx_list, pnBase, fptnVOL, fptnSEG,
                               n_jobs=10):
    """
    Create silver standard segmentations given gold standard indices using
    multi-processing.


    Parameters
    ----------
    gold_idx: list
    idx_list: list
    pnBase: str
    fptnVOL: str
    fptnSEG: str
    n_jobs: int

    Returns
    -------
    None
    """
    # For each gold standard segmentations
    # Transform to target volume
    for k, idx_mov in enumerate(gold_idx):
        Parallel(n_jobs=n_jobs)(
            delayed(create_silver_seg)(idx_fix, idx_mov, pnBase, fptnVOL,
                                       fptnSEG) \
            for idx_fix in idx_list)
        print('%s done out of %s' % (k + 1, len(gold_idx)))


def fuse_labels(idx_gold, idx_list, pn_base, fp_seg, ff_out):
    """
    Fuse labels using ImageMath script.
    This is a simple wrapper to the
    ImageMath command line call.


    Parameters
    ----------
    idx_gold: list
    idx_list: list
    pn_base: str
    fp_seg: str
    ff_out: str

    Returns
    -------
    Will return 0 if successful. Return 123 and print indices which failed.
    """
    # Fuse labels using gold standard raters
    failed_list = []
    len_idx = len(idx_list)
    counter = 0
    for k, cur_idx in enumerate(idx_list):
        cur_list = [osj(pn_base, fp_seg % (x, cur_idx)) for x in idx_gold]
        ffOUT = osj(pn_base, ff_out % (cur_idx))
        res_list = ['ImageMath', '3', ffOUT, 'MajorityVoting'] + cur_list
        print(res_list)
        print('\n')
        res_ = subprocess.run(res_list)
        if res_.returncode == 0 and os.path.exists(ffOUT):
            print('== %s OK ==' % (ffOUT))
            counter += 1
        else:
            failed_list.append(ffOUT)
        print('Running %02d of %s' % (k + 1, len_idx))
    if counter == len_idx:
        print('== Label fusion finished ==')
        return 0
    else:
        print('== Label fusion did not finish properly ==\n Still missing: \n')
        print(failed_list)
        return 123


def preprocess_data(idx, goldIndices, silverIndices, pnSRC, pnBase,
                    fnSRCPatternVOLgold, fnSRCPatternVOLsilver, \
                    fnSRCPatternSEG, fptnVOL, fptnSEG):
    """
    Preprocess all MRI volumes and segmentations. Will output files at pnBase.
    1. Preprocess will perform.
        1.1 ants_n4_bf_correction - bias field correction
        1.2 robust_cropping_fsl - neck removal
        1.3 nibet - brain extraction
        1.4 crop_blackspace
        1.5 resample_to_target_resolution -
            uncomment if you want to resample volumes as initial pre-processing.


    Parameters
    ----------
    idx: int
    goldIndices: list
    silverIndices: list
    pnSRC: str
    pnBase: str
    fnSRCPatternVOLgold: str
    fnSRCPatternVOLsilver: str
    fnSRCPatternSEG: str
    fptnVOL: str
    fptnSEG: str

    Returns
    -------
    None

    """
    # Arrange correct file paths
    if idx in goldIndices:
        fptnTMP = fnSRCPatternVOLgold
        tag = 'gold'
    else:
        fptnTMP = fnSRCPatternVOLsilver
        tag = 'silver'
    ffVOL = osj(pnSRC, fptnTMP % idx)
    ffOUT = osj(pnBase, fptnVOL % idx)

    # N4 bias field correction
    ants_n4_bf_correction(ffVOL, ffOUT, n_iterations=[1, 0, 0, 0], run=True)

    # cropping of image and segmentation (robustfov -i
    # SUBJ01_FS30_T2_MRI.nii.gz -r SUBJ01_FS30_T2_MRI_cropped.nii.gz)
    robust_cropping_fsl(ffOUT, ffOUT, run=True)
    #
    # # brain extraction (bet) on cropped image
    nibet(ffOUT, ffOUT, run=True)

    # crop_blackspace --> ffReady
    crop_blackspace(ffOUT, ffOUT)

    # If on-the-fly augmentation will handle resampling then there is no need
    # to resample input data for deep learning.
    targetResolution = .75
    resample_to_target_resolution(ffOUT, ffOUT, targetResolution,
                                  interpolation='Linear', run=True)

    # resample the segmentation labelmap with ffReady as reference
    if tag == "gold":
        ffSEG = osj(pnSRC, fnSRCPatternSEG % idx)
        ffSEGOUT = osj(pnBase, fptnSEG % (idx, tag))
        resample_to_reference_volume(ffOUT, ffSEG, ffSEGOUT,
                                     interpolation='NearestNeighbor')

    print('\n\n\nPreprocessing for subject %02d finished.\n\n\n' % idx)


def main_ants_joint_label_fusion(seq, mfs):
    """
    Main call to perform ANTs joint label fusion.

    Will perform the following steps:
    1. Preprocess all the volumes.
        1.1 ants_n4_bf_correction
        1.2 robust_cropping_fsl
        1.3 nibet
        1.4 crop_blackspace
        1.5 resample_to_target_resolution - if uncommented in preprocess_data
                                            function
    2. ANTS joint label fusion.


    Parameters
    ----------
    seq: int
        2 for T2, 1 for T1, and 0 for SWI.
    mfs: int
        Magnetic field strength, 0 for 7T, 1 for 3T

    Returns
    -------
    None
        Output will be saved to pnBase.
    """
    # File identifiers.
    if mfs == 0:
        f_str = '70'
        mfs_ = '7T'
    elif mfs == 1:
        f_str = '30'
        mfs_ = '3T'
    else:
        raise ValueError('Select 0 or 1 for magnetic field strength.')

    if seq == 2:
        seq_name = 'T2'

    elif seq == 1:
        seq_name = 'T1'

    elif seq == 0:
        seq_name = 'SWI'
    else:
        raise ValueError('Select sequence type from 0 to 2 only.')

    # Output file location
    pnSRC = './vols_all_%s_%s' % (mfs_, seq_name)

    fnSRCPatternVOLgold = 'SUBJ%02d_' + 'FS%s_%s_MRI.nii.gz' % (f_str,
                                                                seq_name)
    fnSRCPatternVOLsilver = 'SUBJ%02d_' + 'FS%s_%s_MRI_NoSeg.nii.gz' % (
        f_str, seq_name)
    fnSRCPatternSEG = 'SUBJ%02d_' + 'FS%s_%s_SEG.nii.gz' % (
        f_str, seq_name)
    pnBase = './vols_all_%s_%s'  % (mfs_, seq_name)
    fptnVOL = 'S%02d_'+ '%s.nii.gz' % seq_name

    goldIndices = [1, 3, 4, 7, 8, 9, 12, 13, 14, 17]
    silverIndices = [2, 5, 6, 10, 11, 15, 16, 18, 19, 20]
    idx_list = goldIndices + silverIndices
    fptnSEG = 'S%02d_SEG%s.nii.gz'

    # Preprocessing steps
    Parallel(n_jobs=5)(delayed(preprocess_data)(idx, \
                                                 goldIndices, \
                                                 silverIndices, \
                                                 pnSRC, pnBase, \
                                                 fnSRCPatternVOLgold, \
                                                 fnSRCPatternVOLsilver, \
                                                 fnSRCPatternSEG, \
                                                 fptnVOL, fptnSEG) \
                        for idx in idx_list)

    # For every other subject: register gold to other subject!
    for idx_mov in goldIndices:
        # for every other subject: register gold to other subject!
        Parallel(n_jobs=5)(
            delayed(ants_register)(idx_fix, idx_mov, pnBase, fptnVOL,
                                   speed='normal') \
            for idx_fix in idx_list if idx_mov != idx_fix)

    # Parameters to create silver standard seg
    idx_gold_set1 = goldIndices[:int(len(goldIndices)/2)] # [1, 3, 4, 7, 8]
    idx_gold_set2 = goldIndices[int(len(goldIndices)/2):]

    # File pattern of MRI volume and gold standard segmentations
    fptnSEG = 'S%02d_SEGgold.nii.gz'

    # Create silver standard segmentations via parallel processing using
    # create_silver_seg()
    parallel_create_silver_seg(idx_gold_set1, idx_list, pnBase, fptnVOL,
                               fptnSEG, n_jobs=10)
    parallel_create_silver_seg(idx_gold_set2, idx_list, pnBase, fptnVOL,
                               fptnSEG, n_jobs=10)

    # File pattern of segmentations volumes of gold and silver standard volumes
    # File pattern of 1st set of gold segmentations
    ff_out = 'S%02d_SEGjoint_from_gold_set1.nii.gz'
    fp_seg = 'S%02d_on_%02d_SEG.nii.gz'
    fuse_labels(idx_gold_set1, idx_list, pnBase, fp_seg, ff_out)

    # File pattern for 2nd set of gold standard segmentation.
    ff_out = 'S%02d_SEGjoint_from_gold_set2.nii.gz'
    fuse_labels(idx_gold_set2, idx_list, pnBase, fp_seg, ff_out)


def resample_t1_and_swi_to_t2():
    """ Resample T1 and SWI volumes to T2 volumes (preprocessed volumes,
    such as bet). """
    # Location of preprocessed 3T T2 volumes
    osj = os.path.join
    pathname_src_t2 = './vols_all_3T_T2'
    file_pattern_t2 = 'S%02d_T2.nii.gz'

    # Location of 3T T1 and SWI volumes
    pathname_src_t1_swi = './vols_all'
    file_pattern_t1 = 'SUBJ%02d_FS30_T1_MRI.nii.gz'
    file_pattern_swi = 'SUBJ%02d_FS30_SWI_MRI.nii.gz'

    # Output filename of T1 and SWI
    output_path = './vols_all_3T_t1_t2_swi'
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    t1_output_fname = osj(output_path, 'S%02d_T1.nii.gz')
    swi_output_fname = osj(output_path, 'S%02d_SWI.nii.gz')

    # Non-gold standard volumes has 'MRI_NoSeg' in filename istead of 'MRI'
    goldIndices = [1, 3, 4, 7, 8, 9, 12, 13, 14, 17]

    for i in range(1, 21):
        if i in goldIndices:
            img_t2_p = osj(pathname_src_t2, file_pattern_t2 % i)
            img_t1 = osj(pathname_src_t1_swi, file_pattern_t1 % i)
            img_swi = osj(pathname_src_t1_swi, file_pattern_swi % i)
        else:
            img_t2_p = osj(pathname_src_t2, file_pattern_t2.replace('MRI', \
                                                            'MRI_NoSeg') % i)
            img_t1 = osj(pathname_src_t1_swi, file_pattern_t1.replace('MRI', \
                                                            'MRI_NoSeg') % i)
            img_swi = osj(pathname_src_t1_swi, file_pattern_swi.replace('MRI', \
                                                            'MRI_NoSeg') % i)

        # Load T2 volume
        img_t2 = sitk.ReadImage(img_t2_p)
        img_t2_mask = sitk.GetArrayFromImage(img_t2) != 0
        if os.path.exists(img_t1):
            print(img_t1)
            img_t1 = resample_to_reference_volume(img_t2_p, img_t1,
                                                  t1_output_fname % i,
                                                  interpolation='Cubic',
                                                  save=False)
            arr_t1 = sitk.GetArrayFromImage(img_t1)
            arr_t1 = arr_t1 * img_t2_mask
            new_img_t1 = sitk.GetImageFromArray(arr_t1)
            new_img_t1.CopyInformation(img_t1)
            sitk.WriteImage(new_img_t1, t1_output_fname % i)
            if os.path.exists(t1_output_fname % i):
                print(" == File SAVED %s ==" % str(t1_output_fname % i))

        if os.path.exists(img_swi):
            print(img_swi)
            img_swi = resample_to_reference_volume(img_t2_p, img_swi,
                                                  t1_output_fname % i,
                                                  interpolation='Cubic',
                                                   save=False)
            arr_swi = sitk.GetArrayFromImage(img_swi)
            arr_swi = arr_swi * img_t2_mask
            new_img_swi = sitk.GetImageFromArray(arr_swi)
            new_img_swi.CopyInformation(img_swi)
            sitk.WriteImage(new_img_swi, swi_output_fname % i)
            if os.path.exists(swi_output_fname % i):
                print(" == File SAVED %s ==" % str(swi_output_fname % i))


def get_path_from_dir(folder_path, str_of_interest='T1|T2|SWI'):
    """
    Get path which contains the following strings given
    a directory.

    Paramaters:
        folder_path: str
            '/path/of/dir/'
        str_of_interest: str
            Regex to use to find T1, T2, and SWI paths.
            'T1|T2|SWI'
    Returns
        res_paths: list
    """
    assert type(folder_path) is str, 'folder_path must be str'
    assert type(str_of_interest) is str, 'str_of_interest must be str'

    # Check for folders with string names "T1" or "T2" or "SWI"
    seg_folder = []  # folders of interest
    for path, dirs, files in os.walk(folder_path):
        cur_folder = path.split("/")[-1]
        cur_folder = cur_folder.lower()
        seg_folder.append(path)

    seg_folder = pd.Series(seg_folder)
    _mask = seg_folder.str.lower().str.contains(str_of_interest.lower())
    res_paths = seg_folder[_mask]
    res_paths = list(res_paths)
    return res_paths


def format_path_to_df(path_lists):
    """
    Format list of paths into pandas dataframe

    Parameters:
        path_lists: list of str paths
            Paths in list should have this format.
        './with_segmentation/DE_ES_PROBAND_SUBJ3_39M_2015_11_06_10_44_13_82/'
                'Study_1.5_T/SWI_randomnumbers'
    Returns:
        res_df: pandas DataFrame
            Resulting formatted paths into dataframe.
    """
    # Create table of list of files with dicom
    cur_data = []
    for k, cur_folder in enumerate(path_lists):
        cur_files = os.listdir(cur_folder)

        # Will not include image*.dcm files in with_segmentation folder
        if 'with_segmentation' in cur_folder.split('/')[:2]:
            cur_dicom_files = [x for x in cur_files if
                               not re.match("^image", x.lower()) and x.endswith(
                                   "dcm")]
            cur_nifty_files = [x for x in cur_files if
                               not re.match("^image", x.lower()) and x.endswith(
                                   ".nii.gz")]
            with_seg = True
        else:
            cur_dicom_files = [x for x in cur_files if
                               re.match("^image", x.lower()) and x.endswith(
                                   "dcm")]
            cur_nifty_files = [x for x in cur_files if
                               not re.match("^image", x.lower()) and x.endswith(
                                   ".nii.gz")]
            with_seg = False

        # Parsing information in current path
        cur_folder_splits = cur_folder.split('/')
        cur_id = cur_folder_splits[2]
        cur_id = '_'.join(
            cur_id.split('_')[3:4])  # only take the subject numbers
        cur_study = cur_folder_splits[3]
        cur_stype = cur_folder_splits[4].split('_')[0]

        # Insert values in dataframe
        cur_data.append([with_seg, cur_id, cur_study, cur_stype, cur_folder,
                         len(cur_nifty_files), cur_nifty_files,
                         len(cur_dicom_files), cur_dicom_files])

    # Format result into dataframe with correct column names
    data_df = pd.DataFrame(cur_data)
    data_df.columns = ['has_segmentation', 'id', 'study', 'seq_type', 'p_path',
                       'len_nifti', 'nifti_files', 'len_dicom', 'dicom_files']
    data_df.sort_values(
        ['has_segmentation', 'id', 'study', 'seq_type', 'p_path'], inplace=True)
    return data_df


def get_uniq_segmentations(seg_series):
    """
    Find unique segmentations given a pd.Series of list
    of segmentations names. Will replace whitespace between words with "_".

    Parameters:
        seg_series: pd.Series
            Padas series containing list of strings of segmmentaion names
    Returns:
        res_list: list, list
            List of segmentation names, List of list of segmentation names
    """
    assert type(seg_series) is pd.Series, 'seg_series must be pandas Series'
    all_nifti_list = []
    for i in seg_series:
        cur_nifti = [re.sub('\s+', '_', x) for x in i if
                     not re.search('\d', re.sub('\s+', '_', x))]
        cur_nifti = [x for x in cur_nifti if 'onmri' not in x]
        all_nifti_list.append(cur_nifti)
    return set.intersection(*[set(x) for x in all_nifti_list if len(x) > 1]),\
           all_nifti_list


def get_data(data, seq_type, study_type, with_seg=True):
    """
    Get data summary given sequence type and study type.

    Parameters:
        data: pandas.DataFrame
            Dataframe of interest.
        seq_type: str
            Select one from 'T1' or 'T2' or 'SWI'
        study_type: str
            Select one from 'Study_3.0_T' or 'Study_7.0_T' or 'Study_1.5_T'
    Returns:
        data_of_interest: pandas.DataFrame
    """
    data_df = data.copy()
    assert type(data_df) is pd.DataFrame, 'data_df must be a pandas DataFrame'
    assert seq_type in ['T1', 'T2', 'SWI']
    assert study_type in ['Study_3.0_T', 'Study_7.0_T', 'Study_1.5_T']
    data_df_cols = data_df.columns
    assert 'has_segmentation' in data_df_cols
    assert 'seq_type' in data_df_cols
    assert 'study' in data_df_cols
    assert 'len_renamed_nifti_seg' in data_df_cols
    assert 'id' in data_df_cols
    assert 'renamed_mri_seq' in data_df_cols

    # len_renamed_nifti_seg will be > 0 for files with segmentation and
    # 0 for non-segmented files.
    if with_seg:
        data_of_interest = data_df[(data_df.has_segmentation == with_seg) \
                                   & (data_df.seq_type == seq_type) \
                                   & (data_df.study == study_type) \
                                   & (data_df.len_renamed_nifti_seg > 0)]
    else:
        data_of_interest = data_df[(data_df.has_segmentation == with_seg) \
                                   & (data_df.seq_type == seq_type) \
                                   & (data_df.study == study_type) \
                                   & (data_df.len_renamed_nifti_seg == 0)]

    # Add file sizes of MRI sequences and take the largest one
    mri_seq_size_list = []
    for x in data_of_interest.renamed_mri_seq:
        print(x, type(x))
        if type(x) is float and np.isnan(x):
            mri_seq_size_list.append(np.nan)
        else:
            print(x, type(x))
            cur_size = os.stat(x).st_size
            print(cur_size)
            mri_seq_size_list.append(cur_size)
    data_of_interest['mri_seq_size_bytes'] = mri_seq_size_list
    data_of_interest = data_of_interest.sort_values(
        ['id', 'mri_seq_size_bytes'])
    data_of_interest.drop_duplicates(['id'], keep='last', inplace=True)

    print('Number of subjects: {}'.format(data_of_interest.id.size))
    print('Number of rows per subject:')
    print(data_of_interest.id.value_counts())
    return data_of_interest


def get_segmentation_summary(seg_list, col_names):
    """
    Get segmentation summary given a list of list of sementations

    Parameters:
        seg_list: list
            List of list of segmentations.
        col_names: list of str
            Subject names to use
    Returns:
        seg_summary: pandas.DataFrame
            Dataframe where columns are subject names and
            indexnames are segmentation names
    """
    assert type(seg_list) is list

    # Given list of list of segmentation names create a pandas dataframe summary
    cur_lst_series = []
    for x in seg_list:
        cur_item = pd.Series([i.replace('.nii.gz', '') for i in x])
        cur_lst_series.append(
            pd.DataFrame(np.ones_like(cur_item), index=cur_item))
    seg_summary = pd.concat(cur_lst_series, 1)
    seg_summary.columns = list(col_names)
    return seg_summary


def cp_mri(data, seq_type, study_type, dst_folder, uniq_str='_MRI.nii.gz'):
    """
    Will copy mri sequence with formatted filename

    Parameters:
        data: pd.DataFrame
            Dataframe containing path of mri sequence
        seq_type: str
        study_type: str
        uniq_str: str
            Suffix '_MRI.nii.gz'
    Returns:
        None
    """
    data_cols = data.columns
    assert "renamed_mri_seq" in data_cols
    assert seq_type in ['T1', 'T2', 'SWI']
    assert study_type in ['Study_3.0_T', 'Study_7.0_T', 'Study_1.5_T']
    assert type(uniq_str) is str

    osj = os.path.join
    for i in data.renamed_mri_seq:
        if type(i) is float and np.isnan(i):
            print('print %s has no sequence' % (i))
        else:
            cur_subj_name = 'SUBJ%02d' % int(
                i[re.search('SUBJ', i).end():].split('_')[0])
            print()
            cur_study_type = '_FS' + study_type.split('_')[1]
            cur_seq_type = '_' + seq_type
            cur_new_name = cur_subj_name + cur_study_type.replace('.',
                                                                  '') + \
                           cur_seq_type

            new_mri_path = osj(dst_folder, cur_new_name + uniq_str)
            print('from %s' % (i))
            copyfile(i, new_mri_path)
            if os.path.exists(new_mri_path):
                print('==> to ==> %s' % (new_mri_path))
            else:
                print('=== Not copied ===')


def rename_nifti_of_subj(data_df, subj_name, str_find, str_replace):
    """
    Rename nifti file path of a given subject.
    Replace the wrong string in the file path.

    Parameters:
        data_df: pd.DataFrame
            Dataframe of interest.
        subj_name: str
            Subject name where the nifti path is.
        str_find: str
            String to find in the nifti path list.
        str_replace: str
            str_find string will be replace to this str_replace string.

    Returns:
        None
        Will rename nifti file path
    """
    data_df_col = data_df.columns
    assert 'id' in data_df_col
    assert 'renamed_nifti_seg' in data_df_col

    # Rename incorect segmentation names
    # Nifti paths of a given subject
    renamed_nifti_paths = \
    data_df.renamed_nifti_seg[data_df.id == subj_name].values[0]
    renamed_nifti_paths = renamed_nifti_paths

    cur_nifti_to_rename = [x for x in renamed_nifti_paths if str_find in x]

    # Rename nifti file
    for nifti in cur_nifti_to_rename:
        new_path = nifti.replace(str_find, str_replace)
        os.rename(nifti, new_path)

        if os.path.exists(new_path):
            print('{} ==> renamed to ==> {}'.format(nifti, new_path))


def get_mri_and_seg_path_df(data):
    """
    Get path of MRI sequences and segmentations.

    Parameters:
        data: pd.DataFrame

    Return
        res_df: pd.DataFrame

    """
    assert type(data) is pd.DataFrame
    data_cols = data.columns
    assert 'p_path' in data_cols
    assert 'nifti_files' in data_cols

    data_df = data.copy()

    nifti_seg_list = []
    mri_seq_list = []
    for p in data_df.values:
        cur_path = p[4]

        # print(cur_path)
        cur_nifti = p[6]
        mri_seq = [x for x in cur_nifti if re.search('\d', x)]
        seg_paths = [x for x in cur_nifti if not re.search('\d', x)]
        seg_paths = [os.path.join(cur_path, x) for x in seg_paths]

        # If current path has no mri_seq set it to nan
        if len(mri_seq) > 0:
            mri_seq_path = os.path.join(cur_path, mri_seq[0])
        else:
            mri_seq_path = np.nan

        # If current path has no segmentation set it to nan
        if len(seg_paths) == 0:
            seg_paths = []
        mri_seq_list.append(mri_seq_path)
        nifti_seg_list.append(seg_paths)

    # Add additional columns in data_df for paths of MRI sequence and
    # segmentations
    data_df.reset_index(drop=True, inplace=True)
    data_df['renamed_nifti_seg'] = pd.Series(nifti_seg_list)
    data_df['renamed_mri_seq'] = pd.Series(mri_seq_list)
    data_df['len_renamed_nifti_seg'] = data_df.renamed_nifti_seg.apply(len)

    return data_df


def itk_to_numpy_local(vol):
    """
    Local normalization using winsorization between the 0.1 to 99.9
    percentile of image intensities.
    Parameters
    ----------
    vol: sitk.Image

    Returns
    -------
    npvol: np.ndarray
    """
    npvol = sitk.GetArrayFromImage(vol).astype(dtype=float)
    volmin = np.percentile(npvol.ravel(), 0.1)  # np.min(npvol)
    volmax = np.percentile(npvol.ravel(), 99.9)  # np.max(npvol)
    npvol = (npvol - volmin) / (volmax - volmin)
    npvol[npvol < 0.0] = 0.0
    npvol[npvol > 1.0] = 1.0
    return npvol


def itk_to_numpy(vol):
    """
    Convert sitk.Image object to numpy array.

    Parameters
    ----------
    vol: sitk.Image

    Returns
    -------
    npvol: np.ndarray
    """
    return itk_to_numpy_local(vol)

def itk_to_np_segmentation(vol, total_class, select_idx):
    """
    Convert sitk.Image to np.array

    Parameters
    ----------
    vol: sitk.Image
    total_class: int
        Total number of available class
    selet_idx: list
        List of segmentation indices

    Returns
    -------
    npvol: np.ndarray
    """
    npvol = sitk.GetArrayFromImage(vol).astype(dtype=int)
    npvol = to_categorical(npvol, num_classes=total_class)
    npvol = npvol[..., select_idx]
    return npvol
