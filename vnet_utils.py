import SimpleITK as sitk
from PIL import Image
import numpy as np
import tensorflow as tf
import keras
from skimage.measure import label
# from scipy.ndimage.measurements import label
from nipype.interfaces import c3

def normalizeArray(array):
    # TODO docstring
    return (array-array.min()) / (array.max() - array.min())

def saveImages(imgs,atSlice,path):
    # TODO docstring
   np_imgs = [im.reshape(128,128,128).astype(dtype=float)[atSlice] for im in
              imgs]
   np_imgs = [255*normalizeArray(im) for im in np_imgs]
   con_img = Image.fromarray(np.concatenate(np_imgs,axis=1)).convert('RGB')
   con_img.save(path)

def predictByModel(input_path,output_path):
    # TODO docstring
    def getCenteredOrigin(img):
        return np.array(img.GetDirection()).reshape(3, 3).dot(np.array([-0.5, -0.5, -0.5])) * np.array(
            img.GetSize()) * np.array(img.GetSpacing())

    def dice(labels, prediction):
      dc = 2.0 * \
           tf.reduce_sum(labels * prediction, axis=[1, 2, 3, 4]) / \
           tf.reduce_sum(labels ** 2 + prediction ** 2, axis=[1, 2, 3, 4]) + np.finfo(float).eps

      return dc

    def dice_loss(labels, prediction):
      return 1.0 - dice(labels, prediction)

    def compute_dice_loss(labels, prediction):
      return tf.reduce_mean(
         dice_loss(labels=labels, prediction=prediction)
      )

    image = sitk.ReadImage(input_path)
    m = keras.models.load_model("model.pkl",custom_objects={'compute_dice_loss':compute_dice_loss})
    np_image = sitk.GetArrayFromImage(image).astype(dtype=float)
    np_seg = m.predict(np_image.reshape(1,60,60,60,1))
    np_seg_final = np.array(np_seg[1]).reshape(60,60,60)

    seg = sitk.GetImageFromArray(maskLargestConnectedComponent(np_seg_final).astype(float))

    # copy metadata/properties
    seg.SetSpacing(image.GetSpacing())
    for k in image.GetMetaDataKeys():
      seg.SetMetaData(k, image.GetMetaData(k))
    # center image
    seg.SetOrigin(getCenteredOrigin(image))
    seg.SetDirection(image.GetDirection())

    sitk.WriteImage(seg,output_path)

def maskLargestConnectedComponent(image_np):
    # TODO docstring
    image = sitk.GetImageFromArray(image_np)
    f = sitk.BinaryThresholdImageFilter()
    # define threshold to be at half beween min and max
    f.SetLowerThreshold(image_np.min() + 0.5*(image_np.max() - image_np.min()))
    # set upper threshold to exceed max image value
    f.SetUpperThreshold(image_np.max() + 10.0)
    i_bin = f.Execute(image)
    # find connected components
    labeled_i = sitk.ConnectedComponent(i_bin)
    # convert to numpy array
    l_i_np = sitk.GetArrayFromImage(labeled_i)
    # get the label map with the largest component
    return l_i_np == np.argsort(np.bincount(l_i_np.ravel()))[-2]

# from https://github.com/faustomilletari/VNet/blob/master/utilities.py
def computeQualityMeasures(lP,lT):
    # TODO docstring
    quality={}
    labelPred=sitk.GetImageFromArray(lP, isVector=False)
    labelTrue=sitk.GetImageFromArray(lT, isVector=False)
    hausdorffcomputer=sitk.HausdorffDistanceImageFilter()
    hausdorffcomputer.Execute(labelTrue>0.5,labelPred>0.5)
    quality["avgHausdorff"]=hausdorffcomputer.GetAverageHausdorffDistance()
    quality["Hausdorff"]=hausdorffcomputer.GetHausdorffDistance()

    dicecomputer=sitk.LabelOverlapMeasuresImageFilter()
    dicecomputer.Execute(labelTrue>0.5,labelPred>0.5)
    quality["dice"]=dicecomputer.GetDiceCoefficient()

    return quality


def resampleToTargetResolution(ffVOL, ffOUT, targetResolution,
                               interpolation='Linear', run=False):
    """
    # TODO docstring
    Resample MRI volume to a target resolution.
    Parameters:
        interpolation: str
            Can have values: <NearestNeighbor|Linear|Cubic|Sinc|Gaussian>
    Returns:
    """
    if isinstance(targetResolution, float):
        targetResolution = [targetResolution] * 3

    # using c3d
    # c3d documentation at: https://sourceforge.net/p/c3d/git/ci/master/tree/doc/c3d.md
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


def keep_largest_cc(vol):
    """
    Given binary label volumes, keep the largest connected component of
    given volume.

    Parameters
    ----------
    vol: numpy.nd.array
        Binary label map/volume

    Returns
    -------
    res_vol: numpy.nd.array
        Output dimension is same as input but only keeping the largest
        component.
    """
    try:
        labels = label(vol, connectivity=3)
        largest_component_idx = np.argmax(np.bincount(labels.flatten())[1:]) + 1
        res_vol = np.zeros_like(labels, dtype=np.uint8)
        res_vol[labels == largest_component_idx] = 1
        return res_vol
    except ValueError as e:
        print(e)
        return vol

