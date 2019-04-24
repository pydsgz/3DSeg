import SimpleITK as sitk
import numpy as np
import logging


class VolumeDataGenerator(object):
    def __init__(self,
                 enableDeformation=False,
                 enableRotation=False,
                 enableTranslation=False,
                 enableFlips=False,
                 enableCropZoom=False,
                 rotationCenterFn=None,
                 rotationAngleRanges=[0.0, 0.0, 0.0],
                 translationRanges=[0.0, 0.0, 0.0],
                 deformationControlPoints=3,
                 deformationSigma=(0.0, 0.0),
                 FlipAxes=[False, False, False],
                 FlipProb=[0.0, 0.0, 0.0],
                 cropZoomLevelRange=[(1.0, 1.0), (1.0, 1.0), (1.0, 1.0)],
                 constant_fill_value=0.0,
                 size=[256, 256, 256]
                 ):
        """
        :param enableDeformation:
         enables deformation module.
        :param enableRotation:
         enables rotation module.
        :param enableTranslation:
         enables translation module.
        :param enableFlips:
         enables vertical/horizontal flips module.
        :param enableCropZoom:
         enables zoom/crop module.
        :param rotationCenterFn:
         defines function to compute center point of rotation.
         If None, given origin of image is used.
        :param rotationAngleRanges:
         defines euler angles along each axis for rotation.
        :param translationRanges:
         defines translation variation between -t and t along each axis.
        :param deformationControlPoints:
         defines number of control points used for deformable transformation.
        :param deformationSigma:
         defines sigma for deformable transformation.
        :param FlipAxes:
         defines allowed flips for each axis.
         If [False,False,False], no flips are performed!
        :param FlipProb:
         defines the probabilities to perform a flip for each axis.
        :param cropZoomLevelRange:
         defines the ZoomIn/ZoomOut ranges for each axis.
        :param constant_fill_value:
         defines pixel value for pixels outside the given volume.
        :param size:
         a list with ints as size of the volume in 3 dimension.
        """
        # Rotation
        self.enableRotation = enableRotation
        self.rotationAngleRanges = rotationAngleRanges
        self.rotationCenterFn = rotationCenterFn if rotationCenterFn else lambda image: [0, 0, 0]
        # Translation
        self.enableTranslation = enableTranslation
        self.translationRanges = translationRanges
        # Flips
        self.enableFlips = enableFlips
        self.FlipProp = FlipProb
        self.FlipAxes = FlipAxes
        # Zoom/Crop in %
        self.enableCropZoom = enableCropZoom
        self.cropZoomLevels = cropZoomLevelRange
        # Deformation
        self.enableDeformation = enableDeformation
        self.deformationControlPoints = deformationControlPoints
        self.deformationSigma = deformationSigma
        # FillMode = constant -> default fill pixel value
        self.constant_pixel = constant_fill_value
        self.size = size

        # print properties
        logging.debug(self)

    # PUBLIC METHODS
    def augmentSingleImage(self, img, config=None, seg_vol=False,
                           composite_transform=False):
        logging.debug("Generate new image...")
        if config is None:
            logging.debug("Generate new config...")
            config = self.__generateRandomConfig__(img)
        if composite_transform:
            return self.__augmentImageByConfig__(img, config, seg_vol,
                                                 composite_transform=True)
        else:
            return self.__augmentImageByConfig__(img, config, seg_vol)

    def getNewImageFromBatch(self, image_label_stack,
                             composite_transform=False):
        """
        :param image_label_stack:
             list of (label,registered_images)
             see [ (label, [image1,image2,image3,...]),...]
        :rtype: yields (label,augmented_registered_images, selected_index)
        """
        if len(image_label_stack) == 0:
            raise Exception("No items in given stack.")
        test_label, test_images = image_label_stack[0]
        if len(test_images) == 0:
            raise Exception("No images at position 0!")

        if self.enableDeformation:
            logging.info(
                "Deformable transformation is slow! We may should generate multiple images based on each deformed image.")

        while True:
            index_next = np.random.choice(range(len(image_label_stack)))
            next_label, next_imgs = image_label_stack[index_next]

            # When Multi-modal, next_imgs is [[vol_1, T2, vol_1, T1,
            # vol_1_SWI], [vol_seg]]
            if isinstance(next_imgs[0], list) and len(next_imgs[0]) > 1:
                # generate random sample
                config = self.__generateRandomConfig__(next_imgs[0][0])
                transformed = []
                transformed_vol_list = []
                for cur_vol in next_imgs[0]:
                    cur_transformed_vol = self.augmentSingleImage(cur_vol,
                                                                  config,
                                                                  composite_transform=True)
                    transformed_vol_list.append(cur_transformed_vol)

                # Append multi-modal transformed volume
                transformed.append(transformed_vol_list)
                # Append single modal (T2) segmentation
                transformed.append(self.augmentSingleImage(next_imgs[1],
                                                           config,
                                                           seg_vol=True,
                                                           composite_transform=True))
                yield next_label, transformed, index_next

            else:
                config = self.__generateRandomConfig__(next_imgs[0])
                transformed = []
                # for img in next_imgs:
                #     transformed.append(self.augmentSingleImage(img, config))
                if composite_transform:
                    transformed.append(self.augmentSingleImage(next_imgs[0],
                                                               config,
                                                               composite_transform=True))
                    transformed.append(self.augmentSingleImage(next_imgs[1], config,
                                                               seg_vol=True,
                                                               composite_transform=True))
                else:
                    transformed.append(self.augmentSingleImage(next_imgs[0],
                                                               config))
                    transformed.append(self.augmentSingleImage(next_imgs[1], config,
                                                               seg_vol=True))

                yield next_label, transformed, index_next

    # useful high-level functions for users
    def __augmentImageByConfig__(self, image, config, seg_vol=False,
                                 composite_transform=False):
        output_img = image

        # Transform order:
        # First in first last execute in composite transform.
        # Translation -> Rotation -> Deform -> Crop/Zoom
        if composite_transform:
            composite = sitk.Transform(output_img.GetDimension(),sitk.sitkComposite)
            for augmentMethod in [self.__conditionalCropZoom__,
                                  self.__conditionalDeformation__,
                                  self.__conditionalTranslationAndRotation__]:
                composite.AddTransform(augmentMethod(output_img, config,
                                                     seg_vol,
                                                     return_transform=True))
            size = self.size
            origin = image.GetOrigin()
            spacing = [1.0, 1.0, 1.0]
            direction = image.GetDirection()
            if seg_vol:
                output_img = sitk.Resample(image, size, composite,
                                           sitk.sitkNearestNeighbor, origin,
                                           spacing, direction)
            else:
                output_img = sitk.Resample(image, size, composite,
                                           sitk.sitkLinear, origin,
                                           spacing, direction)
        # Crop/Zoom -> Deform -> Flips -> Rotation -> Translation
        else:
            for augmentMethod in [self.__conditionalCropZoom__,
                                  self.__conditionalDeformation__,
                                  self.__conditionalFlips__,
                                  self.__conditionalTranslationAndRotation__]:
                output_img = augmentMethod(output_img, config, seg_vol)
        return output_img

    def __generateRandomConfig__(self, main_image):
        # Rotation/Translation
        rotAngles = [np.random.uniform(-r, r) for r in self.rotationAngleRanges]
        transl = [np.random.uniform(-t, t) for t in self.translationRanges]

        # Deformable
        sigma = np.random.uniform(*self.deformationSigma)
        # setup transform initializer
        params_vector_length = len(sitk.BSplineTransformInitializer(
            main_image, [self.deformationControlPoints] * main_image.GetDimension()).GetParameters())
        deform_params_vector = np.random.randn(params_vector_length)

        # Flips
        axis_flip = []
        for a, p in zip(self.FlipAxes, self.FlipProp):
            # for each axis: do we want to flip?
            if a and np.random.random() <= p:
                axis_flip.append(True)
            else:
                axis_flip.append(False)

        # Crop/Zoom
        zoom_level = [np.random.uniform(*r) for r in self.cropZoomLevels]

        ret_val = {"rotation_angles": rotAngles,
                   "translation": transl,
                   "deform_sigma": sigma,
                   "deform_params": deform_params_vector,
                   "flips": axis_flip,
                   "cropzoom": zoom_level
                   }
        # if seg_vol:
        #     ret_val["vol_type"] = "segmentation"
        # else:
        #     ret_val["vol_type"] = "volume"
        print(np.random.rand(1))
        return ret_val

    # high-level helper functions
    def __conditionalCentering__(self, image):
        center = []
        center[0] = image.GetLargestPossibleRegion().GetSize()[0] / 2;
        center[1] = image.GetLargestPossibleRegion().GetSize()[1] / 2;
        center[2] = image.GetLargestPossibleRegion().GetSize()[2] / 2;


    def __conditionalTranslationAndRotation__(self, image, config, seg_vol,
                                              return_transform=False):  #
        # ROTATION and TRANSLATION
        output_img = image
        if return_transform:
            if self.enableRotation:
                rotAngles = config.get("rotation_angles")
                # rand_fun = np.random.uniform
                # rotAngles = [rand_fun(-r, r) for r in self.rotationAngleRanges]
                rotCenter = self.rotationCenterFn(image)
                logging.debug(
                    "Rotation applied with parameters:\t {}".format(rotAngles))
            else:
                # no Rotation
                rotCenter = [0.0, 0.0, 0.0]
                rotAngles = [0.0, 0.0, 0.0]
            if self.enableTranslation:
                # rand_fun = np.random.uniform
                # transl = [rand_fun(-t, t) for t in self.translationRanges]
                transl = config.get("translation")
                logging.debug(
                    "Translation applied with parameters:\t {}".format(transl))
            else:
                # no Translation
                transl = [0.0, 0.0, 0.0]

            # conditional ROTATION or/and TRANSLATION at once!
            if self.enableRotation or self.enableTranslation:
                transform = self.__generateRotationTranslation__(rotCenter,
                                                                 rotAngles,
                                                                 transl)
            return transform
        else:
            if self.enableRotation:
                rotAngles = config.get("rotation_angles")
                #rand_fun = np.random.uniform
                #rotAngles = [rand_fun(-r, r) for r in self.rotationAngleRanges]
                rotCenter = self.rotationCenterFn(image)
                logging.debug("Rotation applied with parameters:\t {}".format(rotAngles))
            else:
                # no Rotation
                rotCenter = [0.0, 0.0, 0.0]
                rotAngles = [0.0, 0.0, 0.0]
            if self.enableTranslation:
                #rand_fun = np.random.uniform
                #transl = [rand_fun(-t, t) for t in self.translationRanges]
                transl = config.get("translation")
                logging.debug("Translation applied with parameters:\t {}".format(transl))
            else:
                # no Translation
                transl = [0.0, 0.0, 0.0]

            # conditional ROTATION or/and TRANSLATION at once!
            if self.enableRotation or self.enableTranslation:
                transform = self.__generateRotationTranslation__(rotCenter, rotAngles, transl)
                if seg_vol:
                    filter = self.__resampleTransformFilter__(image, transform,
                                                              sitk.sitkNearestNeighbor)
                else:
                    filter = self.__resampleTransformFilter__(image, transform)
                output_img = filter.Execute(image)
            return output_img

    def __conditionalDeformation__(self, image, config, seg_vol,
                                   return_transform=False):
        output_img = image
        # DEFORMATION
        if return_transform:
            if self.enableDeformation:
                # rand_fun = np.random.uniform
                # sigma = rand_fun(*self.deformationSigma)
                sigma = config.get("deform_sigma")
                params = config.get("deform_params")
                transform = self.__generateDeformationTransform__(output_img,
                                                                  sigma, params)

                if seg_vol:
                    filter = self.__resampleTransformFilter__(output_img,
                                                              transform,
                                                              sitk.sitkNearestNeighbor)
                else:
                    filter = self.__resampleTransformFilter__(output_img,
                                                              transform)
                logging.debug(
                    "Deformation applied with parameters:\t sigma = {}".format(
                        sigma))
            return transform
        else:
            if self.enableDeformation:
                #rand_fun = np.random.uniform
                #sigma = rand_fun(*self.deformationSigma)
                sigma = config.get("deform_sigma")
                params = config.get("deform_params")
                transform = self.__generateDeformationTransform__(output_img,
                                                                  sigma, params)

                if seg_vol:
                    filter = self.__resampleTransformFilter__(output_img, transform,
                                                              sitk.sitkNearestNeighbor)
                else:
                    filter = self.__resampleTransformFilter__(output_img, transform)
                output_img = filter.Execute(output_img)
                logging.debug("Deformation applied with parameters:\t sigma = {}".format(sigma))
            return output_img

    def __conditionalCropZoom__(self, image, config, seg_vol,
                                return_transform=False):
        output_img = image
        # Crop/Zoom
        if return_transform:
            if self.enableCropZoom:
                # compute cut boundaries
                old_size = output_img.GetSize()
                # rand_fun = np.random.uniform
                # zoom_level = [rand_fun(*r) for r in self.cropZoomLevels]
                zoom_level = config.get("cropzoom")
                # scale image
                transform = sitk.ScaleTransform(output_img.GetDimension(),
                                                zoom_level)
                # resample based on old image size to remove outer regions
                if seg_vol:
                    transform_filter = self.__resampleTransformFilter__(
                        output_img, transform,
                        sitk.sitkNearestNeighbor)
                else:
                    transform_filter = self.__resampleTransformFilter__(
                        output_img, transform)
                transform_filter.SetSize(old_size)
                logging.debug("Crop/Zoom applied with parameters:\t {}".format(
                    zoom_level))
            return transform
        else:
            if self.enableCropZoom:
                # compute cut boundaries
                old_size = output_img.GetSize()
                #rand_fun = np.random.uniform
                #zoom_level = [rand_fun(*r) for r in self.cropZoomLevels]
                zoom_level = config.get("cropzoom")
                # scale image
                transform = sitk.ScaleTransform(output_img.GetDimension(), zoom_level)
                # resample based on old image size to remove outer regions
                if seg_vol:
                    transform_filter = self.__resampleTransformFilter__(output_img, transform,
                                                              sitk.sitkNearestNeighbor)
                else:
                    transform_filter = self.__resampleTransformFilter__(output_img, transform)
                transform_filter.SetSize(old_size)
                output_img = transform_filter.Execute(output_img)
                logging.debug("Crop/Zoom applied with parameters:\t {}".format(zoom_level))
            return output_img

    def __conditionalFlips__(self, image, config, seg_vol,
                             return_transform=False):
        output_img = image
        # FLIPPING/MIRROR
        if return_transform:
            if self.enableFlips:
                #axis_flip = []
                #for a, p in zip(self.FlipAxes, self.FlipProp):
                #    # for each axis: do we want to flip?
                #    if a and np.random.random() <= p:
                #        axis_flip.append(True)
                #    else:
                #        axis_flip.append(False)
                axis_flip = config.get("flips")
                if True in axis_flip:
                    logging.debug("Flips applied:\t\t\t {}".format(axis_flip))
                    filter = self.__generateFlipFilter__(axis_flip)
                    return filter
        else:
            if self.enableFlips:
                #axis_flip = []
                #for a, p in zip(self.FlipAxes, self.FlipProp):
                #    # for each axis: do we want to flip?
                #    if a and np.random.random() <= p:
                #        axis_flip.append(True)
                #    else:
                #        axis_flip.append(False)
                axis_flip = config.get("flips")
                if True in axis_flip:
                    logging.debug("Flips applied:\t\t\t {}".format(axis_flip))
                    filter = self.__generateFlipFilter__(axis_flip)
                    output_img = filter.Execute(output_img)
            return output_img

    # low-level helper functions
    def __generateDeformationTransform__(self, volume, sigma, parameters):
        transfromDomainMeshSize = [self.deformationControlPoints] * volume.GetDimension()
        # setup transform initializer
        transform = sitk.BSplineTransformInitializer(volume, transfromDomainMeshSize)
        # setup deform parameters
        params = transform.GetParameters()
        paramsNp = np.asarray(params, dtype=float)

        #paramsNp = paramsNp + np.random.randn(paramsNp.shape[0]) * sigma
        paramsNp =  paramsNp + parameters * sigma

        # remove z deformations! The resolution in z is too bad
        paramsNp[0:int(len(params) / 3)] = 0

        transform.SetParameters(paramsNp)
        return transform

    def __generateFlipFilter__(self, axes):
        filter = sitk.FlipImageFilter()
        filter.SetFlipAxes(axes)
        return filter

    def __generateRotationTranslation__(self, rotCtr, rotAxes, translation):
        rX, rY, rZ = rotAxes
        return sitk.Euler3DTransform(rotCtr, rX, rY, rZ, translation)

    def __resampleTransformFilter__(self, vol, transform, interpolator=sitk.sitkLinear):
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(vol)
        resampler.SetInterpolator(interpolator)
        resampler.SetDefaultPixelValue(self.constant_pixel)
        resampler.SetTransform(transform)
        return resampler

    # pretty printer
    def __repr__(self):
        yes_no = lambda x: " is ENABLED." if x else " is disabled."
        repr_text = "VolumeAugmentation\n"
        repr_text += "Translation\t\t {}\n".format(yes_no(self.enableTranslation))
        repr_text += "Rotation\t\t {}\n".format(yes_no(self.enableRotation))
        repr_text += "Deformation\t\t {}\n".format(yes_no(self.enableDeformation))
        repr_text += "Zoom/Crop\t\t {}\n".format(yes_no(self.enableCropZoom))
        repr_text += "Flips\t\t\t {}\n".format(yes_no(self.enableFlips))

        if self.enableTranslation:
            repr_text += "\n--Translation:\n"
            for i, x in enumerate(self.translationRanges):
                repr_text += " x{}: no translation\n".format(
                    i) if x == 0.0 else " x{}: translation range [{},{}]\n".format(i, -x, x)

        if self.enableRotation:
            repr_text += "\n--Rotation:\n"
            for i, x in enumerate(self.rotationAngleRanges):
                repr_text += " x{}: no rotation\n".format(
                    i) if x == 0.0 else " x{}: rotation range [{},{}]\n".format(i, -x, x)

        if self.enableDeformation:
            repr_text += "\n--Deformation:\n Number ControlPoints: {}\n Sigma ranges: {}".format(
                self.deformationControlPoints, self.deformationSigma)

        if self.enableCropZoom:
            repr_text += "\n--Crop/Zoom:\n"
            for i, x in enumerate(self.cropZoomLevels):
                repr_text += " x{}: no crop/zoom\n".format(i) if x[0] == x[1] and x[
                                                                                      0] == 1.0 else " x{}: crop/zoom range [{},{}]\n".format(
                    i, *x)
            pass

        if self.enableFlips:
            repr_text += "\n--Flips:\n"
            for i, (axis, prob) in enumerate(zip(self.FlipAxes, self.FlipProp)):
                prop = "no flip" if not axis else "may flip with p = {}".format(prob)
                repr_text += " x{}: {}\n".format((i + 1), prop)
        return repr_text

def printImages(i1, i2):
    from matplotlib import pyplot as plt
    x, y, z = i1.GetSize()
    i_slice = int(z / 2)
    plt.interactive(False)
    i = np.concatenate(
        (sitk.GetArrayFromImage(i1).astype(dtype=float)[i_slice],
         sitk.GetArrayFromImage(i2).astype(dtype=float)[i_slice]),
        axis=1)
    plt.imshow(i)
    plt.show()
