import numpy as np
import SimpleITK as sitk


# --- DO NOT CHANGE ---
def _get_registration_method(atlas_img, img) -> sitk.ImageRegistrationMethod:
    registration_method = sitk.ImageRegistrationMethod()

    # Similarity metric settings.
    registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    registration_method.SetMetricSamplingStrategy(registration_method.REGULAR)
    registration_method.SetMetricSamplingPercentage(0.2)

    registration_method.SetMetricUseFixedImageGradientFilter(False)
    registration_method.SetMetricUseMovingImageGradientFilter(False)

    registration_method.SetInterpolator(sitk.sitkLinear)

    # Optimizer settings.
    registration_method.SetOptimizerAsGradientDescent(
        learningRate=1.0,
        numberOfIterations=100,
        convergenceMinimumValue=1e-6,
        convergenceWindowSize=10,
    )
    registration_method.SetOptimizerScalesFromPhysicalShift()

    # Setup for the multi-resolution framework.
    registration_method.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])
    registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[2, 1, 0])
    registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    # Set initial transform
    initial_transform = sitk.CenteredTransformInitializer(
        atlas_img,
        img,
        sitk.Euler3DTransform(),
        sitk.CenteredTransformInitializerFilter.GEOMETRY,
    )
    registration_method.SetInitialTransform(initial_transform, inPlace=False)
    return registration_method
# --- DO NOT CHANGE ---


def load_image(img_path, is_label_img):
    """
    LOAD_IMAGE:
    # todo: load the image from the image path with the SimpleITK interface (hint: 'ReadImage')
    # todo: if 'is_label_img' is True use argument outputPixelType=sitk.sitkUInt8,
    #  else use outputPixelType=sitk.sitkFloat32
    """
    if is_label_img:
        pixel_type = sitk.sitkUInt8  # todo: modify here
    else:
        pixel_type = sitk.sitkFloat32
    img = sitk.ReadImage(img_path, pixel_type)  # todo: modify here

    return img


def to_numpy_array(img):
    """
    TO_NUMPY_ARRAY:
    # todo: transform the SimpleITK image to a numpy ndarray (hint: 'GetArrayFromImage')
    """
    np_img = sitk.GetArrayFromImage(img)  # todo: modify here

    return np_img


def to_sitk_image(np_image, reference_img):
    """
    TO_SITK_IMAGE:
    # todo: transform the numpy ndarray to a SimpleITK image (hint: 'GetImageFromArray')
    # todo: do not forget to copy meta-information (e.g., spacing, origin, etc.) from the reference image
    #  (hint: 'CopyInformation')! (otherwise defaults are set)
    """

    img = sitk.GetImageFromArray(np_image)  # todo: modify here
    img.CopyInformation(reference_img)

    return img


def preprocess_rescale_numpy(np_img, new_min_val, new_max_val):
    """
    PREPROCESS_RESCALE_NUMPY:
    # todo: rescale the intensities of the np_img to the range [new_min_val, new_max_val].
    # Use numpy arithmetics only.
    """
    max_val = np_img.max()
    min_val = np_img.min()

    normalized_np_img = (np_img - min_val)/(max_val - min_val)  # todo: modify here
    rescaled_np_img = normalized_np_img * (new_max_val - new_min_val) + new_min_val

    return rescaled_np_img


def preprocess_rescale_sitk(img, new_min_val, new_max_val):
    """
    PREPROCESS_RESCALE_SITK:
    # todo: rescale the intensities of the img to the range [new_min_val, new_max_val]
    # (hint: RescaleIntensity)
    """
    rescaled_img = sitk.RescaleIntensity(img, new_min_val, new_max_val)   # todo: modify here

    return rescaled_img


def register_images(img, label_img, atlas_img):
    """
    REGISTER_IMAGES:
    # todo: execute the registration_method to the img (hint: fixed=atlas_img, moving=img)
    # the registration returns the transformation of the moving image (parameter img) to the space of
    # the atlas image (atlas_img)
    """
    registration_method = _get_registration_method(atlas_img, img)
    transform = registration_method.Execute(atlas_img, img)

    registered_img = sitk.Resample(
        img,
        referenceImage=atlas_img,
        transform=transform,
        interpolator=sitk.sitkLinear,
        defaultPixelValue=0.0,
        outputPixelType=img.GetPixelIDValue()
    )

    registered_label = sitk.Resample(
        label_img,
        referenceImage=atlas_img,
        transform=transform,
        interpolator=sitk.sitkNearestNeighbor,
        defaultPixelValue=0.0,
        outputPixelType=label_img.GetPixelIDValue()
    )

    return registered_img, registered_label


def extract_feature_median(img):
    """
    EXTRACT_FEATURE_MEDIAN:
    # todo: apply median filter to image (hint: 'Median')
    """
    median_img = sitk.Median(img)
    return median_img


def postprocess_largest_component(label_img):
    """
    POSTPROCESS_LARGEST_COMPONENT:
    # todo: get the connected components from the label_img (hint: 'ConnectedComponent')
    """
    connected_components = sitk.ConnectedComponent(label_img)
    relabeled_components = sitk.RelabelComponent(connected_components)
    largest_component = sitk.GetArrayFromImage(relabeled_components) == 1
    return largest_component
