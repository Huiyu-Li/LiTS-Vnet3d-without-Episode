#!/usr/bin/env python
from __future__ import print_function
import SimpleITK as sitk
import numpy as np
from scipy.ndimage.measurements import label as label_connected_components
from medpy import metric
from helpers.utils import time_elapsed
t = time_elapsed()

# Load reference and submission volumes with Nibabel.
ref = sitk.ReadImage('./segmentation-0.nii')
sub = sitk.ReadImage('./segmentation-0.nii')
# Get Numpy data and compress to int8.
reference_volume = sitk.GetArrayFromImage(ref)#.astype(np.int8)
lreference_volume = reference_volume.view((1,75,512,512))
submission_volume = sitk.GetArrayFromImage(sub).astype(np.int8)
# Ensure that the shapes of the masks match.
if submission_volume.shape!=reference_volume.shape:
    raise AttributeError("Shapes do not match! Prediction mask {}, "
                         "ground truth mask {}"
                         "".format(submission_volume.shape,
                                   reference_volume.shape))

# Create  masks with labeled connected components.
# (Assuming there is always exactly one liver - one connected comp.)
pred_mask_liver = submission_volume;pred_mask_liver[submission_volume>=1]=1
pred_mask_lesion, num_predicted = label_connected_components(submission_volume==2, output=np.int16)#default structuring is 2-D array, so the input demension is?

true_mask_liver = reference_volume;true_mask_liver[reference_volume>=1]=1
true_mask_lesion, num_reference = label_connected_components(reference_volume==2, output=np.int16)
liver_prediction_exists = np.any(submission_volume==1)
# Compute per-case (per patient volume) dice.
if not np.any(pred_mask_lesion) and not np.any(true_mask_lesion):
    tumor_dice_per_case = 1.
else:
    tumor_dice_per_case = metric.dc(pred_mask_lesion, true_mask_lesion)
if liver_prediction_exists:
    liver_dice_per_case = metric.dc(pred_mask_liver, true_mask_liver)
else:
    liver_dice_per_case = 0

print(tumor_dice_per_case)
print(liver_dice_per_case)



