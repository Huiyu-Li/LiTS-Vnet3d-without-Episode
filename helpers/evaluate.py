#!/usr/bin/env python
from __future__ import print_function
import os
import SimpleITK as sitk
import numpy as np
from scipy.ndimage.measurements import label as label_connected_components
import gc
from helpers.calc_metric import (dice,
                                 detect_lesions,
                                 compute_segmentation_scores,
                                 compute_tumor_burden,
                                 LARGE)
from helpers.utils import time_elapsed

# Check input directories.
# csv_path = os.path.join('/home01/weileyi/jinqiangguo/jqg/py3EnvRoad/lung-segmentation-3d/Demo/',sys.argv[1])
# print(csv_path)

# Create output directory.
output_dir = './output/'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Segmentation metrics and their default values for when there are no detected
# objects on which to evaluate them.
#
# Surface distance (and volume difference) metrics between two masks are
# meaningless when any one of the masks is empty. Assign maximum (infinite)
# penalty. The average score for these metrics, over all objects, will thus
# also not be finite as it also loses meaning.
segmentation_metrics = {'dice': 0,
                        'jaccard': 0,
                        'voe': 1,
                        'rvd': LARGE,
                        'assd': LARGE,
                        'rmsd': LARGE,
                        'msd': LARGE}

# Initialize results dictionaries
lesion_detection_stats = {0:   {'TP': 0, 'FP': 0, 'FN': 0},
                          0.5: {'TP': 0, 'FP': 0, 'FN': 0}}
lesion_segmentation_scores = {}
liver_segmentation_scores = {}
dice_per_case = {'lesion': [], 'liver': []}
dice_global_x = {'lesion': {'I': 0, 'S': 0},
                 'liver':  {'I': 0, 'S': 0}} # 2*I/S
tumor_burden_list = []
t = time_elapsed()

# Load reference and submission volumes with Nibabel.
ref = sitk.ReadImage('./segmentation-0.nii')
sub = sitk.ReadImage('./segmentation-0.nii')
# Get the current voxel spacing.
spacing = ref.GetSpacing()
print(spacing)#(0.703125, 0.703125, 5.0)
# Get Numpy data and compress to int8.
reference_volume = sitk.GetArrayFromImage(ref).astype(np.int8)
submission_volume = sitk.GetArrayFromImage(sub).astype(np.int8)
# reference_volume = reference_volume[:,:,59]#get a slice
# submission_volume = submission_volume[:,:,59]#get a slice

# Ensure that the shapes of the masks match.
if submission_volume.shape!=reference_volume.shape:
    raise AttributeError("Shapes do not match! Prediction mask {}, "
                         "ground truth mask {}"
                         "".format(submission_volume.shape,
                                   reference_volume.shape))
print("Done loading files ({:.2f} seconds)".format(t()))

# Create lesion masks with labeled connected components.
# (Assuming there is always exactly one liver - one connected comp.)
pred_mask_lesion, num_predicted = label_connected_components( \
                                     submission_volume==2, output=np.int16)#default structuring is 2-D array, so the input demension is?
true_mask_lesion, num_reference = label_connected_components( \
                                     reference_volume==2, output=np.int16)
# import matplotlib.pyplot as plt
# plt.subplot(2, 2, 1)
# plt.imshow(pred_mask_lesion)
# plt.show()

pred_mask_liver = submission_volume
pred_mask_liver[submission_volume>=1]=1
# plt.subplot(2, 2, 2)
# plt.imshow(pred_mask_liver)
# plt.show()
true_mask_liver = reference_volume
true_mask_liver[reference_volume>=1]=1
liver_prediction_exists = np.any(submission_volume==1)
print("Done finding connected components ({:.2f} seconds)".format(t()))

# Identify detected lesions.
# Retain detected_mask_lesion for overlap > 0.5
for overlap in [0, 0.5]:
    detected_mask_lesion, mod_ref_mask, num_detected = detect_lesions( \
                                          prediction_mask=pred_mask_lesion,
                                          reference_mask=true_mask_lesion,
                                          min_overlap=overlap)
    # Count true/false positive and false negative detections.
    lesion_detection_stats[overlap]['TP']+=num_detected
    lesion_detection_stats[overlap]['FP']+=num_predicted-num_detected
    lesion_detection_stats[overlap]['FN']+=num_reference-num_detected
print("Done identifying detected lesions ({:.2f} seconds)".format(t()))

# Compute lesions segmentation scores.
if num_detected>0:
    lesion_scores = compute_segmentation_scores( \
                                      prediction_mask=detected_mask_lesion,
                                      reference_mask=mod_ref_mask,
                                      voxel_spacing=spacing)
    for metric in segmentation_metrics:
        if metric not in lesion_segmentation_scores:
            lesion_segmentation_scores[metric] = []
        lesion_segmentation_scores[metric].extend(lesion_scores[metric])
    print("Done computing lesion scores ({:.2f} seconds)".format(t()))
else:
    print("No lesions detected, skipping lesion score evaluation")

# Compute liver segmentation scores.
if liver_prediction_exists:
    liver_scores = compute_segmentation_scores( \
                                      prediction_mask=pred_mask_liver,
                                      reference_mask=true_mask_liver,
                                      voxel_spacing=spacing)
    for metric in segmentation_metrics:
        if metric not in liver_segmentation_scores:
            liver_segmentation_scores[metric] = []
        liver_segmentation_scores[metric].extend(liver_scores[metric])
    print("Done computing liver scores ({:.2f} seconds)".format(t()))
else:
    # No liver label. Record default score values (zeros, inf).
    # NOTE: This will make some metrics evaluate to inf over the entire
    # dataset.
    for metric in segmentation_metrics:
        if metric not in liver_segmentation_scores:
            liver_segmentation_scores[metric] = []
        liver_segmentation_scores[metric].append(\
                                              segmentation_metrics[metric])
    print("No liver label provided, skipping liver score evaluation")

# Compute per-case (per patient volume) dice.
if not np.any(pred_mask_lesion) and not np.any(true_mask_lesion):
    dice_per_case['lesion'].append(1.)
else:
    dice_per_case['lesion'].append(dice(pred_mask_lesion,
                                        true_mask_lesion))
if liver_prediction_exists:
    dice_per_case['liver'].append(dice(pred_mask_liver,
                                       true_mask_liver))
else:
    dice_per_case['liver'].append(0)

# Accumulate stats for global (dataset-wide) dice score.
dice_global_x['lesion']['I'] += np.count_nonzero( \
    np.logical_and(pred_mask_lesion, true_mask_lesion))
dice_global_x['lesion']['S'] += np.count_nonzero(pred_mask_lesion) + \
                                np.count_nonzero(true_mask_lesion)
if liver_prediction_exists:
    dice_global_x['liver']['I'] += np.count_nonzero( \
        np.logical_and(pred_mask_liver, true_mask_liver))
    dice_global_x['liver']['S'] += np.count_nonzero(pred_mask_liver) + \
                                   np.count_nonzero(true_mask_liver)
else:
    # NOTE: This value should never be zero.
    dice_global_x['liver']['S'] += np.count_nonzero(true_mask_liver)


print("Done computing additional dice scores ({:.2f} seconds)"
      "".format(t()))

# Compute tumor burden.
tumor_burden = compute_tumor_burden(prediction_mask=submission_volume,
                                    reference_mask=reference_volume)
tumor_burden_list.append(tumor_burden)
print("Done computing tumor burden diff ({:.2f} seconds)".format(t()))

print("Done processing volume (total time: {:.2f} seconds)"
      "".format(t.total_elapsed()))
gc.collect()

#####################################################################
# Compute lesion detection metrics.
_det = {}
for overlap in [0, 0.5]:
    TP = lesion_detection_stats[overlap]['TP']
    FP = lesion_detection_stats[overlap]['FP']
    FN = lesion_detection_stats[overlap]['FN']
    precision = float(TP)/(TP+FP) if TP+FP else 0
    recall = float(TP)/(TP+FN) if TP+FN else 0
    _det[overlap] = {'p': precision, 'r': recall}
lesion_detection_metrics = {'precision': _det[0.5]['p'],
                            'recall': _det[0.5]['r'],
                            'precision_greater_zero': _det[0]['p'],
                            'recall_greater_zero': _det[0]['r']}

# Compute lesion segmentation metrics.
lesion_segmentation_metrics = {}
for m in lesion_segmentation_scores:
    lesion_segmentation_metrics[m] = np.mean(lesion_segmentation_scores[m])
if len(lesion_segmentation_scores)==0:
    # Nothing detected - set default values.
    lesion_segmentation_metrics.update(segmentation_metrics)
lesion_segmentation_metrics['dice_per_case'] = np.mean(dice_per_case['lesion'])
dice_global = 2.*dice_global_x['lesion']['I']/dice_global_x['lesion']['S']
lesion_segmentation_metrics['dice_global'] = dice_global

# Compute liver segmentation metrics.
liver_segmentation_metrics = {}
for m in liver_segmentation_scores:
    liver_segmentation_metrics[m] = np.mean(liver_segmentation_scores[m])
if len(liver_segmentation_scores)==0:
    # Nothing detected - set default values.
    liver_segmentation_metrics.update(segmentation_metrics)
liver_segmentation_metrics['dice_per_case'] = np.mean(dice_per_case['liver'])
dice_global = 2.*dice_global_x['liver']['I']/dice_global_x['liver']['S']
liver_segmentation_metrics['dice_global'] = dice_global

# Compute tumor burden.
tumor_burden_rmse = np.sqrt(np.mean(np.square(tumor_burden_list)))
tumor_burden_max = np.max(tumor_burden_list)


# Print results to stdout.
print("####Computed leision detection metrics:")
for metric, value in lesion_detection_metrics.items():
    print("{}: {:.3f}".format(metric, float(value)))
print("####Computed leision segmentation metrics (for detected lesions):")
for metric, value in lesion_segmentation_metrics.items():
    print("{}: {:.3f}".format(metric, float(value)))
print("####Computed liver segmentation metrics:")
for metric, value in liver_segmentation_metrics.items():
    print("{}: {:.3f}".format(metric, float(value)))
print("####Computed tumor burden: \n"
    "rmse: {:.3f}\nmax: {:.3f}".format(tumor_burden_rmse, tumor_burden_max))

# Write metrics to file.
output_filename = os.path.join(output_dir, 'scores.txt')
output_file = open(output_filename, 'w')
for metric, value in lesion_detection_metrics.items():
    output_file.write("lesion_{}: {:.3f}\n".format(metric, float(value)))
for metric, value in lesion_segmentation_metrics.items():
    output_file.write("lesion_{}: {:.3f}\n".format(metric, float(value)))
for metric, value in liver_segmentation_metrics.items():
    output_file.write("liver_{}: {:.3f}\n".format(metric, float(value)))

#Tumorburden
output_file.write("RMSE_Tumorburden: {:.3f}\n".format(tumor_burden_rmse))
output_file.write("MAXERROR_Tumorburden: {:.3f}\n".format(tumor_burden_max))

output_file.close()
