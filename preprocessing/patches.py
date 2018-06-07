import tensorflow as tf
import numpy as np
from BraTS.modalities import mri_shape, seg_shape
from segmentation.params import Params

#Randomly choose a tumor voxel and return its coordinates
def get_tumor_index_single_class(seg):
	tumor_indices = tf.where(tf.greater(seg, 0))	#returns indices of tumor voxels 
	tumor_index = convert_to_tensor(np.random.choice(tf.shape(tumor_indices).as_list()[0])) 	#may need to change this dim
	print(tf.shape(tumor_indices).as_list()[0])
	return tumor_indices[tumor_index, :]

''' Returns a randomly chosen voxel that will be the upper left corner of the patch, such that when the patch is 
	created it contains the tumor_index specified'''
def get_patch_index(image_shape, patch_shape, tumor_index):
	
	#get iamge and patch dimensions as tensors
	#image_dim_tensor = convert_to_tensorflow(image_shape)
	p#atch_dim_tensor = convert_to_tensorflow(patch_shape)

	#get tumor indices
	t2 = tumor_index.as_list()[1]
	t3 = tumor_index.as_list()[2]
	t4 = tumor_index.as_list()[3]

	t2_max = np.max(t2 + patch_shape[0], image_shape[1])
	t2_min = np.min(t2 - patch_shape[0], zero)
	t3_max = np.max(t3 + patch_shape[1], image_shape[2])
	t3_min = np.min(t3 -  patch_shape[1], zero)
	t4_max = np.max(t4 + patch_shape[2], image_shape[3])
	t4_min = np.min(t4 - patch_shape[2], zero)

	x = np.random.choice(tumor_index.as_list()[1], t2_max.as_list()[0])
	y = np.random.choice(tumor_index.as_list()[2], t3_max.as_list()[0])
	z = np.random.choice(tumor_index.as_list()[3], t4_max.as_list()[0])

	if (x - patch_shape[0] < 0):
		x = 0
	elif (x + patch_shape[0] >= image_shape[1]):
		x = image_shape[1] - patch_shape[0]

	if (y - patch_shape[1] < 0):
		y = 1
	elif (y + patch_shape[1] >= image_shape[2]):
		y = image_shape[2] - patch_shape[1]
	
	if (z - patch_shape[2] < 0):
		z = 0
	elif (z + patch_shape[2] >= image_shape[3]):
		z = image_shape[3] - patch_shape[2]

	return [x, y, z]

#returns the set of patch indices for an image input
def get_patch_indices(patches_per_image, image_shape, patch_shape, seg):
	patch_indices = np.zeros(patches_per_image)

	for i in range(patches_per_image):
		tumor_index = get_tumor_index_single_class(seg)
		patch_indices[i] = get_patch_index(image_shape, patch_shape, tumor_index)

	return patch_indices

#Returns mri and segmentation patches of size patch_shape. Patch's location (upper corner) is specified by tumor_index
def get_patch(image, patch_shape, image_shape, patch_index):
	start_index = [0]
	end_index = [4]

	for i in range(len(patch_index)):
		start_index.append(patch_index[i])
		end_index.append(patch_index[i])

	return tf.slice(image, start_index, end_index)

# Returns a single mri and ground truth segmentation as a set of patches
def get_patches(image, seg, patch_indices, patch_shape = Params.patch_shape, image_shape = mri_shape, patches_per_image = Params.patches_per_image):
	
	_mri = get_patch(image, patch_shape, image_shape, patch_indices[0])
	_seg = get_patch(seg, patch_shape, image_shape, indices[0])

	for i in range(patches_per_image-1):
		_mri = tf.concat(_mri, get_patch(image, patch_shape, image_shape, indices[i]))
		_seg = tf.concat(_seg, get_patch(seg, patch_shape, image_shape, indices[i]))
	return _mri, seg








