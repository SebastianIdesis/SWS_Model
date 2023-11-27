#!/usr/bin/python
import numpy as np
import scipy.io
import csv
from scipy.stats import pearsonr
from matplotlib import pyplot as plt
import glob
import os
import sys
import nibabel as nib

## atlas
atlas = nib.load('Schaefer235.nii')
#atlas = nib.load('Schaefer2018_200Parcels_7Networks_order_plus_subcort.nii')
atlasdata=atlas.get_fdata()
sa=np.shape(atlasdata)
atlasmap = atlas.affine

# coordinares of atlas file voxels
atlascoords = np.zeros(np.append(np.shape(atlasdata),3))

for c in range(atlasdata.size):
	i,j,k = np.unravel_index(c, np.shape(atlasdata))
	atlascoords[i,j,k,:]=(atlasmap[:3,:3]).dot([i, j, k])+atlasmap[:3, 3]

# coordinates of atlas ROI centers
nroi = np.max(atlasdata).astype(int) #Amount of ROIs

atlasdata = atlasdata.reshape(-1) # Takes away one dimension
atlascoords = np.reshape(atlascoords,((sa[0]*sa[1]*sa[2],3)))

roicoords = np.zeros((nroi,3))
for i in range(nroi):
	roivox=np.where(atlasdata==i+1)[0] #Roivox is an array inside a tuple. Indexes of that ROI
	roicoords[i,:]=np.mean(atlascoords[roivox,:],axis=0) #Mean calculates the centroid of the ROI


# lesion data
lesion_dir = glob.glob('nifti_lesions/')[0] #Find folder
lesion_sbjs = os.listdir(lesion_dir) #Check names of files in that folder


# distance of each atlas ROI from lesion
lesion_distance = np.zeros((196,nroi))


# voxel coordinates of lesion file (without bounding box)
img=nib.load(lesion_dir + lesion_sbjs[0]) # Generic pre-processing with whichever subject
data=img.get_fdata()
data = np.squeeze(data) #As in matlab
M = img.affine[:3, :3]

coords0 = np.zeros(np.append(np.shape(data),3))

for c in range(data.size):
	i,j,k = np.unravel_index(c, np.shape(data))
	coords0[i,j,k,:]=M.dot([i, j, k]) 


# find lesion distance for all subjects

for sbj in lesion_sbjs:
	if sbj.endswith('_lesion_ifhfix.nii.gz'):
		sbjshort= sbj.strip('_lesion_ifhfix.nii.gz')
	else:
		sbjshort=sbj.strip('_lesion.nii.gz')


	n=int(sbjshort[4:7])

	print(lesion_dir + sbj)

	# lesion file
	img=nib.load(lesion_dir + sbj)

	data=img.get_fdata()
	data = np.squeeze(data)

	lesioned = np.where(data >0)

	# voxel coordinates of lesion file (with bounding box)
	abc = img.affine[:3, 3]
	coords = np.zeros(np.append(np.shape(data),3))
	coords = coords0 + abc

	# coordinates of lesion voxels
	lesioned_coords = coords[lesioned]

	# compute minimum distance of ROI center from lesioned voxel
	for i in range(nroi):
		dmin=1000
		for l in range(np.shape(lesioned_coords)[0]):
			d=np.sqrt(np.sum(np.square(roicoords[i,:]-lesioned_coords[l,:])))
			if(d<dmin):
				dmin=d

		lesion_distance[n-1,i]=dmin

np.savetxt("lesion_distance_peri.txt",lesion_distance,fmt="%f")


