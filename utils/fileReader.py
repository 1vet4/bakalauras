import os
import nibabel as nib
import numpy as np

def make_files_list(folder_path):

#Function makes a list of files in a folder

  file_list = []

  # Get list of files in the folder
  files = os.listdir(folder_path)

  # Iterate over each filename and append it to the list
  for file in files:
      file_list.append(os.path.join(folder_path, file))


  return file_list



def concatenate_slices(nifti_file_list):

    #Function concatenates all of the slices to one volume

    # Initialize an empty list to store slices
    all_slices = []

    # Loop through each NIfTI file
    for nifti_file in nifti_file_list:

        # Load NIfTI file
        nifti_img = nib.load(nifti_file).get_fdata()

        # Append slices to the list
        all_slices.extend([nifti_img[..., i] for i in range(nifti_img.shape[2])])

    # Stack all slices along the first axis to create the concatenated volume
    concatenated_volume = np.stack(all_slices, axis=0)

    print('Sllices concatenated.')
    return concatenated_volume

