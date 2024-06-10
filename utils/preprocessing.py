import cv2
from keras.utils import to_categorical
import numpy as np
def crop_images(images, crop_size, image_size):
    """
    Crop images by taking from both the top and the bottom.

    Args:
    - images: A numpy array containing the images. Shape: (num_images, height, width, channels)
    - crop_height: The height to which the images will be cropped.

    Returns:
    - cropped_images: A numpy array containing the cropped images. Shape: (num_images, crop_height, width, channels)
    """
    cropped_images=[]
    for image in images:
      start_row = (image_size - crop_size) // 2
      start_col = (image_size - crop_size) // 2
      left=start_row
      right=start_row+crop_size
      bottom=start_col
      top=start_col+crop_size

    # Crop the image
      cropped_image = image[left:right, bottom:top]

    # Crop the images
      cropped_images.append(cropped_image)

    return cropped_images




def resize_images(images, target_size):
    """
    Resize a batch of images to a target size.

    Parameters:
        images (list): List of input images (each image is a NumPy array).
        target_size (tuple): Target size in the format (width, height).

    Returns:
        resized_images (list): List of resized images.
    """
    resized_images = []
    for img in images:
        resized_img = cv2.resize(img, target_size)
        resized_images.append(resized_img)
    return resized_images

def preprocessing_pipeline(data, mask=False, augmented=False):
  #data=resize_images(data,(256,256))
  data=crop_images(data,256, 512)
  data=np.expand_dims(data, axis=-1)
  data=np.array(data)
  if mask:
        data = to_categorical(data, num_classes=3)


  return data