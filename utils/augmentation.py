from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np

def augment_images(images_to_augment, masks_to_augment, num_images_to_augment):

    augmentation_generator = ImageDataGenerator(
    rotation_range=20,       # Random rotation up to 20 degrees
    width_shift_range=0.15,   # Random horizontal shift
    height_shift_range=0.15,  # Random vertical shift
    shear_range=0.1,         # Shear intensity
    horizontal_flip=True,    # Random horizontal flip
    fill_mode='nearest'
    )


    images_to_augment = np.expand_dims(images_to_augment, axis=-1)
    masks_to_augment = np.expand_dims(masks_to_augment, axis=-1)

    augmented_images = []
    augmented_masks = []
    seed=42

    for i in range(num_images_to_augment):
        augmented_img = augmentation_generator.random_transform(images_to_augment[i], seed=seed)
        augmented_mask = augmentation_generator.random_transform(masks_to_augment[i], seed=seed)
        augmented_images.append(augmented_img)
        augmented_masks.append(augmented_mask)


    augmented_images=np.array(augmented_images)
    augmented_images=augmented_images[:,:,:,-1]
    augmented_masks=np.array(augmented_masks)
    augmented_masks=augmented_masks[:,:,:,-1]

    return augmented_images, augmented_masks