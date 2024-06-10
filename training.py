import os
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import random
import tensorflow as tf
import cv2
import keras
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.metrics import MeanIoU


from utils.fileReader import make_files_list, concatenate_slices
from utils.preprocessing import preprocessing_pipeline
from utils.augmentation import augment_images
from utils.callbacks import callbacks_list
from metrics.metrics import (dice_coef_loss, IoU_per_class, dice_per_class)
from models.unet import Unet
from models.resunet import ResUNet

CT_slice_train_path='/content/drive/MyDrive/Bakalauras/Filtruoti_duomenys/treniravimas/paveiksleliai'
mask_train_path='/content/drive/MyDrive/Bakalauras/Filtruoti_duomenys/treniravimas/kaukes'

CT_slice_validation_path='/content/drive/MyDrive/Bakalauras/Filtruoti_duomenys/validavimas/paveiksleliai'
mask_validation_path='/content/drive/MyDrive/Bakalauras/Filtruoti_duomenys/validavimas/kaukes'

CT_slice_test_path='/content/drive/MyDrive/Bakalauras/Filtruoti_duomenys/testavimas/paveiksleliai'
mask_test_path='/content/drive/MyDrive/Bakalauras/Filtruoti_duomenys/testavimas/kaukes'

CT_list_train=make_files_list(CT_slice_train_path)
mask_list_train=make_files_list(mask_train_path)

CT_list_validation=make_files_list(CT_slice_validation_path)
mask_list_validation=make_files_list(mask_validation_path)

CT_list_test=make_files_list(CT_slice_test_path)
mask_list_test=make_files_list(mask_test_path)

CT_list_train_combined = concatenate_slices(CT_list_train)
CT_list_test_combined = concatenate_slices(CT_list_test)
CT_list_val_combined = concatenate_slices(CT_list_validation)
mask_list_train_combined = concatenate_slices(mask_list_train)
mask_list_test_combined = concatenate_slices(mask_list_test)
mask_list_val_combined = concatenate_slices(mask_list_validation)

np.random.seed(42)
num_images_to_augment = 300
random_indices = np.random.choice(len(CT_list_train_combined), size=num_images_to_augment, replace=False)
images_to_augment = CT_list_train_combined[random_indices]
masks_to_augment = mask_list_train_combined[random_indices]

augmented_images, augmented_masks = augment_images(images_to_augment, masks_to_augment, num_images_to_augment)

CT_list_train_combined=np.concatenate((CT_list_train_combined, augmented_images),axis=0)
mask_list_train_combined=np.concatenate((mask_list_train_combined, augmented_masks), axis=0)

print('CT_train ', CT_list_train_combined.shape)
print('CT_val ', CT_list_val_combined.shape)
print('CT_test ', CT_list_test_combined.shape)
print('Mask_trai ', mask_list_train_combined.shape)
print('Mask_val ', mask_list_val_combined.shape)
print('Mask_test ', mask_list_test_combined.shape)

X_train=preprocessing_pipeline(CT_list_train_combined)
X_val=preprocessing_pipeline(CT_list_val_combined)
X_test=preprocessing_pipeline(CT_list_test_combined)
y_train=preprocessing_pipeline(mask_list_train_combined, mask=True)
y_val=preprocessing_pipeline(mask_list_val_combined, mask=True)
y_test=preprocessing_pipeline(mask_list_test_combined, mask=True)

print("X_train shape: ", X_train.shape)
print("y_train shape: ", y_train.shape)
print("X_val shape: ", X_val.shape)
print("y_val shape: ", y_val.shape)
print("X_test shape: ", X_test.shape)
print("y_test shape: ", y_test.shape)

from sklearn.utils import class_weight
#class_weights = class_weight.compute_class_weight(class_weight='balanced',
                                                 #classes=np.unique(mask_list_train_combined),
                                                 #y=mask_list_train_combined.flatten())

class_weights = {0: 0.33,
                 1: 424.88,
                 2: 3929.37}
print("Class weights are...:", class_weights)



train_datagen = ImageDataGenerator(

)

train_generator = train_datagen.flow(
    X_train,
    y_train,
    batch_size = 6)

val_generator=train_datagen.flow(
    X_val,
    y_val,
    batch_size = 6)

callbacks_list = callbacks_list('')

unet = Unet((256,256,1), dropout= 0.2)

resunet = ResUNet()

iou_metric = MeanIoU(num_classes=3)


"""# Training"""

#optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
unet.compile(optimizer='adam', loss=dice_coef_loss, metrics=[iou_metric, 'accuracy'])

history=unet.fit(train_generator,
                  validation_data=val_generator,
                  epochs=5,
                  class_weight=class_weights,
                  callbacks = callbacks_list
                  )

"""# Evaluating"""

pred = unet.predict(X_test)

if pred.shape[-1] == 1:
    pred = np.squeeze(pred, axis=-1)

# Get the index of the maximum value along the channel axis
argmax_indices = np.argmax(pred, axis=-1)

# Create a new array with zeros
post_processed_predictions = np.zeros_like(pred)

# Set the maximum value for each class to 1, keeping zeros elsewhere
for i in range(pred.shape[-1]):
    post_processed_predictions[argmax_indices == i, i] = 1

y_true_reshaped = y_test.reshape(-1, 3)
y_pred_reshaped = post_processed_predictions.reshape(-1, 3)
y_true_labels = np.argmax(y_true_reshaped, axis=1)
y_pred_labels = np.argmax(y_pred_reshaped, axis=1)

IoU_per_class(post_processed_predictions, y_test)
sum(IoU_per_class(post_processed_predictions, y_test))/3

dice_per_class(post_processed_predictions, y_test)
sum(dice_per_class(post_processed_predictions, y_test))/3



#115 245 360
# Plot original images
plt.figure(figsize=(15, 5))
index=245
plt.subplot(1,3,1)
plt.imshow(X_test[index], cmap='gray')
plt.title('Originalus vaizdas')
plt.axis('off')


plt.subplot(1, 3,2)
plt.imshow(y_test[index], cmap='gray')
plt.title('Segmentavimo kaukė')
plt.axis('off')

plt.subplot(1, 3,3)
plt.imshow(post_processed_predictions[index], cmap='gray')
plt.title('Tinklo išvestis')
plt.axis('off')
plt.tight_layout()
plt.show()