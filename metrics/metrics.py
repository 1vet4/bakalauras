import numpy as np
from keras import backend as K


def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    dice = (2. * intersection + 1e-5) / (K.sum(y_true_f) + K.sum(y_pred_f) + 1e-5)
    return dice


def dice_coef_loss(y_true, y_pred):
    return 1 - dice_coef_multiclass(y_true, y_pred)

def dice_coef_multiclass(y_true, y_pred):
    dice = 0
    for index in range(3):
        dice += dice_coef(y_true[:,:,:,index], y_pred[:,:,:,index])
    return dice/3

def dice_per_class(y_true, y_pred):

     dice_per_class=[]
     for class_index in range(3):

        y_pred_class=y_pred[..., class_index]
        y_true_class=y_true[..., class_index]

        y_true_f = K.flatten(y_true_class)
        y_pred_f = K.flatten(y_pred_class)
        intersection = K.sum(y_true_f * y_pred_f)
        dice = (2. * intersection) / (K.sum(y_true_f) + K.sum(y_pred_f) + 1e-5)

        dice_per_class.append(dice)

     return dice_per_class


def IoU_per_class(y_pred, y_true):

     iou_per_class=[]
     for class_index in range(3):

        y_pred_class=y_pred[..., class_index]
        y_true_class=y_true[..., class_index]

        intersection = np.sum(np.logical_and(y_pred_class, y_true_class))
        union = np.sum(np.logical_or(y_pred_class, y_true_class))

        IoU=intersection/(union + 1e-6)

        iou_per_class.append(IoU)

     return iou_per_class
