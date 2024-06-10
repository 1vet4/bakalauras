from keras.callbacks import Callback
from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger

class AppendHistory(Callback):
    def __init__(self):
        super(AppendHistory, self).__init__()
        self.history = {}

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        for key, value in logs.items():
            self.history.setdefault(key, []).append(value)

def callbacks_list(filename):
    filepath=filename+"-{epoch:02d}.h5" #File name includes epoch and validation accuracy.
    #Use Mode = max for accuracy and min for loss.
    checkpoint = ModelCheckpoint(filepath,
                             save_weights_only=True,
                             save_freq=5
                            )


    early_stop = EarlyStopping(monitor='val_loss',
                           patience=8,
                           verbose=1)

    #CSVLogger logs epoch, acc, loss, val_acc, val_loss
    log_csv = CSVLogger('antra_unet_dice_su_svoriu_mazesnislaaaar.csv', separator=',', append=True)
    append_history=AppendHistory()
    callbacks_list = [checkpoint, early_stop, log_csv, append_history]
    return callbacks_list
