from tensorflow import keras
import tensorflow as tf

from Pills import DataLoader
from image_segmentation import Utils
from .PillNet import PillNet

BATCH_SIZE = 32
INPUT_SHAPE = (128, 128, 3)
ANNOTATION_PATH = "/workspaces/dev/datasets/pills/data/annotations.json"
IMAGE_DIR = "/workspaces/dev/datasets/pills"
MODEL_PATH = "/workspaces/dev/models/R-CNN/checkpoints/pillnet.keras"

def main():
  data_loader = DataLoader(BATCH_SIZE, IMAGE_DIR, ANNOTATION_PATH)
  
  model = PillNet()
  model.build(BATCH_SIZE, INPUT_SHAPE)
  model.compile(['accuracy', Utils.iou_metric, Utils.dice_metric, Utils.pixel_accuracy])

  checkpoint_cb = keras.callbacks.ModelCheckpoint(
    filepath=MODEL_PATH,
    monitor="val_loss",
    save_best_only=True,
    save_weights_only=False,
    verbose=1
  )

  early_stopping_cb = keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=10,
    restore_best_weights=True,
  )

  model.fit(
    data_loader.train_dataset,
    validation_data=data_loader.val_dataset,
    epochs=100,
    callbacks=[checkpoint_cb, early_stopping_cb],
    workers = 4,
    use_multiprocessing = True
  )


