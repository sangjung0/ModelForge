import tensorflow as tf
from tensorflow.keras import layers, Model
from image_segmentation.Utils import *

GRID_SIZE = 4
INPUT_SHAPE = (128, 128, 3)

class PillNet(Model):
  def __init__(self, batch_size:int, num_classes:int, initial_lr = 0.001, **kwargs):
    super(PillNet, self).__init__(**kwargs)

    self.__INPUT_SHAPE = INPUT_SHAPE
    self.__GRID_SIZE = GRID_SIZE
    self.__INITIAL_LR = initial_lr
    
    self.feature_extractor = tf.keras.Sequential([
      layers.DepthwiseConv2D(depth_multiplier= 4, kernel_size = (9, 9), activation='relu', padding="same"), # S x S x 48
      layers.LayerNormalization(axis=-1), # 채널 정규화
      layers.Conv2D(16, (1, 1), activation='relu', padding="same"), # Pointwise Convolution, S x S x 64
      layers.Conv2D(16, (9, 9), activation='relu', padding="same"), # S x S x 64 
      layers.LayerNormalization(), # 정규화
      layers.MaxPooling2D((2, 2), strides = 2, padding="same"), # S/2 x S/2 x 64
      layers.Conv2D(32, (7, 7), activation='relu', padding="same"), # S/2 x S/2 x 64
      layers.Conv2D(32, (9, 9), activation='relu', padding="same"), # S/2 x S/2 x 64
      layers.LayerNormalization(), # 정규화
      layers.MaxPooling2D((2, 2), strides = 2, padding="same"), # S/4 x S/4 x 64
    ], name="FeatureExtractor")

    self.roi_head = tf.keras.Sequential([
      layers.Conv2DTranspose(16, (3, 3), strides=2, activation='gelu', padding="same"), # S/2 x S/2 x 16
      layers.Conv2DTranspose(8, (3, 3), strides=2, activation='gelu', padding="same"), # S x S x 8
      layers.Conv2D(1, (1, 1), activation='sigmoid', padding="same"), # S x S x 1
    ], name="ROIHead")
    
    self.detection_head = tf.keras.Sequential([
      layers.Conv2D(128, (7, 7), activation='relu', padding="same"), # S/4 x S/4 x 128
      layers.Conv2D(256, (5, 5), activation='relu', padding="same"), # S/4 x S/4 x 256
      layers.MaxPooling2D((2, 2), strides=2, padding="same"), # S/8 x S/8 x 256
      layers.LayerNormalization(), # 정규화
      layers.Conv2D(256, (5, 5), strides=2, activation='relu', padding="same"), # S/16 x S/16 x 256
      layers.Conv2D(256, (3, 3), strides=2, activation='relu', padding="same"), # S/32 x S/32 x 128
      layers.Conv2D(4, (1, 1), activation='sigmoid', padding="same"), # S/32 x S/32 x 4 S/32 = g_s
      layers.Reshape((self.__GRID_SIZE * self.__GRID_SIZE, 4)), # g x g x 4
    ], name="DetectionHead")
    
    self.centroid_head = tf.keras.Sequential([
      layers.Conv2D(16, (5, 5), activation='relu', padding="same"), # S/4 x S/4 x 16
      layers.Conv2D(8, (9, 9), activation='relu', padding="same"), # S/4 x S/4 x 8
      layers.GlobalAveragePooling2D(), # 8
      layers.Dense(2, activation='sigmoid'), # 2
    ], name="CentroidHead")

    self.segmentation_head = tf.keras.Sequential([
      layers.Conv2D(16, (5, 5), activation='gelu', padding="same"), # S/4 x S/4 x 32
      layers.Conv2D(16, (9, 9), activation='gelu', padding="same"), # S/4 x S/4 x 32
      layers.LayerNormalization(), # 정규화
      layers.Conv2D(8, (3, 3), activation='gelu', padding="same"), # S/4 x S/4 x 16
      layers.Conv2D(8, (7, 7), activation='gelu', padding="same"), # S/4 x S/4 x 16
      layers.Conv2D(1, (1, 1), activation='sigmoid', padding="same"), # S/4 x S/4 x 1
    ], name="SegmentationHead")

    self.classification_head = tf.keras.Sequential([
      layers.Conv2D(64, (5, 5), strides=2, activation='relu', padding="same"), # S/4 x S/4 x 64
      layers.Conv2D(128, (3, 3), strides=2, activation='relu', padding="same"), # S/4 x S/4 x 128
      layers.LayerNormalization(), # 정규화
      layers.Conv2D(128, (5, 5), strides=2, activation='relu', padding="same"), # S/4 x S/4 x 256
      layers.Conv2D(256, (5, 5), strides=2, activation='relu', padding="same"), # S/4 x S/4 x 256
      layers.GlobalAveragePooling2D(), # 256
      layers.Dense(num_classes, activation='softmax') # num_classes
    ], name="ClassificationHead")

    self.loss_fn_roi = tf.keras.losses.BinaryCrossentropy()
    self.loss_fn_detection = tf.keras.losses.Huber()
    self.loss_fn_segmentation = tf.keras.losses.BinaryCrossentropy()
    self.loss_fn_centroid = tf.keras.losses.Huber()
    self.loss_fn_classification = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
    self.optimizer = tf.keras.optimizers.AdamW(learning_rate=self.__INITIAL_LR)
    
    self.__build(batch_size, self.__INPUT_SHAPE)
  
  def __build(self, batch_size:int, input_shape:tuple[int, int, int]):
    super(PillNet, self).build(input_shape=(batch_size, *input_shape))

    dummy_input = tf.keras.Input(shape=(input_shape))
    self(dummy_input)
    
  def build(self):
    raise Exception("PillNet is already built")
    
  def call(self, inputs, training=False, gt_detections=None):
    feature_map = self.feature_extractor(inputs)
    roi_mask = self.roi_head(feature_map)
    detection_offsets = self.detection_head(feature_map)
    
    if training and gt_detections is not None:
      oy, ox, oh, ow = tf.split(gt_detections, 4, axis=-1)
    else:
      oy, ox, oh, ow = tf.split(detection_offsets, 4, axis=-1)

    cropped_regions = self.__crop_regions(feature_map, oy, ox, oh, ow)

    segmentation = self.segmentation_head(cropped_regions)
    centroid = self.centroid_head(cropped_regions)
    classification = self.classification_head(cropped_regions)
    
    return {
      "roi": roi_mask,
      "detection": detection_offsets,
      "segmentation": segmentation,
      "centroid": centroid,
      "classification": classification
    }
  
  def __crop_regions(self, feature_map, oy, ox, oh, ow):

    g = self.__GRID_SIZE
    rows, cols = tf.meshgrid(tf.range(g, dtype=tf.float32), tf.range(g, dtype=tf.float32))
    rows = tf.reshape(rows, [-1])  # shape (g*g,)
    cols = tf.reshape(cols, [-1])  # shape (g*g,)

    batch_size = tf.shape(feature_map)[0]
    box_indices = tf.range(batch_size, dtype=tf.int32)
    box_indices = tf.repeat(box_indices, repeats=g*g)
    rows = tf.tile(rows, [batch_size])
    cols = tf.tile(cols, [batch_size])

    oy = tf.reshape(oy, [-1]) 
    ox = tf.reshape(ox, [-1])
    oh = tf.reshape(oh, [-1])
    ow = tf.reshape(ow, [-1])

    img_height = tf.cast(tf.shape(feature_map)[1], tf.float32)
    img_width = tf.cast(tf.shape(feature_map)[2], tf.float32)
    grid_height = img_height / tf.cast(g, tf.float32)  
    grid_width = img_width / tf.cast(g, tf.float32)    

    y_min = rows * grid_height + ((oy * 2.0) - 1.0) * img_height
    x_min = cols * grid_width  + ((ox * 2.0) - 1.0) * img_width
    height = ((oh * 2.0) - 1.0) * img_height + grid_height
    width = ((ow * 2.0) - 1.0) * img_width  + grid_width

    y_min = tf.clip_by_value(y_min, 0.0, img_height -1.0)
    x_min = tf.clip_by_value(x_min, 0.0, img_width -1.0)
    height = tf.clip_by_value(height, 1.0, img_height - y_min)
    width = tf.clip_by_value(width, 1.0, img_width - x_min)

    feature_map_repeated = tf.tile(feature_map[:, None, :, :, :], [1, g*g, 1, 1, 1])
    feature_map_repeated = tf.reshape(feature_map_repeated, [
      batch_size*g*g, tf.shape(feature_map)[1], tf.shape(feature_map)[2], tf.shape(feature_map)[3]
    ])
    cropped_regions = tf.map_fn(
      PillNet.__crop,(
        feature_map_repeated,  
        y_min, x_min, height, width
      ), fn_output_signature= tf.TensorSpec(shape=(None, None, None), dtype=tf.float32)
    )

    crop_h = tf.cast(img_height, tf.int32)
    crop_w = tf.cast(img_width, tf.int32)

    cropped_regions = tf.image.resize(cropped_regions, (crop_h, crop_w), method='bilinear')

    cropped_regions = tf.reshape(
      cropped_regions, [
        batch_size * g*g, crop_h, crop_w, tf.shape(feature_map)[-1]
      ]
    )

    return cropped_regions
  
  @staticmethod
  def __crop(args):
    img, y, x, h, w = args
    return tf.image.crop_to_bounding_box(
      img, 
      tf.cast(y, tf.int32), 
      tf.cast(x, tf.int32), 
      tf.cast(h, tf.int32), 
      tf.cast(w, tf.int32)
    )

  def compile(self):
    super(PillNet, self).compile(
      optimizer=self.optimizer, 
      loss={
        "roi": self.loss_fn_roi,
        "detection": self.loss_fn_detection,
        "segmentation": self.loss_fn_segmentation,
        "centroid": self.loss_fn_centroid,
        "classification": self.loss_fn_classification  
      }, 
      metrics={
        "roi": [iou_metric, dice_metric, pixel_accuracy],
        "detection": ["mae"],
        "segmentation": [iou_metric, dice_metric, pixel_accuracy],
        "centroid": ["mae"],
        "classification": ["accuracy", "precision", "recall", "f1"]
      }
    )

  def train_step(self, data):
    X, Y = data
    
    with tf.GradientTape() as tape:
      y_pred = self(X, training=True, gt_detections=Y["detection"])

      roi_loss = self.loss_fn_roi(Y["roi"], y_pred["roi"])
      detection_loss = self.loss_fn_detection(Y["detection"], y_pred["detection"])
      segmentation_loss = self.loss_fn_segmentation(Y["segmentation"], y_pred["segmentation"])
      centroid_loss = self.loss_fn_centroid(Y["centroid"], y_pred["centroid"])
      classification_loss = self.loss_fn_classification(Y["classification"], y_pred["classification"])
      
      total_loss = roi_loss + detection_loss + segmentation_loss + centroid_loss + classification_loss

    gradients = tape.gradient(total_loss, self.trainable_variables)
    self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
    
    return {
      "roi_loss": roi_loss,
      "detection_loss": detection_loss,
      "segmentation_loss": segmentation_loss,
      "centroid_loss": centroid_loss,
      "classification_loss": classification_loss,
      "total_loss": total_loss
    }
  
  def adjust_learning_rate(self, epoch, decay_rate = 0.1, decay_epochs = 10):
    if epoch % decay_epochs == 0:
      new_lr = self.__INITIAL_LR * (decay_rate ** (epoch // decay_epochs))
      self.optimizer.learning_rate.assign(new_lr)

  def get_config(self):
    config = super(PillNet, self).get_config()
    config.update({
      "initial_lr": self.__INITIAL_LR
    })
    return config
  
  @classmethod
  def from_config(cls, config):
    return cls(**config)

# def test():
#   import numpy as np 
  
#   model = PillNet()
#   model.build(4, (128, 128, 3))
#   print(model.summary())
#   model.compile()

#   num_samples = 16
#   X_dummy = np.random.rand(num_samples, 128, 128, 3).astype(np.float32)  # 입력 이미지
#   Y_dummy = np.random.randint(0, 2, (num_samples, 128, 128, 2)).astype(np.float32)  # 이진 마스크 (0: 배경, 1: 알약)

#   model.fit(X_dummy, Y_dummy, epochs=5, batch_size=4, verbose=1)

#   X_test = np.random.rand(4, 128, 128, 3).astype(np.float32)
#   Y_test = np.random.randint(0, 2, (4, 128, 128, 2)).astype(np.float32)

#   loss = model.evaluate(X_test, Y_test)
#   print(f"🔍 테스트 손실: {loss}")
  
# if __name__ == "__main__":
#   test()