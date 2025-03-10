import tensorflow as tf
from tensorflow.keras import layers, Model
from image_segmentation.Utils import *

class PillNet(Model):
  def __init__(
    self, 
    batch_size:int, 
    num_classes:int, 
    input_shape:tuple[int, int, int], 
    initial_lr = 0.001, 
    **kwargs
  ):
    super(PillNet, self).__init__(**kwargs)

    self.__INPUT_SHAPE = input_shape
    self.__GRID_SIZE = input_shape[0] // 64
    self.__BATCH_SIZE = batch_size
    self.__INITIAL_LR = initial_lr
    self.__NUM_CLASSES = num_classes
    
    # S x S x 3 -> S/4 x S/4 x 1
    self.roi = tf.keras.Sequential([
      layers.MaxPooling2D((4, 4), strides=4, padding="same"), # S/4 x S/4 x 3
      layers.DepthwiseConv2D(depth_multiplier= 16, kernel_size = (9, 9), activation='relu', padding="same"), # S/4 x S/4 x 48
      layers.LayerNormalization(axis=-1), # 채널 정규화
      layers.Conv2D(16, (1, 1), activation='relu', padding="same"), # Pointwise Convolution, S/4 x S/4 x 64
      layers.Conv2D(16, (9, 9), activation='relu', padding="same"), # S/4 x S/4 x 64 
      layers.LayerNormalization(), # 정규화
      layers.MaxPooling2D((2, 2), strides = 2, padding="same"), # S/8 x S/8 x 64
      layers.Conv2D(32, (7, 7), activation='relu', padding="same"), # S/8 x S/8 x 64
      layers.Conv2D(32, (9, 9), activation='relu', padding="same"), # S/8 x S/8 x 64
      layers.Conv2DTranspose(16, (3, 3), strides=2, activation='gelu', padding="same"), # S/4 x S/4 x 16
      layers.Conv2D(1, (1, 1), activation='sigmoid', padding="same"), # S/4 x S/4 x 1
    ], name="ROI") # 

    # S/4 x S/4 x 1 -> G x G x 4
    self.detection_head = tf.keras.Sequential([
      layers.Conv2D(16, (7, 7), activation='relu', padding="same"), # S/4 x S/4 x 16
      layers.Conv2D(64, (5, 5), activation='relu', padding="same"), # S/4 x S/4 x 64
      layers.MaxPooling2D((2, 2), strides=2, padding="same"), # S/8 x S/8 x 128
      layers.LayerNormalization(), # 정규화
      layers.Conv2D(128, (5, 5), strides=2, activation='relu', padding="same"), # S/16 x S/16 x 128
      layers.Conv2D(128, (3, 3), strides=2, activation='relu', padding="same"), # S/32 x S/32 x 128
      layers.Conv2D(256, (3, 3), strides=2, activation='relu', padding="same"), # G x G x 256
      layers.Conv2D(4, (1, 1), activation='sigmoid', padding="same"), # G x G x 4 
    ], name="DetectionHead")
    
    # S/4 x S/4 x 3 -> S/4 x S/4 x 64
    self.feature_extractor = tf.keras.Sequential([
      layers.DepthwiseConv2D(depth_multiplier= 4, kernel_size = (7, 7), activation='relu', padding="same"), # S/4 x S/4 x 12
      layers.LayerNormalization(axis=-1), # 채널 정규화
      layers.Conv2D(16, (1, 1), activation='relu', padding="same"), # S/4 x S/4 x 16
      layers.Conv2D(32, (5, 5), activation='relu', padding="same"), # S/4 x S/4 x 32
      layers.LayerNormalization(), # 정규화
      layers.Conv2D(64, (3, 3), activation='relu', padding="same"), # S/4 x S/4 x 64
    ], name="FeatureExtractor")
    
    # S/4 x S/4 x 64 -> 2
    self.centroid_head = tf.keras.Sequential([
      layers.MaxPooling2D((2, 2), strides=2, padding="same"), # S/8 x S/8 x 16
      layers.Conv2D(8, (3, 3), strides=2, activation='relu', padding="same"), # S/16 x S/16 x 8
      layers.Conv2D(4, (3, 3), strides=2, activation='relu', padding="same"), # S/32 x S/32 x 4
      layers.Flatten(),
      layers.Dense(2, activation='linear')
    ], name="CentroidHead")

    # S/4 x S/4 x 64 -> S/4 x S/4 x 1
    self.segmentation_head = tf.keras.Sequential([
      layers.Conv2D(16, (3, 3), activation='gelu', padding="same"), # S/4 x S/4 x 16
      layers.Conv2D(32, (7, 7), activation='gelu', padding="same"), # S/4 x S/4 x 16
      layers.Conv2D(1, (1, 1), activation='sigmoid', padding="same"), # S/4 x S/4 x 1
    ], name="SegmentationHead")

    # S/4 x S/4 x 64 -> num_classes
    self.classification_head = tf.keras.Sequential([
      layers.Conv2D(64, (5, 5), strides=2, activation='gelu', padding="same"), # S/4 x S/4 x 64
      layers.Conv2D(64, (5, 5), strides=2, activation='gelu', padding="same"), # S/4 x S/4 x 64
      layers.Conv2D(num_classes, (1, 1), activation='gelu', padding="same"), # S/4 x S/4 x num_classes
      layers.GlobalAveragePooling2D(),
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
    # input: (BATCH, S, S, 3)
    # Grid: G x G, G = S/64
    
    # roi_mask: (BATCH, S/4, S/4, 1)
    roi_mask = self.roi(inputs)

    # detection_offsets: (BATCH, G, G, 4)
    detection_offsets = self.detection_head(roi_mask)
    
    if training and gt_detections is not None:
      # cropped_regions: (BATCH * G*G, S/4, S/4, 3)
      cropped_regions = self.__crop_regions(inputs, gt_detections)
    else:
      # cropped_regions: (BATCH * G*G, S/4, S/4, 3)
      cropped_regions = self.__crop_regions(inputs, detection_offsets)

    # feature_map: (BATCH * G*G, S/4, S/4, 64)
    feature_map = self.feature_extractor(cropped_regions)
    # segmentation: (BATCH * G*G, S/4, S/4, 1)
    segmentation = self.segmentation_head(feature_map)
    # centroid: (BATCH * G*G, 2)
    centroid = self.centroid_head(feature_map)
    # classification: (BATCH * G*G, num_classes)
    classification = self.classification_head(feature_map)
    
    grid_size = self.__GRID_SIZE * self.__GRID_SIZE
    segmentation = tf.reshape(segmentation, [
      self.__BATCH_SIZE, grid_size, self.__INPUT_SHAPE[0]//4, self.__INPUT_SHAPE[1]//4])
    centroid = tf.reshape(centroid, [self.__BATCH_SIZE, grid_size, 2])
    classification = tf.reshape(classification, [self.__BATCH_SIZE, grid_size, self.classification_head.layers[-1].units])
    
    return {
      "roi": roi_mask,
      "detection": detection_offsets,
      "segmentation": segmentation,
      "centroid": centroid,
      "classification": classification
    }
  
  def __crop_regions(self, img, detection_offsets):
    g = self.__GRID_SIZE
    cols, rows = tf.meshgrid(tf.range(g, dtype=tf.float32), tf.range(g, dtype=tf.float32))
    cols = tf.reshape(cols, [-1]) # (g*g,) [0, ... , g-1, 0, ... , g-1, ...]
    rows = tf.reshape(rows, [-1]) # (g*g,) [0, 0, ... , 0, 1, 1, ... , g-1]

    batch_size = tf.shape(img)[0] 
    rows = tf.tile(rows, [batch_size]) # (batch_size*g*g,)
    cols = tf.tile(cols, [batch_size]) # (batch_size*g*g,)

    oy, ox, oh, ow = tf.split(detection_offsets, 4, axis=-1)
    oy = tf.reshape(oy, [-1]) # (batch_size*g*g,) [y1, y2, ... , yg, y1, y2, ... , yg, ...]
    ox = tf.reshape(ox, [-1]) 
    oh = tf.reshape(oh, [-1])
    ow = tf.reshape(ow, [-1])

    img_height = tf.cast(tf.shape(img)[1], tf.float32)
    img_width = tf.cast(tf.shape(img)[2], tf.float32)
    grid_height = img_height / tf.cast(g, tf.float32)  
    grid_width = img_width / tf.cast(g, tf.float32)    

    y_min = rows * grid_height + ((oy * 2.0) - 1.0) * img_height
    x_min = cols * grid_width  + ((ox * 2.0) - 1.0) * img_width
    height = ((oh * 2.0) - 1.0) * img_height
    width = ((ow * 2.0) - 1.0) * img_width

    y_min = tf.clip_by_value(y_min, 0.0, img_height -1.0)
    x_min = tf.clip_by_value(x_min, 0.0, img_width -1.0)
    height = tf.clip_by_value(height, 1.0, img_height - y_min)
    width = tf.clip_by_value(width, 1.0, img_width - x_min)

    # y_max = y_min + height #
    # x_max = x_min + width #
    
    # y_min = y_min / img_height #
    # x_min = x_min / img_width #
    # y_max = y_max / img_height #
    # x_max = x_max / img_width #

    # boxes = tf.stack([y_min, x_min, y_max, x_max], axis=-1) #

    # box_indices = tf.range(batch_size, dtype=tf.int32) #
    # box_indices = tf.repeat(box_indices, repeats=g*g) #

    img = tf.image.convert_image_dtype(img, tf.uint8)
    img_map_repeated = tf.tile(img[:, None, :, :, :], [1, g*g, 1, 1, 1])
    img_map_repeated = tf.reshape(img_map_repeated, [
      batch_size*g*g, tf.shape(img)[1], tf.shape(img)[2], tf.shape(img)[3]
    ])
    cropped_regions = tf.map_fn(
      PillNet.__crop,(
        img_map_repeated,  
        y_min, x_min, height, width
      ), fn_output_signature= tf.TensorSpec(shape=(None, None, None), dtype=tf.uint8)
    )

    crop_h = tf.cast(img_height, tf.int32) // 4
    crop_w = tf.cast(img_width, tf.int32) // 4

    cropped_regions = tf.image.resize(cropped_regions, (crop_h, crop_w), method='bilinear')
    cropped_regions = tf.image.convert_image_dtype(cropped_regions, tf.float32)

    cropped_regions = tf.reshape(
      cropped_regions, [
        batch_size * g*g, crop_h, crop_w, tf.shape(img)[-1]
      ]
    )
    
    # cropped_regions = self.__crop_and_resize(
    #   img, 
    #   boxes, 
    #   box_indices, 
    #   (crop_h, crop_w)
    # ) #

    return cropped_regions
  
  # @staticmethod
  # @tf.function(jit_compile=False)
  # def __crop_and_resize(img, boxes, box_idx, crop_size):
  #   return tf.image.crop_and_resize(
  #     img, 
  #     boxes = boxes, 
  #     box_indices = box_idx, 
  #     crop_size = crop_size,
  #     method='bilinear'
  #   )
  
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

      grid_size = self.__GRID_SIZE * self.__GRID_SIZE * self.__BATCH_SIZE
      y_centroid = tf.reshape(Y["centroid"], [grid_size, 2])
      y_pred_centroid = tf.reshape(y_pred["centroid"], [grid_size, 2])
      centroid_loss = self.loss_fn_centroid(y_centroid, y_pred_centroid)

      y_detection = tf.reshape(Y["detection"], [grid_size, 4])
      y_pred_detection = tf.reshape(y_pred["detection"], [grid_size, 4])
      detection_loss = self.loss_fn_detection(y_detection, y_pred_detection)

      height = self.__INPUT_SHAPE[0] // 4
      width = self.__INPUT_SHAPE[1] // 4
      y_segmentation = tf.reshape(Y["segmentation"], [grid_size, height, width])
      y_pred_segmentation = tf.reshape(y_pred["segmentation"], [grid_size, height, width])
      segmentation_loss = self.loss_fn_segmentation(y_segmentation, y_pred_segmentation)

      y_classification = tf.reshape(Y["classification"], [grid_size])
      y_pred_classification = tf.reshape(y_pred["classification"], [grid_size, self.__NUM_CLASSES])
      classification_loss = self.loss_fn_classification(y_classification, y_pred_classification)
      
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