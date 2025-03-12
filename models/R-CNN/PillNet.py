import tensorflow as tf
from tensorflow.keras import layers, Model
from image_segmentation.loss import dice_loss, dice_using_position_loss, iou_loss, pixel_accuracy_loss, weighted_huber_loss
from image_segmentation.Metrics import IoUMetric, DiceMetric, PixelAccuracy

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

    self.__BATCH_SIZE = batch_size
    self.__NUM_CLASSES = num_classes
    self.__INPUT_SHAPE = input_shape
    self.__INITIAL_LR = initial_lr

    # 굳이 모델이 인식을 해야 할까? 일종의 attention 사용해서 비교를 하는건 어떨까?
    # 현재 metric의 부재와 잘못된 loss 함수로 인해 학습이 잘 안되고 있음
    
    # 이거 왜케 큼? 줄이자
    # 스킵커넥션 왜 안씀? inception 구조도 보자
    # 커널 크기를 키우면 
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

    self.loss_fn_roi = dice_loss
    self.loss_fn_detection = dice_using_position_loss
    self.loss_fn_segmentation = dice_loss
    self.loss_fn_centroid = weighted_huber_loss(100)
    self.loss_fn_classification = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
    self.optimizer = tf.keras.optimizers.AdamW(learning_rate=self.__INITIAL_LR)
    
    self.__build(batch_size, self.__INPUT_SHAPE)
  
  def __build(self, batch_size:int, input_shape:tuple[int, int, int]):
    super(PillNet, self).build(input_shape=(batch_size, *input_shape))

    dummy_input = tf.keras.Input(shape=(input_shape))
    self(dummy_input)
    
  def build(self):
    raise Exception("PillNet is already built")

  def grid_size(self):
    return self.__INPUT_SHAPE[0] // 64
    
  def call(self, inputs, training=False, gt_detections=None):
    # input: (BATCH, S, S, 3)
    # Grid: G x G, G = S/64
    batch_size = tf.shape(inputs)[0]
    
    # roi_mask: (BATCH, S/4, S/4, 1)
    roi_mask = self.roi(inputs)

    # detection_offsets: (BATCH, G, G, 4)
    detection_offsets = self.detection_head(roi_mask)
    grid_size = tf.shape(detection_offsets)[1]
    grid_area = grid_size * grid_size
    
    if training and gt_detections is not None:
      # cropped_regions: (BATCH * G*G, S/4, S/4, 3)
      cropped_regions = self.__crop_regions(inputs, gt_detections, grid_size)
    else:
      detection_offsets = tf.reshape(detection_offsets, [batch_size, grid_area, 4])
      # cropped_regions: (BATCH * G*G, S/4, S/4, 3)
      cropped_regions = self.__crop_regions(inputs, detection_offsets, grid_size)

    # feature_map: (BATCH * G*G, S/4, S/4, 64)
    feature_map = self.feature_extractor(cropped_regions)
    # segmentation: (BATCH * G*G, S/4, S/4, 1)
    segmentation = self.segmentation_head(feature_map)
    # centroid: (BATCH * G*G, 2)
    centroid = self.centroid_head(feature_map)
    # classification: (BATCH * G*G, num_classes)
    classification = self.classification_head(feature_map)
    
    segmentation = tf.reshape(segmentation, [
      batch_size, grid_area, self.__INPUT_SHAPE[0]//4, self.__INPUT_SHAPE[1]//4])
    centroid = tf.reshape(centroid, [batch_size, grid_area, 2])
    classification = tf.reshape(classification, [batch_size, grid_area, self.classification_head.layers[-1].units])

    return {
      "roi": roi_mask,
      "detection": detection_offsets,
      "segmentation": segmentation,
      "centroid": centroid,
      "classification": classification
    }
  
  @tf.function(jit_compile=False)
  def __crop_regions(self, img, detection_offsets, g):
    cols, rows = tf.meshgrid(tf.range(g, dtype=tf.float32), tf.range(g, dtype=tf.float32))
    cols = tf.reshape(cols, [-1]) # (g*g,) [0, ... , g-1, 0, ... , g-1, ...]
    rows = tf.reshape(rows, [-1]) # (g*g,) [0, 0, ... , 0, 1, 1, ... , g-1]

    batch_size = tf.shape(img)[0] 
    rows = tf.tile(rows, [batch_size]) # (batch_size*g*g,)
    cols = tf.tile(cols, [batch_size]) # (batch_size*g*g,)

    ly, lx, ry, rx = tf.split(detection_offsets, 4, axis=-1)
    ly = tf.reshape(ly, [-1]) # (batch_size*g*g,) [y1, y2, ... , yg, y1, y2, ... , yg, ...]
    lx = tf.reshape(lx, [-1]) 
    ry = tf.reshape(ry, [-1])
    rx = tf.reshape(rx, [-1])

    img_height = tf.cast(tf.shape(img)[1], tf.float32)
    img_width = tf.cast(tf.shape(img)[2], tf.float32)
    grid_height = img_height / tf.cast(g, tf.float32)  
    grid_width = img_width / tf.cast(g, tf.float32)    

    y_min = rows * grid_height + ((ly * 2.0) - 1.0) * img_height
    x_min = cols * grid_width  + ((lx * 2.0) - 1.0) * img_width
    y_max = rows * grid_height + ((ry * 2.0) - 1.0) * img_height
    x_max = cols * grid_width + ((rx * 2.0) - 1.0) * img_width

    y_min = tf.clip_by_value(y_min, 0.0, img_height -1.0)
    x_min = tf.clip_by_value(x_min, 0.0, img_width -1.0)
    y_max = tf.clip_by_value(y_max, 1.0, img_height)
    x_max = tf.clip_by_value(x_max, 1.0, img_width)
    
    y_min = y_min / img_height #
    x_min = x_min / img_width #
    y_max = y_max / img_height #
    x_max = x_max / img_width #

    boxes = tf.stack([y_min, x_min, y_max, x_max], axis=-1) #

    box_indices = tf.range(batch_size, dtype=tf.int32) #
    box_indices = tf.repeat(box_indices, repeats=g*g) #

    crop_h = tf.cast(img_height, tf.int32) // 4
    crop_w = tf.cast(img_width, tf.int32) // 4

    cropped_regions = self.__crop_and_resize(
      img, 
      boxes, 
      box_indices, 
      (crop_h, crop_w)
    ) #

    return cropped_regions
  
  @staticmethod
  @tf.function(jit_compile=False)
  def __crop_and_resize(img, boxes, box_idx, crop_size):
    return tf.image.crop_and_resize(
      img, 
      boxes = boxes, 
      box_indices = box_idx, 
      crop_size = crop_size,
      method='bilinear'
    )
  
  # @staticmethod
  # def __crop(args):
  #   img, y, x, h, w = args
  #   return tf.image.crop_to_bounding_box(
  #     img, 
  #     tf.cast(y, tf.int32), 
  #     tf.cast(x, tf.int32), 
  #     tf.cast(h, tf.int32), 
  #     tf.cast(w, tf.int32)
  #   )

  def compile(self, **kwargs):
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
        "roi": [IoUMetric(), DiceMetric(), PixelAccuracy()],
        "detection": ["mae"],
        "segmentation": [IoUMetric(), DiceMetric(), PixelAccuracy()],
        "centroid": ["mae"],
        "classification": ["accuracy"]
      },
      **kwargs
    )

  def train_step(self, data):
    X, Y = data
    
    with tf.GradientTape() as tape:
      y_pred = self(X, training=True, gt_detections=Y["detection"])
      grid_size = tf.shape(y_pred["detection"])[1]
      batch_size = tf.shape(X)[0]
      grid_area = grid_size * grid_size * batch_size

      roi_height, roi_width = Y["roi"].shape[1], Y["roi"].shape[2]
      roi_true = tf.reshape(tf.cast(Y["roi"], tf.float32), [batch_size, roi_height, roi_width, 1])
      roi_pred = tf.reshape(tf.cast(y_pred["roi"] > 0.5, tf.float32), [batch_size, roi_height, roi_width, 1])
      roi_loss = self.loss_fn_roi(roi_true, roi_pred)

      y_centroid = tf.reshape(Y["centroid"], [grid_area, 2])
      y_pred_centroid = tf.reshape(y_pred["centroid"], [grid_area, 2])
      centroid_loss = self.loss_fn_centroid(y_centroid, y_pred_centroid)

      y_detection = tf.reshape(Y["detection"], [grid_area, 4])
      y_pred_detection = tf.reshape(y_pred["detection"], [grid_area, 4])
      detection_loss = self.loss_fn_detection(y_detection, y_pred_detection)

      seg_height, seg_width = Y["segmentation"].shape[2], Y["segmentation"].shape[3]
      y_segmentation = tf.reshape(tf.cast(Y["segmentation"], tf.float32), [grid_area, seg_height, seg_width, 1])
      y_pred_segmentation = tf.reshape(tf.cast(y_pred["segmentation"], tf.float32), [grid_area, seg_height, seg_width, 1])
      segmentation_loss = self.loss_fn_segmentation(y_segmentation, y_pred_segmentation)

      y_classification = tf.reshape(Y["classification"], [grid_area])
      y_pred_classification = tf.reshape(y_pred["classification"], [grid_area, self.__NUM_CLASSES])
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
      "batch_size": self.__BATCH_SIZE,
      "num_classes": self.__NUM_CLASSES,
      "input_shape": self.__INPUT_SHAPE,
      "initial_lr": self.__INITIAL_LR
    })
    return config
  
  @classmethod
  def from_config(cls, config):
    return cls(
      batch_size = config.get("batch_size", 4),
      num_classes = config.get("num_classes", 50),
      input_shape = config.get("input_shape", (256, 256, 3)),
      initial_lr = config.get("initial_lr", 0.001)
    )

def test():
  model = PillNet(4, 50, (256, 256, 3))
  print(model.summary())
  model.compile(jit_compile=False)

  print("🟩 Model test done")

  
if __name__ == "__main__":
  test()