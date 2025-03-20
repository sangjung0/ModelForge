import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras import metrics as Metrics
from image_segmentation.loss import dice_loss, dice_using_position_loss, iou_loss, pixel_accuracy_loss
from image_segmentation.Metrics import DiceUsingPositionMetric, IoUMetric, DiceMetric, PixelAccuracy

class PillNet(Model):
  def __init__(
    self, 
    num_classes:int, 
    input_shape:tuple[int, int, int] = (512, 512, 3),
    initial_lr = 0.001, 
    **kwargs
  ):
    super(PillNet, self).__init__(**kwargs)

    self._NUM_CLASSES = num_classes
    self._INITIAL_LR = initial_lr
    self._INPUT_SHAPE = input_shape

    # 굳이 모델이 인식을 해야 할까? 일종의 attention 사용해서 비교를 하는건 어떨까?
    # 현재 metric의 부재와 잘못된 loss 함수로 인해 학습이 잘 안되고 있음
    # lag 사용
    
    # 이거 왜케 큼? 줄이자
    # 스킵커넥션 왜 안씀? inception 구조도 보자, 증명된 최신 기술을 너무 안씀
    # 커널 크기를 키우면 
    # S x S x 3 -> S/4 x S/4 x 1
    self.roi = tf.keras.Sequential([
      layers.DepthwiseConv2D(depth_multiplier= 8, kernel_size = (9, 9), activation='relu', padding="same"), # S x S x 24
      layers.DepthwiseConv2D(depth_multiplier= 8, kernel_size = (5, 5), activation='relu', padding="same"), # S x S x 24
      layers.LayerNormalization(axis=-1), # 채널 정규화
      layers.Conv2D(32, (1, 1), activation='relu', padding="same"), # Pointwise Convolution, S x S x 32
      layers.Conv2D(32, (7, 7), strides=2, activation='relu', padding="same"), # S/2 x S/2 x 32 
      layers.LayerNormalization(), # 정규화
      layers.Conv2D(32, (5, 5), strides=2, activation='relu', padding="same"), # S/4 x S/4 x 32
      layers.Conv2D(32, (3, 3), activation='relu', padding="same"), # S/4 x S/4 x 32
      layers.Conv2D(1, (1, 1), activation='sigmoid', padding="same"), # S/4 x S/4 x 1
    ], name="ROI") # 

    # S/4 x S/4 x 1 -> G x G x 4
    self.detection_head = tf.keras.Sequential([
      layers.Conv2D(4, (5, 5), strides=2, activation='relu', padding="same"), # S/8 x S/8 x 4
      layers.Conv2D(8, (3, 3), strides=2, activation='relu', padding="same"), # S/16 x S/16 x 8
      layers.Conv2D(8, (3, 3), strides=2, activation='relu', padding="same"), # S/32 x S/32 x 8
      layers.Conv2D(4, (3, 3), strides=2, activation='relu', padding="same"), # S/64 x S/64 x 4
      layers.Flatten(),
      layers.Dense(self.grid_size ** 2 * 4, activation='relu'), # G x G x 4
      layers.Dense(self.grid_size ** 2 * 4, activation='sigmoid'), # G x G x 4
      layers.Reshape([self.grid_size, self.grid_size, 4])
    ], name="DetectionHead")
    
    # S/4 x S/4 x 3 -> S/4 x S/4 x 64
    self.feature_extractor = [tf.keras.Sequential([
      layers.DepthwiseConv2D(depth_multiplier= 8, kernel_size = (7, 7), activation='relu', padding="same"), # S/4 x S/4 x 24
      layers.DepthwiseConv2D(depth_multiplier= 8, kernel_size = (3, 3), activation='relu', padding="same"), # S/4 x S/4 x 24
      layers.LayerNormalization(axis=-1), # 채널 정규화
      layers.Conv2D(32, (1, 1), activation='relu', padding="same"), # S/4 x S/4 x 32
      layers.LayerNormalization(), # 정규화
    ], name="FeatureExtractor_1"), tf.keras.Sequential([
      layers.Conv2D(32, (5, 5), activation='relu', padding="same"), # S/4 x S/4 x 32
      layers.Conv2D(32, (3, 3), activation='relu', padding="same"), # S/4 x S/4 x 32
      layers.LayerNormalization(), # 정규화
    ], name="FeatureExtractor_2"), tf.keras.Sequential([
      layers.Conv2D(32, (5, 5), activation='relu', padding="same"), # S/4 x S/4 x 32
      layers.Conv2D(32, (3, 3), activation='relu', padding="same"), # S/4 x S/4 x 32
      layers.LayerNormalization(), # 정규화
    ], name="FeatureExtractor_3")]
    
    # S/4 x S/4 x 64 -> 2
    self.centroid_head = tf.keras.Sequential([
      layers.Conv2D(16, (3, 3), strides=2, activation='relu', padding="same"), # S/8 x S/8 x 16
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
      layers.Conv2D(self._NUM_CLASSES, (1, 1), activation='gelu', padding="same"), # S/4 x S/4 x num_classes
      layers.GlobalAveragePooling2D(),
      layers.Dense(self._NUM_CLASSES, activation='softmax') # num_classes
    ], name="ClassificationHead")
    
  def stage_1(self, inputs, training=False):
    # roi_mask: (BATCH, S/4, S/4, 1)
    roi_mask = self.roi(inputs)
    # detection_offsets: (BATCH, S/4, S/4, 32)
    detection_offsets = self.detection_head(roi_mask)

    return {
      "roi": roi_mask,
      "detection": detection_offsets
    }
    
  def stage_2(self, inputs, training=False):
    # feature_map: (BATCH * G*G, S/4, S/4, 64)
    feature_map_1 = self.feature_extractor[0](inputs)
    # feature_map: (BATCH * G*G, S/4, S/4, 64)
    feature_map_2 = self.feature_extractor[1](feature_map_1)
    feature_map_3 = layers.Add()([feature_map_1, feature_map_2])
    # feature_map: (BATCH * G*G, S/4, S/4, 64)
    feature_map_4 = self.feature_extractor[2](feature_map_3)
    feature_map = layers.Add()([feature_map_3, feature_map_4])

    # segmentation: (BATCH * G*G, S/4, S/4, 1)
    segmentation = self.segmentation_head(feature_map)
    # centroid: (BATCH * G*G, 2)
    centroid = self.centroid_head(feature_map)
    # classification: (BATCH * G*G, num_classes)
    classification = self.classification_head(feature_map)

    return {
      "segmentation": segmentation,
      "centroid": centroid,
      "classification": classification
    }
    
  def call(self, inputs, training=False, gt_detections=None):
    # input: (BATCH, S, S, 3)
    # Grid: G x G, G = S/64
    batch_size = tf.shape(inputs)[0]
    resized_inputs = tf.image.resize(inputs, self._INPUT_SHAPE[:2])

    stage_1 = self.stage_1(resized_inputs, training=training)
    roi_mask, detection_offsets = stage_1["roi"], stage_1["detection"]
    grid_size = tf.shape(detection_offsets)[1]

    # 사실 안쓰임
    if training and gt_detections is not None:
      # cropped_regions: (BATCH * G*G, S/4, S/4, 3)
      cropped_regions = self._crop_regions(inputs, gt_detections, grid_size)
    else:
      # cropped_regions: (BATCH * G*G, S/4, S/4, 3)
      cropped_regions = self._crop_regions(inputs, detection_offsets, grid_size)

    stage_2 = self.stage_2(cropped_regions, training=training)
    segmentation, centroid, classification = stage_2["segmentation"], stage_2["centroid"], stage_2["classification"]
    
    segmentation = tf.reshape(segmentation, [
      batch_size, grid_size, grid_size, self._INPUT_SHAPE[0]//4, self._INPUT_SHAPE[1]//4])
    centroid = tf.reshape(centroid, [batch_size, grid_size, grid_size, 2])
    classification = tf.reshape(classification, [batch_size, grid_size, grid_size, self.classification_head.layers[-1].units])

    return {
      "roi": roi_mask,
      "detection": detection_offsets,
      "segmentation": segmentation,
      "centroid": centroid,
      "classification": classification
    }

  def compile(self, loss:dict=None, metrics:dict=None, optimizer = None, **kwargs):
    if loss is None:
      loss = {
        "roi": dice_loss,
        "detection": dice_using_position_loss,
        "segmentation": dice_loss,
        "centroid": tf.keras.losses.Huber(),
        "classification": tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
      }

    if metrics is None:
      metrics = {
        "roi":{
          "iou": IoUMetric(),
          "dice": DiceMetric(),
          "pixel_accuracy": PixelAccuracy()
        }, "detection": {
          "dice_using_position": DiceUsingPositionMetric()
        }, "segmentation": {
          "iou": IoUMetric(),
          "dice": DiceMetric(),
          "pixel_accuracy": PixelAccuracy()
        }, "centroid": {
          "mae": Metrics.MeanAbsoluteError()
        }, "classification": {
          "accuracy": Metrics.SparseCategoricalAccuracy()
        }
      }

    if not isinstance(loss, dict):
      raise ValueError("loss should be a dictionary")
    for key, value in loss.items():
      if not callable(value):
        raise ValueError(f"loss[{key}] should be a callable")

    if "roi" not in loss:
      loss["roi"] = dice_loss
    if "detection" not in loss:
      loss["detection"] = dice_using_position_loss
    if "segmentation" not in loss:
      loss["segmentation"] = dice_loss
    if "centroid" not in loss:
      loss["centroid"] = tf.keras.losses.Huber()
    if "classification" not in loss:
      loss["classification"] = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)

    keys = ["roi", "detection", "segmentation", "centroid", "classification"]
    
    if not isinstance(metrics, dict):
      raise ValueError("metrics should be a dictionary")
    for key, value in metrics.items():
      if key not in keys:
        raise ValueError(f"metrics should contain key '{key}'")
      if not isinstance(value, dict):
        raise ValueError(f"metrics[{key}] should be a dictionary")

    if optimizer is None:
      optimizer = tf.keras.optimizers.AdamW(learning_rate=self._INITIAL_LR)
    if not isinstance(optimizer, tf.keras.optimizers.Optimizer):
      raise ValueError("optimizer should be an instance of tf.keras.optimizers.Optimizer")

    super(PillNet, self).compile(
      optimizer=optimizer, 
      # loss=lambda y_true, y_pred: 0,
      loss = loss, 
      metrics = metrics,
      **kwargs
    )

    self.__metrics = metrics
    self.__loss = loss
    self.__optimizer = optimizer

  def compile_option(self, loss:dict=None, metrics:dict=None, optimizer = None, **kwargs):
    super(PillNet, self).compile(**kwargs)

  def build(self, batch_size:int, input_shape:tuple[int, int, int], **kwargs):
    self._BATCH_SIZE = batch_size
    super(PillNet, self).build(input_shape=(batch_size, *self._INPUT_SHAPE), **kwargs)

    dummy_input = tf.keras.Input(shape=(self._INPUT_SHAPE))
    self(dummy_input)
    self.__optimizer.build(self.trainable_variables)

  def summary(self, **kwargs):
    super(PillNet, self).summary(**kwargs)
  
  def _crop_regions(self, img, detection_offsets, g):
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

    crop_h = tf.cast(self._INPUT_SHAPE[0], tf.int32) // 4
    crop_w = tf.cast(self._INPUT_SHAPE[1], tf.int32) // 4

    # tf.print("\n 이미지 크기, 박스 크기, 박스 인덱스 크기, 높이, 너비")
    # tf.print(tf.shape(img), tf.shape(boxes), tf.shape(box_indices), crop_h, crop_w)
    # tf.print(boxes)
    # tf.print("\n")
    cropped_regions = self._crop_and_resize(
      img, 
      boxes, 
      box_indices, 
      (crop_h, crop_w)
    ) #

    return cropped_regions
  
  def _crop_and_resize(self, img, boxes, box_idx, crop_size):
    return tf.image.crop_and_resize(
      img, 
      boxes = boxes, 
      box_indices = box_idx, 
      crop_size = crop_size,
      method='bilinear'
    )
    
  def _stage1_reshape(self, y_true, y_pred):
    grid_size = tf.shape(y_pred["detection"])[1]
    batch_size = tf.shape(y_pred["roi"])[0]
    grid_area = grid_size * grid_size * batch_size
    
    roi_height, roi_width = y_pred["roi"].shape[1], y_pred["roi"].shape[2]
    # 이 코드 문제 있음
    roi_true = tf.reshape(
      tf.image.resize(tf.cast(y_true["roi"], tf.float32), (roi_height, roi_width)),
      [batch_size, roi_height, roi_width, 1])

    detection_true = tf.reshape(tf.cast(y_true["detection"], tf.float32), [grid_area, 4])
    detection_pred = tf.reshape(y_pred["detection"], [grid_area, 4])
    
    true = {
      "roi": roi_true,
      "detection": detection_true,
    }
    pred = {
      "roi": y_pred["roi"],
      "detection": detection_pred,
    }

    return true, pred

  def _stage2_reshape(self, y_true, y_pred):
    grid_area = tf.shape(y_pred["centroid"])[0]
    
    centroid_true = tf.reshape(y_true["centroid"], [grid_area, 2])

    seg_height, seg_width = y_pred["segmentation"].shape[1], y_pred["segmentation"].shape[2]
    segmentation_true = tf.reshape(tf.cast(y_true["segmentation"], tf.float32), [grid_area, seg_height, seg_width, 1])

    classification_true = tf.reshape(y_true["classification"], [grid_area])
    
    true = {
      "centroid": centroid_true,
      "segmentation": segmentation_true,
      "classification": classification_true
    }
    pred = {
      "centroid": y_pred["centroid"],
      "segmentation": y_pred["segmentation"],
      "classification": y_pred["classification"] 
    }

    return true, pred
    
  def _stage1_metrics(self, y_true, y_pred):
    metrics = {}
    for key, value in self.__metrics["roi"].items():
      value.update_state(y_true["roi"], y_pred["roi"])
      metrics[f"roi_{key}"] = value.result()
    for key, value in self.__metrics["detection"].items():
      value.update_state(y_true["detection"], y_pred["detection"])
      metrics[f"detection_{key}"] = value.result()
    return metrics

  def _stage2_metrics(self, y_true, y_pred):
    metrics = {}
    for key, value in self.__metrics["segmentation"].items():
      value.update_state(y_true["segmentation"], y_pred["segmentation"])
      metrics[f"segmentation_{key}"] = value.result()
    for key, value in self.__metrics["centroid"].items():
      value.update_state(y_true["centroid"], y_pred["centroid"])
      metrics[f"centroid_{key}"] = value.result()
    for key, value in self.__metrics["classification"].items():
      value.update_state(y_true["classification"], y_pred["classification"])
      metrics[f"classification_{key}"] = value.result()

    return metrics

  def _stage1_losses(self, y_true, y_pred):
    losses = {}
    losses['roi'] = self.__loss['roi'](y_true["roi"], y_pred["roi"])
    losses['detection'] = self.__loss['detection'](y_true["detection"], y_pred["detection"])
    return losses

  def _stage2_losses(self, y_true, y_pred):
    losses = {}
    losses['segmentation'] = self.__loss['segmentation'](y_true["segmentation"], y_pred["segmentation"])
    losses['centroid'] = self.__loss['centroid'](y_true["centroid"], y_pred["centroid"])
    losses['classification'] = self.__loss['classification'](y_true["classification"], y_pred["classification"])
    return losses

  def test_step(self, data):
    X, Y = data

    resized_inputs = tf.image.resize(X, self._INPUT_SHAPE[:2])
    stage_1 = self.stage_1(resized_inputs, training=False)
    grid_size = tf.shape(stage_1["detection"])[1]
    cropped_regions = self._crop_regions(X, stage_1["detection"], grid_size)
    stage_2 = self.stage_2(cropped_regions, training=False)

    stage1_y_true, stage1_y_pred = self._stage1_reshape(Y, stage_1)
    stage2_y_true, stage2_y_pred = self._stage2_reshape(Y, stage_2)

    stage1_metrics = self._stage1_metrics(stage1_y_true, stage1_y_pred)
    stage2_metrics = self._stage2_metrics(stage2_y_true, stage2_y_pred)

    return {**stage1_metrics, **stage2_metrics}

  def train_step(self, data):
    X, Y = data

    resized_inputs = tf.image.resize(X, self._INPUT_SHAPE[:2])
    cropped_regions = self._crop_regions(X, Y["detection"], tf.shape(Y["detection"])[1])
    with tf.GradientTape(persistent=True) as tape:
      stage_1 = self.stage_1(resized_inputs, training=True)
      stage1_true, stage1_pred = self._stage1_reshape(Y, stage_1)
      stage1_losses = self._stage1_losses(stage1_true, stage1_pred)
      stage1_total_loss = stage1_losses['roi'] + stage1_losses['detection']
      stage1_variables = (
        self.roi.trainable_variables + 
        self.detection_head.trainable_variables
      )
      
      stage_2 = self.stage_2(cropped_regions, training=True)
      stag2_true, stage2_pred = self._stage2_reshape(Y, stage_2)
      stage2_losses = self._stage2_losses(stag2_true, stage2_pred)
      stage2_total_loss = +stage2_losses['segmentation'], stage2_losses['centroid'], stage2_losses['classification']])
      stage2_variables = (
        self.feature_extractor[0].trainable_variables + 
        self.feature_extractor[1].trainable_variables +
        self.feature_extractor[2].trainable_variables +
        self.segmentation_head.trainable_variables + 
        self.centroid_head.trainable_variables + 
        self.classification_head.trainable_variables
      )

    stage1_gradients = tape.gradient(stage1_total_loss, stage1_variables)
    tf.print("Stage 1 Gradients: ", stage1_gradients)
    self.__optimizer.apply_gradients(zip(stage1_gradients, stage1_variables))
    
    stage2_gradients = tape.gradient(stage2_total_loss, stage2_variables)
    tf.print("Stage 2 Gradients: ", stage2_gradients)
    self.__optimizer.apply_gradients(zip(stage2_gradients, stage2_variables))
    
    del tape

    stage1_metrics = self._stage1_metrics(stage1_true, stage1_pred)
    stage2_metrics = self._stage2_metrics(stag2_true, stage2_pred)
    
    return {**stage1_metrics, **stage2_metrics}
  
  def adjust_learning_rate(self, epoch, decay_rate = 0.1, decay_epochs = 10):
    if epoch % decay_epochs == 0:
      new_lr = self._INITIAL_LR * (decay_rate ** (epoch // decay_epochs))
      self.__optimizer.learning_rate.assign(new_lr)

  def get_config(self):
    config = super(PillNet, self).get_config()
    config.update({
      "num_classes": self._NUM_CLASSES,
      "input_shape": self._INPUT_SHAPE,
      "initial_lr": self._INITIAL_LR
    })
    return config

  @property
  def grid_size(self):
    return self._INPUT_SHAPE[0] // 64

  @property
  def metrics(self):
    metrics = []
    for value in self.__metrics.values():
      for v in value.values():
        metrics.append(v)
    return metrics
  
  @classmethod
  def from_config(cls, config):
    return cls(
      num_classes = config.get("num_classes", 50),
      input_shape = config.get("input_shape", (512, 512, 3)),
      initial_lr = config.get("initial_lr", 0.001)
    )

def test():
  model = PillNet(4, 50, (256, 256, 3))
  print(model.summary())
  model.compile(jit_compile=False)

  print("🟩 Model test done")

  
if __name__ == "__main__":
  test()