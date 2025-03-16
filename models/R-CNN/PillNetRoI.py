import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras import metrics as Metrics

from image_segmentation.Metrics import DiceMetric, IoUMetric, PixelAccuracy
from image_segmentation.loss import dice_loss

INPUT_SHAPE = (128, 128, 3)

class PillNetRoI(Model):
  def __init__(self, initial_lr:float = 0.001, **kwargs):
    super(PillNetRoI, self).__init__(**kwargs)
    
    self._INITIAL_LR = initial_lr

    # S x S x 3 -> S x S x 16, S x S x 16, S x S x 16, S x S x 16
    self.roi_inseption = [
      layers.Conv2D(16, (1, 1), activation='gelu', padding='same'),
      layers.Conv2D(16, (3, 3), activation='gelu', padding='same'),
      layers.Conv2D(16, (5, 1), activation='gelu', padding='same'),
      layers.Conv2D(16, (1, 5), activation='gelu', padding='same'),
    ]

    # S x S x 64 -> S x S x 16
    self.roi_step0 = tf.keras.Sequential([
      layers.Conv2D(16, (3, 3), activation='gelu', padding='same'),
      layers.LayerNormalization(),
    ])

    # S x S x 16 -> S x S x 16
    self.roi_step1 = tf.keras.Sequential([
      layers.Conv2D(8, (3, 3), activation='gelu', padding='same'),
      layers.Conv2D(16, (3, 3), activation='gelu', padding='same'),
      layers.LayerNormalization(),
    ])

    # S x S x 16 -> S x S x 16
    self.roi_step2 = tf.keras.Sequential([
      layers.Conv2D(8, (3, 3), activation='gelu', padding='same'),
      layers.Conv2D(16, (3, 3), activation='gelu', padding='same'),
      layers.LayerNormalization(),
    ])

    # S x S x 16 -> S x S x 16
    self.roi_step3 = tf.keras.Sequential([
      layers.Conv2D(8, (3, 3), activation='gelu', padding='same'),
      layers.Conv2D(16, (3, 3), activation='gelu', padding='same'),
      layers.LayerNormalization(),
    ])

    # S x S x 16 -> S x S x 16
    self.roi_head = tf.keras.Sequential([
      layers.Conv2D(8, (3, 3), activation='gelu', padding='same'),
      layers.Conv2D(16, (3, 3), activation='gelu', padding='same'),
      layers.Conv2D(1, (1, 1), activation='sigmoid', padding='same'),
    ])

  def call(self, inputs, training:bool=False):
    # input: (BATCH, S, S, 3)
    inputs = tf.image.resize(inputs, (self._INPUT_SHAPE[0], self._INPUT_SHAPE[1]))

    # S x S x 3 -> S x S x 16, S x S x 16, S x S x 16, S x S x 16
    x1 = self.roi_inseption[0](inputs)
    x2 = self.roi_inseption[1](inputs)
    x3 = self.roi_inseption[2](inputs)
    x4 = self.roi_inseption[3](inputs)
    x = tf.concat([x1, x2, x3, x4], axis=-1)

    # S x S x 64 -> S x S x 16
    step0 = self.roi_step0(x)
    step1 = self.roi_step1(step0)
    concat1 = tf.concat([x, step1], axis=-1)
    
    step2 = self.roi_step2(concat1)
    concat2 = tf.concat([concat1, step2], axis=-1)
    
    step3 = self.roi_step3(concat2)
    concat3 = tf.concat([concat2, step3], axis=-1)

    # S x S x 16 -> S x S x 16
    roi = self.roi_head(concat3)

    return {
      "roi": roi
    }

  def compile(
    self, 
    loss: tf.keras.losses.Loss = None, 
    optimizer: tf.keras.optimizers.Optimizer = None, 
    metrics: list = [],
    update: bool = True,
    **kwargs
  ):

    if not update:
      super(PillNetRoI, self).compile(**kwargs)
      return

    if loss is None:
      loss = dice_loss
    
    if optimizer is None:
      optimizer = tf.keras.optimizers.Adam(learning_rate=self._INITIAL_LR)
    
    if len(metrics) == 0:
      metrics = {
        "IoU": IoUMetric(),
        "Dice": DiceMetric(),
        "Pixel": PixelAccuracy(),
        "Binary": Metrics.BinaryAccuracy()
      }
  
    self.__losses = {
      "roi": loss
    }
    self.__optimizer = optimizer
    self.__metrics = metrics

    super(PillNetRoI, self).compile(
      optimizer=self.__optimizer,
      loss=self.__losses,
      metrics=self.__metrics,
      **kwargs
    )

  def build(self, batch_size:int, input_shape:tuple[int, int, int]=INPUT_SHAPE, **kwargs):
    self._BATCH_SIZE = batch_size
    self._INPUT_SHAPE = input_shape
    super(PillNetRoI, self).build(input_shape=(batch_size, *input_shape), **kwargs)

    dummy_input = tf.keras.Input(shape=self._INPUT_SHAPE)
    self(dummy_input)
    self.__optimizer.build(self.trainable_variables)

  def _reshape(self, y_true:dict):
    roi_true = tf.image.resize(
      tf.reshape(
        tf.cast(y_true['roi'], tf.float32), 
        (*y_true['roi'].shape, 1)
      ), (self._INPUT_SHAPE[0], self._INPUT_SHAPE[1])
    )
    
    return {
      "roi": roi_true
    }

  def _metric(self, y_true:dict, y_pred:dict):
    roi_true = y_true['roi']
    roi_pred = y_pred['roi']
    metrics = {}

    for key, metric in self.__metrics.items():
      metric.update_state(roi_true, roi_pred)
      metrics[f"roi_{key}"] = metric.result()

    return metrics

  def _loss(self, y_true:dict, y_pred:dict):
    roi_true = y_true['roi']
    roi_pred = y_pred['roi']

    roi_loss = self.__losses['roi'](roi_true, roi_pred)
    
    return {
      "roi": roi_loss
    }
      
  def test_step(self, data):    
    X, Y = data
    
    y_pred = self(X, training=False)

    y_true = self._reshape(Y)
    metrics = self._metric(y_true, y_pred)

    return metrics

  def train_step(self, data):
    X, Y = data
    
    with tf.GradientTape() as tape:
      y_pred = self(X, training=True)
      y_true = self._reshape(Y)
      losses = self._loss(y_true, y_pred)
      
    gradients = tape.gradient(losses['roi'], self.trainable_variables)
    self.__optimizer.apply_gradients(zip(gradients, self.trainable_variables))

    metrics = self._metric(y_true, y_pred)
    
    return metrics

  def get_config(self):
    config = super(PillNetRoI, self).get_config()
    config.update({
      "initial_lr": self._INITIAL_LR
    })
    return config

  @classmethod
  def from_config(cls, config):
    return cls(**config)

  @property
  def metrics(self):
    metrics = []
    for value in self.__metrics.values():
      metrics.append(value)
    return metrics

def test():
  model = PillNetRoI()
  model.compile(jit_compile=False)
  model.build(32)
  model.summary()

  print("🟩 Model test done")

if __name__ == "__main__":
  test()
  



