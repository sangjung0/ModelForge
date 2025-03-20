import tensorflow as tf
from tensorflow.keras import layers

from image_segmentation.Metrics import FeatureMatchingMetric, MAEMetric
from image_segmentation.loss import mae_with_feature_matching_loss
from PillNetBackbone import PillNetBackbone

INPUT_SHAPE = (256, 256, 3)

class PillNetPT(PillNetBackbone):
  def __init__(self, initial_lr:float = 0.001, **kwargs):
    super(PillNetPT, self).__init__(**kwargs)

    self._INITIAL_LR = initial_lr

    self.P3_conv = layers.Conv2D(64, (1, 1), padding='same', name='P3_conv')
    self.P3_upsample = layers.UpSampling2D(size=(2, 2), name='P3_upsample')
    self.P2_conv = layers.Conv2D(32, (1, 1), padding='same', name='P2_conv')
    self.P2_upsample = layers.UpSampling2D(size=(2, 2), name='P2_upsample')
    self.final_conv = layers.Conv2D(3, (3, 3), padding='same', activation='sigmoid', name='final_conv')
    
  def call(self, inputs, training:bool=False):

    c1, c2, c3 = super(PillNetPT, self).call(inputs, training, detailed=True)

    p3 = self.P3_conv(c3)
    p3 = self.P3_upsample(p3)
    p3 = layers.Add()([p3, c2])
    
    p2 = self.P2_conv(p3)
    p2 = self.P2_upsample(p2)
    p2 = layers.Add()([p2, c1])
    
    x = self.final_conv(p2)
    return x

  def compile(
    self,
    loss: tf.keras.losses.Loss = None,
    optimizer: tf.keras.optimizers.Optimizer = None,
    metrics: list[tf.keras.metrics.Metric] = None,
    update:bool = True,
    **kwargs
  ):
    if not update:
      super(PillNetPT, self).compile(**kwargs)
      return

    if loss is None:
      loss = mae_with_feature_matching_loss
    if metrics is None:
      metrics = {
        "need_model": [FeatureMatchingMetric()],
        "general": [MAEMetric()]
      }
    if optimizer is None:
      optimizer = tf.keras.optimizers.Adam(learning_rate=self._INITIAL_LR)

    self.__optimizer = optimizer
    self.__loss = loss
    self.__metrics = metrics
    
    super(PillNetPT, self).compile(
      loss=loss,
      optimizer=optimizer,
      metrics=metrics,
      **kwargs
    )

  def build(
    self, 
    batch_size:int, 
    input_shape:tuple[int, int, int] = INPUT_SHAPE, 
    **kwargs
  ):
    super(PillNetPT, self).build(batch_size, input_shape, **kwargs)
    self.__optimizer.build(self.trainable_variables)

  def get_metric(self, y_true, y_pred, model):
    metrics = {}
    for metric in self.__metrics["need_model"]:
      metric.update_state(y_true, y_pred, model)
      metrics[metric.name] = metric.result()
    for metric in self.__metrics["general"]:
      metric.update_state(y_true, y_pred)
      metrics[metric.name] = metric.result()
    return metrics

  def train_step(self, data):
    X, Y = data

    def model(input):
      return super(PillNetPT, self).call(input, training=False)
    
    with tf.GradientTape() as tape:
      Y_pred = self(X, training=True)
      loss = self.__loss(Y, Y_pred, model)
      # tf.print(loss)
    
    gradients = tape.gradient(loss, self.trainable_variables)
    # tf.print(gradients)
    self.__optimizer.apply_gradients(zip(gradients, self.trainable_variables))
    
    return self.get_metric(Y, Y_pred, model)

  def test_step(self, data):
    def model(input):
      return super(PillNetPT, self).call(input, training=False)

    X, Y = data
    Y_pred = self(X, training=False)
    return self.get_metric(Y, Y_pred, model)

  def get_config(self):
    config = super(PillNetPT, self).get_config()
    config.update({
      'initial_lr': self._INITIAL_LR
    })
    return config

  @classmethod
  def from_config(cls, config):
    return cls(**config)

  @property
  def metrics(self):
    metrics = []
    for kind in self.__metrics.values():
      for metric in kind:
        metrics.append(metric)
    return metrics

def test():
  model = PillNetPT()
  model.compile(jit_compile=False)
  model.build(32)
  model.summary()

  print("🟩 Model test done")

if __name__ == "__main__":
  test()