import tensorflow as tf
from tensorflow.keras import layers, models

INPUT_SHAPE = (256, 256, 3)

class PillNetBackbone(models.Model):
  def __init__(self, **kwargs):
    super(PillNetBackbone, self).__init__(**kwargs)

    self.stem = tf.keras.Sequential([
      layers.Conv2D(32, (7, 7),  activation='relu', padding='same'),
      layers.BatchNormalization(),
    ], name='stem')

    self.C1 = self._make_layer(32, 2, name='C1')
    self.C2 = self._make_layer(64, 2, name='C2', downsample=True)
    self.C3 = self._make_layer(128, 4, name='C3', downsample=True)

  def call(self, inputs, training:bool=False, detailed:bool=False):
    # input: (BATCH, S, S, 3)
    inputs = tf.image.resize(inputs, (self._INPUT_SHAPE[0], self._INPUT_SHAPE[1]))

    x = self.stem(inputs)

    c1 = self.C1(x)
    c2 = self.C2(c1)
    c3 = self.C3(c2)

    if detailed:
      return c1, c2, c3
    return c3

  def compile(
    self,
    loss: tf.keras.losses.Loss = None,
    optimizer: tf.keras.optimizers.Optimizer = None,
    metrics: list[tf.keras.metrics.Metric] = None,
    **kwargs
  ):
    if type(self) == PillNetBackbone:
      return super(PillNetBackbone, self).compile(**kwargs)
    return super(PillNetBackbone, self).compile(
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
    self._BATCH_SIZE = batch_size
    self._INPUT_SHAPE = input_shape
    super(PillNetBackbone, self).build((batch_size, *input_shape), **kwargs)

    dummy_input = tf.keras.Input(shape=self._INPUT_SHAPE)
    self(dummy_input)

  # def fit(self, *args, **kwargs):
  #   if type(self) == PillNetBackbone:
  #     raise NotImplementedError()
  #   return super(PillNetBackbone, self).fit(*args, **kwargs)
  
  def evaluate(self, *args, **kwargs):
    if type(self) == PillNetBackbone:
      raise NotImplementedError()
    return super(PillNetBackbone, self).evaluate(*args, **kwargs)

  def train_step(self, data):
    raise NotImplementedError()
   
  def test_step(self, data):
    raise NotImplementedError()

  def get_config(self):
    return super(PillNetBackbone, self).get_config()

  @classmethod
  def from_config(cls, config):
    return cls(**config)

  def _make_layer(
    self, 
    filters:int, 
    blocks:int, 
    name = None, 
    downsample=False, 
    **kwargs
  ):
    layers = []
    index = 0
    layers.append(ResidualBlock(
      filters, 
      downsample=downsample, 
      name=f"{name}_{index}",
      **kwargs))

    for i in range(index + 1, blocks):
      layers.append(ResidualBlock(filters, name = f"{name}_{i}", **kwargs))
    return tf.keras.Sequential(layers, name=name)

class ResidualBlock(layers.Layer):
  def __init__(
    self, 
    filters:int, 
    kernel:tuple[int, int] = (3, 3), 
    downsample=False, 
    name = None,
    **kwargs
  ):
    super(ResidualBlock, self).__init__(name = name, **kwargs)
    
    stride = 2 if downsample else 1
    
    self.conv1 = layers.Conv2D(filters, kernel, strides=stride, padding='same', use_bias=False)
    self.bn1 = layers.BatchNormalization()
    self.relu = layers.ReLU()
    
    self.conv2 = layers.Conv2D(filters, kernel, strides=1, padding='same', use_bias=False)
    self.bn2 = layers.BatchNormalization()
    
    self.downsample = None
    if downsample:
      self.downsample = tf.keras.Sequential([
        layers.Conv2D(filters, kernel, strides=2, padding='same', use_bias=False),
        layers.BatchNormalization()
      ])
  
  def call(self, x):
    identity = x
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.conv2(x)
    x = self.bn2(x)
    
    if self.downsample is not None:
      identity = self.downsample(identity)
    
    x = layers.Add()([x, identity])
    x = self.relu(x)
    return x

def test():
  model = PillNetBackbone()
  model.build(1)
  model.summary()
  
  print("🟩 Model test done")

if __name__ == "__main__":
  test()