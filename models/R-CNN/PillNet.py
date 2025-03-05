import tensorflow as tf
from tensorflow.keras import layers, Model

class PillNet(Model):
  def __init__(self, initial_lr = 0.001, **kwargs):
    super(PillNet, self).__init__(**kwargs)

    self.__INITIAL_LR = initial_lr
    
    # 현재 정규화가 포함되어 있지 않은 문제가 존재함.
    # 정규화 적용해볼 것
    self.feature_extractor = tf.keras.Sequential([
      layers.DepthwiseConv2D(depth_multiplier= 16, kernel_size = (9, 9), activation='relu', padding="same"),
      layers.Conv2D(32, (1, 1), activation='relu', padding="same"), # 128x128x32
      layers.Conv2D(32, (9, 9), activation='relu', padding="same"),
      layers.MaxPooling2D((3, 3), strides = 2, padding="same"), # 64x64x32
      layers.Conv2D(16, (3, 3), activation='relu', padding="same"),
      layers.MaxPooling2D((2, 2), strides = 2, padding="same"), # 32x32x16
    ], name="FeatureExtractor")

    self.segmentation_head = tf.keras.Sequential([
      layers.Conv2D(32, (3, 3), activation='gelu', padding="same"), # 32x32x32
      layers.Conv2DTranspose(32, (3, 3), strides=2, activation='gelu', padding="same"), # 64x64x32
      layers.Conv2DTranspose(32, (3, 3), strides=2, activation='gelu', padding="same"), # 128x128x32
      layers.Conv2D(16, (3, 3), activation='gelu', padding="same"), # 128x128x16
      layers.Conv2D(2, (1, 1), activation='sigmoid', padding="same"),
    ], name="SegmentationHead")

    self.loss_fn_mask = tf.keras.losses.BinaryCrossentropy()
    self.optimizer = tf.keras.optimizers.AdamW(learning_rate=self.__INITIAL_LR)
    
  def build(self, batch_size:int, input_shape:tuple[int]=(128, 128, 3)):
    super(PillNet, self).build(input_shape=(batch_size, *input_shape))

    dummy_input = tf.keras.Input(shape=(input_shape))
    self(dummy_input)
    
  def call(self, inputs, training=False):
    x = self.feature_extractor(inputs)
    x = self.segmentation_head(x)
    return x

  def compile(self, metrics=["accuracy"]):
    super(PillNet, self).compile(optimizer=self.optimizer, loss=self.loss_fn_mask, metrics=metrics)

  def train_step(self, data):
    images, masks = data
    
    with tf.GradientTape() as tape:
      predictions = self(images, training=True)
      loss = self.loss_fn_mask(masks, predictions)

    gradients = tape.gradient(loss, self.trainable_variables)
    self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
    
    return {"loss": loss}
  
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

def test():
  import numpy as np 
  
  model = PillNet()
  model.build(4, (128, 128, 3))
  print(model.summary())
  model.compile()

  num_samples = 16
  X_dummy = np.random.rand(num_samples, 128, 128, 3).astype(np.float32)  # 입력 이미지
  Y_dummy = np.random.randint(0, 2, (num_samples, 128, 128, 2)).astype(np.float32)  # 이진 마스크 (0: 배경, 1: 알약)

  model.fit(X_dummy, Y_dummy, epochs=5, batch_size=4, verbose=1)

  X_test = np.random.rand(4, 128, 128, 3).astype(np.float32)
  Y_test = np.random.randint(0, 2, (4, 128, 128, 2)).astype(np.float32)

  loss = model.evaluate(X_test, Y_test)
  print(f"🔍 테스트 손실: {loss}")
  
if __name__ == "__main__":
  test()