from pathlib import Path
import numpy as np
from tensorflow.keras import layers, models, activations, losses, optimizers, metrics, regularizers
from tensorflow.keras.utils import Sequence

class AlexNet():
  PADDING='same'
  WEIGHT_DECAY=0.0005
  
  def __init__(self, input_shape=(277, 277, 3), num_classes=1000):
    self._model = AlexNet.__create(input_shape, num_classes)
  
  def __create(input_shape, num_classes):
    model = models.Sequential([
      layers.Input(shape=input_shape),

      layers.Conv2D(filters=96, kernel_size=11, strides=4, padding=AlexNet.PADDING, kernel_regularizer=regularizers.l2(AlexNet.WEIGHT_DECAY)),
      layers.LayerNormalization(), 
      layers.Activation(activations.relu),
      layers.MaxPooling2D(pool_size=(3,3), strides=2),

      layers.Conv2D(filters=256, kernel_size=5, padding=AlexNet.PADDING, kernel_regularizer=regularizers.l2(AlexNet.WEIGHT_DECAY)),
      layers.LayerNormalization(),
      layers.Activation(activations.relu),
      layers.MaxPooling2D(pool_size=(3,3), strides=2),

      layers.Conv2D(filters=384, kernel_size=3, padding=AlexNet.PADDING, activation=activations.relu, kernel_regularizer=regularizers.l2(AlexNet.WEIGHT_DECAY)), 

      layers.Conv2D(filters=384, kernel_size=3, padding=AlexNet.PADDING, activation=activations.relu, kernel_regularizer=regularizers.l2(AlexNet.WEIGHT_DECAY)), 

      layers.Conv2D(filters=256, kernel_size=3, padding=AlexNet.PADDING, activation=activations.relu, kernel_regularizer=regularizers.l2(AlexNet.WEIGHT_DECAY)), 
      layers.MaxPooling2D(pool_size=(3,3), strides=2),

      layers.Flatten(),

      layers.Dense(4096, activation=activations.relu, kernel_regularizer=regularizers.l2(AlexNet.WEIGHT_DECAY)),
      layers.Dropout(0.5),

      layers.Dense(4096, activation=activations.relu, kernel_regularizer=regularizers.l2(AlexNet.WEIGHT_DECAY)),
      layers.Dropout(0.5),
      
      layers.Dense(num_classes, activation=activations.softmax)
    ])
    
    model.compile(
      loss = losses.SparseCategoricalCrossentropy(),
      optimizer = optimizers.SGD(learning_rate=0.01, momentum=0.9),
      metrics= [metrics.SparseCategoricalAccuracy()]
    )

    return model
    
  def train(self, train_data, epochs=90, validation_data=None):
    return self._model.fit(
      train_data,
      epochs=epochs,
      validation_data=validation_data,
    )

  def predict(self, x):
    return self._model.predict(x)

  def save(self, path:Path):
    self._model.save(path / "alexnet_model.h5")

  # Property
  @property
  def model(self):
    return self._model



def test():
  class DummyGenerator(Sequence):
    def __len__(self):
        return 10

    def __getitem__(self, idx):
        # 임시로 더미 데이터 반환
        x = np.random.rand(32, 277, 277, 3).astype("float32")
        y = np.random.rand(32, 1000).astype("float32")
        return x, y
      
  model = AlexNet(input_shape=(277,277,3), num_classes=1000)
  gen = DummyGenerator()
  model.train(gen, epochs=1)  # workers=4, use_multiprocessing=True
  print("🟩 AlexNet test passed")

if __name__ == "__main__":
  test()