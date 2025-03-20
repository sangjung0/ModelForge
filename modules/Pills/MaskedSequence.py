import numpy as np

import image_segmentation
from Pills.Utils import *
from Pills import *
from .MaskedGenerator import MaskedGenerator

SOURCE_PATH = '_Shared/Datasets/pills/data'
LABEL_PATH = '_Shared/Datasets/pills/class_label.csv'
SHAPE_PATH = '_Shared/Datasets/pills/class_shape_id.csv'
SHAPE_ID_PATH = '_Shared/Datasets/pills/id_shape.csv'
BACKGROUND_IMAGES_PATH = 'data/pills/background'

class MaskedSequence(image_segmentation.GenerativeSequence):
  def __init__(
    self,
    data_size: int,
    batch_size: int,
    input_shape: tuple[int, int] = (512, 512),
    material_path:str = SOURCE_PATH, 
    label_csv_path:str = LABEL_PATH, 
    shape_csv_path:str = SHAPE_PATH,
    id_shape_csv_path:str = SHAPE_ID_PATH,
    background_images_path:str = BACKGROUND_IMAGES_PATH,
    random_seed=42,
    shuffle: bool = True,
    **kwargs
  ):
    super().__init__(data_size, batch_size, shuffle, **kwargs)

    self.__GENERATOR= MaskedGenerator(material_path, label_csv_path, shape_csv_path, id_shape_csv_path, random_seed=random_seed)
    self.__BACKGROUND_GENERATOR = Background(background_images_path)
    self.__INPUT_SHAPE = input_shape
  
  @property
  def input_shape(self):
    return self.__INPUT_SHAPE

  @property
  def generator(self):
    return self.__GENERATOR

  def _data_generation(self, indices):
    batch_size = len(indices)
    img_height, img_width = self.__INPUT_SHAPE

    X = np.zeros((batch_size, img_height, img_width, 3), dtype=np.float32)
    Y = np.zeros((batch_size, img_height, img_width, 3), dtype=np.float32)

    for i in range(batch_size): 
      img, masked = self.__GENERATOR.generate_masked(
        self.__BACKGROUND_GENERATOR.generate(),
        scaling = 10,
        max_try_of_random_position = 3,
        max_try_of_adaptive_quadtree = 3,
        regeneration = True
      )

      X[i] = cv2.resize(img, (img_width, img_height)).astype(np.float32) / 255.0
      Y[i] = cv2.resize(masked, (img_width, img_height)).astype(np.float32) / 255.0

    return X, Y

def test():
  DATA_SIZE = 10_000_000
  BATCH_SIZE = 1
  INPUT_SHAPE = (256, 256, 3)
  masked_sequence = MaskedSequence(DATA_SIZE, BATCH_SIZE, INPUT_SHAPE[:2], use_multiprocessing=True, workers=16)
  
  print("🟩 Test started")
  for i in range(100_000):
    X, Y = masked_sequence[i]
    if len(X) != BATCH_SIZE:
      print(X.shape)
      raise Exception("X shape is not correct")
    img = X[0]

    if img.shape != INPUT_SHAPE:
      raise Exception("Image shape is not correct")
    if Y[0].shape != INPUT_SHAPE:
      raise Exception("Masked image shape is not correct")
    if i % 10 == 0:
      print(f"🟩 Test {i} passed")
      
  print("🟩 Test passed")

if __name__ == "__main__":
  test()
    