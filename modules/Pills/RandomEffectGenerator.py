import random

from Pills.Utils import *
from Pills import *

SOURCE_PATH = '_Shared/Datasets/pills/data'
LABEL_PATH = '_Shared/Datasets/pills/class_label.csv'
SHAPE_PATH = '_Shared/Datasets/pills/class_shape_id.csv'
SHAPE_ID_PATH = '_Shared/Datasets/pills/id_shape.csv'

class RandomEffectGenerator(Generator):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)

    self.__RANDOM_EFFECTS = [
      lambda img, _: adjust_saturation_rgba(img, self._RANDOM.uniform(0, 2)),
      lambda img, _: adjust_exposure_rgba(img, self._RANDOM.uniform(0.5, 2)),
      lambda img, _: add_directional_light(
        img, random_bright_spots(img.shape[:2], random = self._RANDOM), 
        ksize=(self._RANDOM.randint(0, 100) * 2 + 1),
        intensity=self._RANDOM.randint(50, 150),
        increase_factor=self._RANDOM.uniform(1, 1.5)
      ), lambda img, mask: (
        img.__setitem__(
          mask, gaussian_blur(img, ksize=self._RANDOM.randint(1,5) * 2 + 1)[mask]
        ), img)[1]
      , lambda img, _: random_noise(img, mean=0, std=self._RANDOM.uniform(0, 0.8), np_random = self._NP_RANDOM),
      lambda img, _: gaussian_blur(img, ksize=self._RANDOM.randint(1, 5) * 2 + 1)
    ]

  def generate_with_random_effect(
    self, 
    background:np.ndarray,
    number_of_pills:int = None,
    padding:int = None,
    bias:bool = None,
    **kargs
  ):
    if padding is None:
      padding = self._RANDOM.randint(1, 20)
    if bias is None:
      bias = self._RANDOM.choice([True, False])
    if number_of_pills is None:
      number_of_pills = self._RANDOM.randint(1, 20)
    return self.generate_with_effect(background, number_of_pills, padding = padding, bias = bias, **kargs) 

  def generate_with_effect(self, *args, **kargs):
    img, mask, annotations = self.generate(*args, **kargs)

    b_mask = mask[:, :] > 0

    for effect in self.__RANDOM_EFFECTS:
      if self._RANDOM.choice([True, False]):
        img = effect(img, b_mask)

    return img, mask, annotations
  