import numpy as np
import pandas as pd
import cv2
from pathlib import Path
import random

from Pills.Utils import *

SOURCE_PATH = '_Shared/Datasets/pills/data'
LABEL_PATH = '_Shared/Datasets/pills/class_label.csv'
SHAPE_PATH = '_Shared/Datasets/pills/class_shape_id.csv'
SHAPE_ID_PATH = '_Shared/Datasets/pills/id_shape.csv'

class Generator:
  def __init__(
    self, 
    source_dir:str = SOURCE_PATH, 
    label_csv_path:str = LABEL_PATH, 
    shape_csv_path:str = SHAPE_PATH, 
    id_shape_csv_path:str = SHAPE_ID_PATH,
    suffix:tuple[str] = ('.png', '.jpg'),
    random_seed:int = 42
  ):

    self._RANDOM = random.Random(random_seed)
    self._NP_RANDOM = np.random.default_rng(random_seed)
  
    self.__PILLS_LABEL_DICT = {
      row['class']: row['label']
      for _, row in pd.read_csv(label_csv_path).iterrows()
    }

    self.__PILLS_SHAPE_DICT = {
      row['class']: row['shape']
      for _, row in pd.read_csv(shape_csv_path).iterrows()
    }

    self.__PILLS_IMGS_DICT = {
      c.stem: [
        self.__read_image(img) for img in c.iterdir() if img.suffix in suffix
      ]
      for c in Path(source_dir).iterdir()
    }

    self.__PILLS_ID_SHAPE_DICT = {
      row['id']: row['shape']
      for _, row in pd.read_csv(id_shape_csv_path).iterrows()
    }

    self.__PILLS_IMGS_LIST = [(img, class_) for class_, imgs in self.__PILLS_IMGS_DICT.items() for img in imgs]
    self.__CLASS_LIST = list(sorted(self.__PILLS_LABEL_DICT.keys()))
    self.__INDEX_DICT = {class_: i for i, class_ in enumerate(self.__CLASS_LIST)}
    
  def __read_image(self, path:Path):
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
    return img

  def __len__(self):
    return len(self.__CLASS_LIST)

  @property
  def shape_size(self):
    return len(self.__PILLS_ID_SHAPE_DICT)

  @property
  def number_of_pills(self):
    return len(self.__PILLS_IMGS_LIST)

  def get_class_from_index(self, index:int):
    return self.__CLASS_LIST[index]

  def get_label_from_class(self, class_name:str):
    return self.__PILLS_LABEL_DICT[class_name]
  
  def get_shape_from_class(self, class_name:str):
    return self.__PILLS_SHAPE_DICT[class_name]

  def generate(
    self,
    background:np.ndarray, 
    number_of_objects:int,
    scaling:int = 10,
    max_try_of_random_position:int = 5,
    max_try_of_adaptive_quadtree:int = 5,
    padding:int = 0,
    bias:bool = False,
    regeneration:bool = False,
  ):
    objects = self._RANDOM.sample(self.__PILLS_IMGS_LIST, number_of_objects)
    objects = [(rotate(img, self._RANDOM.randint(0, 360)), class_) for img, class_ in objects]
    objects.sort(key=lambda x: x[0].shape[0] * x[0].shape[1], reverse=True)
    objects, class_ = zip(*objects)
    objects = list(objects)
    indexes = [self.__INDEX_DICT[cls] for cls in class_]

    positions = generate_position(background, objects, scaling, (max_try_of_random_position, max_try_of_adaptive_quadtree), padding, bias)
    if not positions and regeneration:
      return self.generate(background, number_of_objects, scaling, max_try_of_random_position, max_try_of_adaptive_quadtree, padding, bias, regeneration)
    
    objects = list(zip(objects, positions, indexes))
    self._RANDOM.shuffle(objects)
    objects, positions, indexes = zip(*objects)
    objects = list(objects)

    mask = mask_at_position(background, objects, positions)
    centroids = centroid_at_mask(mask)
    bboxes = bbox_at_mask(mask)
    contours = contours_at_mask(mask)
    img = draw_at_position(background, objects, positions)
    return img, mask, (objects, indexes, centroids, bboxes, contours)


    
    