import numpy as np
import pandas as pd
import cv2
from pathlib import Path
import random

from Pills.Utils import *

SOURCE_PATH = 'datasets/pills/data'
LABEL_PATH = 'datasets/pills/class_label.csv'

class Generator:
  def __init__(self, source_dir:str = SOURCE_PATH, label_csv_path:str = LABEL_PATH, suffix:tuple[str] = ('.png', '.jpg')):
    self.__SOURCE_DIR = Path(source_dir)
    self.__LABEL_CSV_PATH = Path(label_csv_path)
    self.__SUFFIX = suffix
    self.__PILLS_DICT = {
      row['class']: row['label']
      for _, row in self.__read_label(self.__LABEL_CSV_PATH).iterrows()
    }
    self.__PILLS_IMGS_DICT = {
      self.__PILLS_DICT[c.stem]: [
        self.__read_image(img) for img in c.iterdir() if img.suffix in self.__SUFFIX
      ]
      for c in self.__SOURCE_DIR.iterdir()
    }
    self.__LABEL_LIST = list(sorted(self.__PILLS_DICT.values()))
    
  def __random_choice(self):
    return random.choice([True, False])
    
  def __read_label(self, label_csv_path:Path):
    return pd.read_csv(label_csv_path)
  
  def __read_image(self, path:Path):
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
    return img

  def __len__(self):
    return len(self.__PILLS_DICT)

  def get_label(self, index:int):
    return self.__LABEL_LIST[index]

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
    objects = [(img, label) for label, imgs in self.__PILLS_IMGS_DICT.items() for img in imgs]
    objects = random.sample(objects, number_of_objects)
    objects = [(rotate(img, random.randint(0, 360)), label) for img, label in objects]
    objects.sort(key=lambda x: x[0].shape[0] * x[0].shape[1], reverse=True)
    objects, labels = zip(*objects)
    objects = list(objects)
    labels = list(labels)

    positions = generate_position(background, objects, scaling, (max_try_of_random_position, max_try_of_adaptive_quadtree), padding, bias)
    if not positions and regeneration:
      return self.generate(background, number_of_objects, scaling, max_try_of_random_position, max_try_of_adaptive_quadtree, padding, bias, regeneration)
    
    objects = list(zip(objects, positions, labels))
    random.shuffle(objects)
    objects, positions, labels = zip(*objects)
    objects = list(objects)
    labels = [self.__LABEL_LIST.index(l) for l in labels]

    mask = mask_at_position(background, objects, positions)
    centroids = centroid_at_mask(mask)
    bboxes = bbox_at_mask(mask)
    contours = contours_at_mask(mask)
    img = draw_at_position(background, objects, positions)
    return img, mask, (objects, labels, centroids, bboxes, contours)

  def random_generate(self, *args, **kargs):
    img, mask, annotations = self.generate(*args, **kargs)

    b_mask = mask[:, :] > 0

    if self.__random_choice():
      img = adjust_saturation_rgba(img, random.uniform(0, 2))
    if self.__random_choice():
      img = adjust_exposure_rgba(img, random.uniform(0.5, 2))
    if self.__random_choice():
      bright_spot = random_bright_spots(img.shape[:2])
      img = add_directional_light(
        img, bright_spot, ksize=(random.randint(0, 100) * 2 + 1), 
        intensity=random.randint(50, 150), increase_factor=random.uniform(1, 1.5)
      )
    if self.__random_choice():
      img[b_mask] = gaussian_blur(img, ksize=random.randint(1, 5) * 2 + 1)[b_mask]
    if self.__random_choice():
      img = random_noise(img, mean=0, std=random.uniform(0, 0.8))
    if self.__random_choice():
      img = gaussian_blur(img, ksize=random.randint(1, 5) * 2 + 1)

    return img, mask, annotations


    
    