import random
import cv2
from pathlib import Path

BACKGROUND_IMAGES_PATH = 'data'

class Background:
  def __init__(self, source_dir:str = BACKGROUND_IMAGES_PATH, suffix:tuple[str] = ('.png', '.jpg')):
    self.__SOURCE_DIR = Path(source_dir)
    self.__SUFFIX = suffix
    self.__BACKGROUND_IMGS = [
      self.__read_image(img) for img in self.__SOURCE_DIR.iterdir() if img.suffix in self.__SUFFIX
    ]
  
  def __read_image(self, path:Path):
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
    return img
  
  def __len__(self):
    return len(self.__BACKGROUND_IMGS)
  
  def generate(self, min_size:int = 512, max_size:int = 1024, index = None):
    if index is not None:
      bkg_imgs = [self.__BACKGROUND_IMGS[index]]
    else:
      bkg_imgs = self.__BACKGROUND_IMGS.copy()
      random.shuffle(bkg_imgs)

    size = random.randint(min_size, max_size)
    
    for bkg_img in bkg_imgs:
      if bkg_img.shape[0] < size or bkg_img.shape[1] < size:
        continue
      h, w = bkg_img.shape[:2]
      row = random.randint(0, h - size)
      col = random.randint(0, w - size)
      return bkg_img[row:row + size, col:col + size]
    raise ValueError('No background image with the required size found')