import image_segmentation
import numpy as np
import cv2

class Sequence(image_segmentation.Sequence):
  def __init__(self, image_paths:list[str], labels:list[np.ndarray], batch_size:int, input_size:tuple[int], augment:bool=False, shuffle:bool=True):
    super().__init__(image_paths, labels, batch_size, input_size, augment, shuffle)
    
  def _data_generation(self, image_paths, labels):

    X = np.empty((len(image_paths), *self._INPUT_SIZE))
    y = np.empty((len(image_paths), *self._INPUT_SIZE[:2], 2))

    for i, path in enumerate(image_paths):
      image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
      shape = image.shape
      image = cv2.resize(image, self._INPUT_SIZE[:2])
      image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)

      rgb_image = image[:, :, :3]
      mask = (image[:, :, 3] == 0)
      rgb_image[mask, :] = 0
      rgb_image = rgb_image / 255.0
      
      if self._AUGMENT:
        rgb_image = self._augment(rgb_image)

      mask = [np.zeros(shape[:2], dtype=np.float32), np.zeros(shape[:2], dtype=np.float32)]
      for label_idx, label in enumerate(labels[i]):
        for contour in label:
          cv2.drawContours(mask[label_idx], [contour], -1, (1, ), thickness=cv2.FILLED) 
      mask = np.stack(mask, axis=-1)
      mask = cv2.resize(mask, self._INPUT_SIZE[:2], interpolation=cv2.INTER_NEAREST)
      
      X[i,] = rgb_image
      y[i,] = mask
    
    return X, y

  def _augment(self, img:np.ndarray):
    pass
      
      

      

