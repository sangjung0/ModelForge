import random
import numpy as np

import image_segmentation
from Pills.Utils import *
from Pills import *

SOURCE_PATH = 'datasets/pills/data'
LABEL_PATH = 'datasets/pills/class_label.csv'
BACKGROUND_IMAGEES_PATH = 'data'

class GenerativeSequence(image_segmentation.GenerativeSequence):
  def __init__(
    self, 
    data_size:int, 
    batch_size:int, 
    input_size:tuple[int, int] = (256, 256),
    max_detection_count_root:int = 7,
    material_path:str = SOURCE_PATH, 
    label_path:str = LABEL_PATH, 
    background_images_path:str = BACKGROUND_IMAGEES_PATH,
    random_seed = 42,
    shuffle:bool=True, **kwargs
  ):
    super().__init__(data_size, batch_size, shuffle, **kwargs)
    random.seed(random_seed)
    np.random.seed(random_seed)

    self.__GENERATOR = Generator(material_path, label_path)
    self.__BACKGROUND_GENERATOR = Background(background_images_path)
    self.__INPUT_SIZE = input_size
    self.__MAX_DETECTION_COUNT = max_detection_count_root ** 2
    self.__MAX_DETECTION_COUNT_ROOT = max_detection_count_root
    
  def __random_choice(self):
    return random.choice([True, False])

  def get_shape(self):
    return self.__INPUT_SIZE
  
  def get_label(self, index:int):
    return self.__GENERATOR.get_label(index)
  
  def _data_generation(self, indexes: list[int]):
    batch_size = len(indexes)
    img_height, img_width = self.__INPUT_SIZE

    X = np.zeros((batch_size, img_height, img_width, 3), dtype=np.uint8)

    Y = {
      "roi": np.zeros((batch_size, img_height, img_width), dtype=np.bool_),
      "centroid": np.zeros((batch_size, self.__MAX_DETECTION_COUNT, 2), dtype=np.float32),
      "detection": np.zeros((batch_size, self.__MAX_DETECTION_COUNT, 4), dtype=np.float32),
      "segmentation": np.zeros((batch_size, self.__MAX_DETECTION_COUNT, img_height, img_width), dtype=np.bool_),
      "classification": np.zeros((batch_size, self.__MAX_DETECTION_COUNT), dtype=np.uint32)
    }

    for i in range(len(indexes)):
        img, mask, (labels, centroids, bboxes, contours) = self.__data_generation()

        y_scale = img_height / img.shape[0]
        x_scale = img_width / img.shape[1]

        X[i] = cv2.resize(img, (img_width, img_height))
        Y["roi"][i] = cv2.resize(mask.astype(np.uint8), (img_width, img_height)) > 0
        
        annotations = list(zip(labels, centroids, bboxes, contours))

        grid_height, grid_width = img_height // self.__MAX_DETECTION_COUNT_ROOT, img_width // self.__MAX_DETECTION_COUNT_ROOT
        
        grid_index = 0
        for row in range(self.__MAX_DETECTION_COUNT_ROOT):
          for col in range(self.__MAX_DETECTION_COUNT_ROOT):
            y_min = row * grid_height
            x_min = col * grid_width
            center = (y_min + grid_height // 2, x_min + grid_width // 2)
            
            label, centroid, bbox, contour = min(
              annotations,
              key=lambda x: np.linalg.norm(np.array([x[1][0] * y_scale, x[1][1] * x_scale]) - np.array(center))
            )

            centroid = (centroid[0] * y_scale / img_height, centroid[1] * x_scale / img_width)

            bbox = [
              bbox[0] * y_scale,
              bbox[1] * x_scale,
              bbox[2] * y_scale,
              bbox[3] * x_scale,
            ]
            height = bbox[2] - bbox[0]
            width = bbox[3] - bbox[1]
            bbox[0] = (((bbox[0] - y_min) / img_height) + 1)/2
            bbox[1] = (((bbox[1] - x_min) / img_width) + 1)/2
            bbox[2] = (((height - grid_height) / img_height) + 1)/2
            bbox[3] = (((width - grid_width) / img_width) + 1)/2

            contour = np.array([[int(x[0][0] * x_scale), int(x[0][1] * y_scale)] for x in contour])

            mask = np.zeros((img_height, img_width), dtype=np.uint8)
            cv2.drawContours(mask, [contour], -1, 1, thickness=cv2.FILLED)

            Y["centroid"][i, grid_index] = centroid
            Y["detection"][i, grid_index] = bbox
            Y["segmentation"][i, grid_index] = mask > 0
            Y["classification"][i, grid_index] = label
            grid_index += 1

    return X, Y
    

  def __data_generation(self):
    img, mask, annotations = self.__GENERATOR.generate(
      self.__BACKGROUND_GENERATOR.generate(),
      random.randint(1, 20),
      10, 3, 3, random.randint(0, 10),
      self.__random_choice(), True, True
    )

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

    return cv2.cvtColor(img, cv2.COLOR_RGBA2RGB), mask, annotations[1:]