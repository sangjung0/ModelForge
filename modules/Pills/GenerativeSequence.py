import numpy as np

import image_segmentation
from Pills.Utils import *
from Pills import *

SOURCE_PATH = '_Shared/Datasets/pills/data'
LABEL_PATH = '_Shared/Datasets/pills/class_label.csv'
SHAPE_PATH = '_Shared/Datasets/pills/class_shape_id.csv'
SHAPE_ID_PATH = '_Shared/Datasets/pills/id_shape.csv'
BACKGROUND_IMAGES_PATH = 'data/pills/background'

class GenerativeSequence(image_segmentation.GenerativeSequence):
  def __init__(
    self, 
    data_size:int, 
    batch_size:int, 
    input_shape:tuple[int, int] = (512, 512),
    material_path:str = SOURCE_PATH, 
    label_csv_path:str = LABEL_PATH, 
    shape_csv_path:str = SHAPE_PATH,
    id_shape_csv_path:str = SHAPE_ID_PATH,
    background_images_path:str = BACKGROUND_IMAGES_PATH,
    random_seed = 42,
    shuffle:bool=True, 
    **kwargs
  ):
    super().__init__(data_size, batch_size, shuffle, **kwargs)

    self.__GENERATOR = RandomEffectGenerator(material_path, label_csv_path, shape_csv_path, id_shape_csv_path, random_seed=random_seed)
    self.__BACKGROUND_GENERATOR = Background(background_images_path, random_seed=random_seed)
    self.__INPUT_SHAPE = input_shape
    self.__GRID_SIZE = input_shape[0] // 64

  @property
  def grid_size(self):
    return self.__GRID_SIZE

  @property
  def input_shape(self):
    return self.__INPUT_SHAPE

  @property
  def class_size(self):
    return len(self.__GENERATOR)
  
  @property
  def generator(self):
    return self.__GENERATOR

  def __generate(self):
    img, mask, annotations = self.__GENERATOR.generate_with_random_effect(
      self.__BACKGROUND_GENERATOR.generate(),
      scaling = 10, 
      max_try_of_random_position = 3, 
      max_try_of_adaptive_quadtree = 3, 
      regeneration = True
    )

    return cv2.cvtColor(img, cv2.COLOR_RGBA2RGB), mask, annotations[1:]
  
  def _data_generation(self, indices: list[int]):
    batch_size = len(indices)
    img_height, img_width = self.__INPUT_SHAPE
    shape_size = self.__GENERATOR.shape_size

    X = np.zeros((batch_size, img_height, img_width, 3), dtype=np.float32)

    Y = {
      "roi": np.zeros((batch_size, img_height, img_width, shape_size), dtype=np.bool_),
      "detection": np.zeros((batch_size, self.__GRID_SIZE, self.__GRID_SIZE, 4), dtype=np.float32),
      "centroid": np.zeros((batch_size, self.__GRID_SIZE, self.__GRID_SIZE, 2), dtype=np.float32),
      "segmentation": np.zeros((batch_size, self.__GRID_SIZE, self.__GRID_SIZE, img_height //4, img_width//4), dtype=np.bool_),
      "classification": np.zeros((batch_size, self.__GRID_SIZE, self.__GRID_SIZE), dtype=np.uint32)
    }
    
    for i in range(len(indices)):
        img, _, (indices_, centroids, bboxes, contours) = self.__generate()

        y_scale = img_height / img.shape[0]
        x_scale = img_width / img.shape[1]

        X[i] = cv2.resize(img, (img_width, img_height)).astype(np.float32) / 255
        
        annotations = list(zip(indices_, centroids, bboxes, contours))

        grid_height, grid_width = img_height // self.__GRID_SIZE, img_width // self.__GRID_SIZE
        
        for row in range(self.__GRID_SIZE):
          for col in range(self.__GRID_SIZE):
            y_min = row * grid_height
            x_min = col * grid_width
            center = (y_min + grid_height // 2, x_min + grid_width // 2)
            
            index, centroid, bbox, contour = min(
              annotations,
              key=lambda x: np.linalg.norm(np.array([x[1][0] * y_scale, x[1][1] * x_scale]) - np.array(center))
            )

            centroid = (
              ((centroid[0] * y_scale - y_min) / img_height + 1)/2, 
              ((centroid[1] * x_scale - x_min) / img_width + 1)/2
            )

            bbox = [
              bbox[0] * y_scale,
              bbox[1] * x_scale,
              bbox[2] * y_scale,
              bbox[3] * x_scale,
            ]

            roi_paper = np.zeros((img_height, img_width), dtype=np.uint8)            

            scaled_contour = np.array([[int(x[0][0] * x_scale), int(x[0][1] * y_scale)] for x in contour])
            cv2.drawContours(roi_paper, [scaled_contour], -1, 1, thickness=cv2.FILLED)
            mask = roi_paper[int(bbox[0]):int(bbox[2]), int(bbox[1]):int(bbox[3])]
            mask = cv2.resize(mask, (img_width//4, img_height//4), interpolation=cv2.INTER_NEAREST) > 0

            bbox[0] = (((bbox[0] - y_min) / img_height) + 1)/2
            bbox[1] = (((bbox[1] - x_min) / img_width) + 1)/2
            bbox[2] = (((bbox[2] - y_min) / img_height) + 1)/2
            bbox[3] = (((bbox[3] - x_min) / img_width) + 1)/2

            class_idx = self.__GENERATOR.get_class_from_index(index)
            shape  = self.__GENERATOR.get_shape_from_class(class_idx) - 1
            roi_source = Y["roi"][i, :, :, shape] 
            roi_source[:, :] = roi_source | (roi_paper > 0)
            Y["detection"][i, row, col] = bbox
            Y["centroid"][i, row, col] = centroid
            Y["segmentation"][i, row, col] = mask
            Y["classification"][i, row, col] = index
          
    return X, Y
  
def test():
  DATA_SIZE = 10_000_000
  BATCH_SIZE = 1
  INPUT_SHAPE = (256, 256, 3)
  generative_sequence = GenerativeSequence(DATA_SIZE, BATCH_SIZE, INPUT_SHAPE[:2], use_multiprocessing=True, workers=16)
  
  print("🟩 Test started")
  for i in range(100_000):
    X, Y = generative_sequence[i]
    if len(X) != BATCH_SIZE:
      print(X.shape)
      raise Exception(f"Invalid length: {len(X)}")
    img = X[0]
    
    if img.shape != INPUT_SHAPE:
      raise Exception(f"Invalid shape: {img.shape}")
    if Y["roi"].shape != (BATCH_SIZE, 64, 64):
      raise Exception(f"Invalid shape: {Y['roi'].shape}")
    if Y["detection"].shape != (BATCH_SIZE, 16, 4):
      raise Exception(f"Invalid shape: {Y['detection'].shape}")
    if Y["centroid"].shape != (BATCH_SIZE, 16, 2):
      raise Exception(f"Invalid shape: {Y['centroid'].shape}")
    if Y["segmentation"].shape != (BATCH_SIZE, 16, 64, 64):
      raise Exception(f"Invalid shape: {Y['segmentation'].shape}")
    if Y["classification"].shape != (BATCH_SIZE, 16):
      raise Exception(f"Invalid shape: {Y['classification'].shape}")
    if i % 10 == 0: 
      print(i)

  print("🟩 Test passed")

if __name__ == "__main__":
  test()