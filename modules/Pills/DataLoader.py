import numpy as np
from pathlib import Path
import json
from sklearn.model_selection import train_test_split

from Pills.Sequence import Sequence

SEED = 42
ANNOTATION_PATH = "Datasets/pills/data/annotations.json"
RESULT_IMG_SHAPE = (128, 128, 3)

class DataLoader:
  def __init__(self, batch_size:int, base_path:str = "./", path:Path = Path(ANNOTATION_PATH)):
    self.__BATCH_SIZE = batch_size
    self.__PATH = path
    self.__load(base_path)
    
  def __load(self, base_path:str):
    with open(self.__PATH, "r") as f:
      data = json.load(f)
    self.__CATEGORY_MAP = data["shape_categories"]
    
    paths = []
    labels = []
    annotation_gen = (annotation for annotation in data["annotations"])
    cnt_ann = next(annotation_gen)
    for image_info in data["images"]:
      paths.append(str(Path(base_path) / Path(image_info["file_path"])))
      label:tuple[list] = ([], [])
      while cnt_ann["image_id"] == image_info["id"]:
        label[cnt_ann["shape"] - 1].append(np.array(cnt_ann["segmentation"], dtype=np.int32).reshape(-1, 2))
        try:
          cnt_ann = next(annotation_gen)
        except StopIteration:
          break
      labels.append(label)
    
    x_train, x_test, y_train, y_test = train_test_split(paths, labels, test_size=0.2, random_state=SEED)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.25, random_state=SEED)

    self.__x_train = x_train
    self.__x_val = x_val
    self.__x_test = x_test
    self.__y_train = y_train
    self.__y_val = y_val
    self.__y_test = y_test
      
  def get_train_sequence(self):
    return Sequence(self.__x_train, self.__y_train, self.__BATCH_SIZE, RESULT_IMG_SHAPE, shuffle=False)

  def get_valid_sequence(self):
    return Sequence(self.__x_val, self.__y_val, self.__BATCH_SIZE, RESULT_IMG_SHAPE, shuffle=False)

  def get_test_sequence(self):
    return Sequence(self.__x_test, self.__y_test, self.__BATCH_SIZE, RESULT_IMG_SHAPE, shuffle=False)


def test():
  loader = DataLoader(4)
  train_sequence = loader.get_train_sequence()
  valid_sequence = loader.get_valid_sequence()
  test_sequence = loader.get_test_sequence()
  print(train_sequence)
  print(valid_sequence)
  print(test_sequence)

if __name__ == "__main__":
  test()