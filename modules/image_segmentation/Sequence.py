from tensorflow import keras
import numpy as np
import cv2

class Sequence(keras.utils.Sequence):
  def __init__(self, image_paths:list[str], labels:list[np.ndarray], batch_size:int, input_size:tuple[int], augment:bool=False, shuffle:bool=True, **kwargs):
    super().__init__(**kwargs)
    self._AUGMENT = augment
    self._INPUT_SIZE = input_size
    
    self.__BATCH_SIZE = batch_size
    self.__IMAGE_PATHS = image_paths
    self.__LABELS = labels
    self.__SHUFFLE = shuffle
    
    self.__INDEXES = np.arange(len(self.__IMAGE_PATHS))
    self.__LEN = int(np.ceil(len(self.__IMAGE_PATHS) / self.__BATCH_SIZE))
    
    self.on_epoch_end()
    
  def __len__(self):
    """한 epoch에 필요한 batch 개수"""
    return self.__LEN
  
  def __getitem__(self, index):
    """index에 해당하는 batch 데이터를 로드"""
    batch_index = index * self.__BATCH_SIZE
    indexes = self.__INDEXES[batch_index:batch_index + self.__BATCH_SIZE]
    
    image_paths = [self.__IMAGE_PATHS[k] for k in indexes]
    labels = [self.__LABELS[k] for k in indexes]
    
    X, y = self._data_generation(image_paths, labels)
    return X, y

  def _data_generation(self, image_paths:list[str], labels:list[np.ndarray]):
    """데이터 로드 및 증강"""
    X = np.empty((len(image_paths), *self._INPUT_SIZE))
    y = np.array(labels)
    
    for i, path in enumerate(image_paths):
        img = cv2.imread(path)
        img = cv2.resize(img, self._INPUT_SIZE[:2])
        img = img / 255.0  # Normalize
        
        if self._AUGMENT:
            img = self._augment(img)
        
        X[i,] = img
    
    return X, y

  def _augment(self, img:np.ndarray):
    """이미지 증강 (필요하면 추가 가능)"""
    return img
  
  def on_epoch_end(self):
    if self.__SHUFFLE:
        np.random.shuffle(self.__INDEXES)
  
  def get_config(self):
    """시퀀스 설정을 JSON 형식으로 변환 (저장용)"""
    return {
        "image_paths": self.__IMAGE_PATHS,
        "labels": self.__LABELS,
        "batch_size": self.__BATCH_SIZE,
        "input_size": self._INPUT_SIZE,
        "augment": self._AUGMENT,
        "shuffle": self.__SHUFFLE
    }

  @classmethod
  def from_config(cls, config):
    """저장된 설정을 기반으로 시퀀스 복원"""
    return cls(**config)