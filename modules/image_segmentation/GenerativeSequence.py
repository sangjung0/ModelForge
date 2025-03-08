from tensorflow import keras
import numpy as np

class GenerativeSequence(keras.utils.Sequence):
  def __init__(self, data_size:int, batch_size:int, shuffle:bool=True, **kwargs):
    super().__init__(**kwargs)
    self.__DATA_SIZE = data_size
    self.__BATCH_SIZE = batch_size
    self.__SHUFFLE = shuffle

    self.__INDEXES = np.arange(self.__DATA_SIZE)
    self.__LEN = int(np.ceil(self.__DATA_SIZE / self.__BATCH_SIZE))

    self.on_epoch_end()
    
  def __len__(self):
    """한 epoch에 필요한 batch 개수"""
    return self.__LEN
  
  def __getitem__(self, index:int):
    """index에 해당하는 batch 데이터를 로드"""
    batch_index = index * self.__BATCH_SIZE
    indexes = self.__INDEXES[batch_index:batch_index + self.__BATCH_SIZE]
    
    return self._data_generation(indexes)

  def _data_generation(self, indexes:list[int]):
    """데이터 로드 및 증강"""
    raise NotImplementedError

  def on_epoch_end(self):
    if self.__SHUFFLE:
        np.random.shuffle(self.__INDEXES)
  