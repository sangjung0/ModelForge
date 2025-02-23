import numpy as np

from data_loader.Consumer import Consumer
from data_loader.Setting import Setting
from data_loader.SharedStorageWithLabel import SharedStorageWithLabel

class ConsumerWithLabel(Consumer):
  def __init__(
    self,
    setting:Setting,
    sharedStorage:SharedStorageWithLabel,
  ):
    super(ConsumerWithLabel, self).__init__(setting, sharedStorage)

  def _allocate_ary(self):
    super()._allocate_ary()
    self._buffer_image_label_list_ary = self._SHARED_STORAGE.buffer_image_label_list_ary

  def _getitem(self, index:int):
    return np.hstack(
      super()._getitem(index),
      self._buffer_image_label_list_ary[index:index+self._BATCH_SIZE]
    )