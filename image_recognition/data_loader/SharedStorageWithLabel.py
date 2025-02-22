import copy
import numpy as np
from multiprocessing.shared_memory import SharedMemory
from data_loader.Setting import Setting
from data_loader.SharedStorage import SharedStorage

class SharedStorageWithLabel(SharedStorage):
  BUFFER_LABEL_LIST_SHM_NAME = "buffer_label_list_shm"
  BUFFER_IMAGE_LABEL_LIST_SHM_NAME = "buffer_image_label_list_shm"

  def __init__(self, name: str, setting:Setting):
    super(SharedStorageWithLabel, self).__init__(name, setting)

    self.__BUFFER_LABEL_LIST_SHM_NAME = f"{self._NAME}_{SharedStorageWithLabel.BUFFER_LABEL_LIST_SHM_NAME}"
    self.__BUFFER_IMAGE_LABEL_LIST_SHM_NAME = f"{self._NAME}_{SharedStorageWithLabel.BUFFER_IMAGE_LABEL_LIST_SHM_NAME}"

  def _allocate_shared_memory(self):
    super()._allocate_shared_memory()

    self.__buffer_label_list_shm = SharedMemory(name=self.__BUFFER_LABEL_LIST_SHM_NAME)
    self.__buffer_image_label_list_shm = SharedMemory(name=self.__BUFFER_IMAGE_LABEL_LIST_SHM_NAME)

  def _allocate_ary(self):
    super()._allocate_ary()

    self.__buffer_label_list_ary = np.ndarray(buffer=self.__buffer_label_list_shm.buf, shape=(self._PATH_BUFFER_SIZE,), dtype=self._SIZE_OF_LABEL)
    self.__buffer_image_label_list_ary = np.ndarray(buffer=self.__buffer_image_label_list_shm.buf, shape=(self._IMAGE_BUFFER_SIZE,), dtype=self._SIZE_OF_LABEL)

  def _create_shared_memory(self):
    super()._create_shared_memory()

    buffer_label_list_shm_size = np.dtype(self._SIZE_OF_LABEL).itemsize * self._PATH_BUFFER_SIZE
    buffer_image_label_list_shm_size = np.dtype(self._SIZE_OF_LABEL).itemsize * self._IMAGE_BUFFER_SIZE

    self.__buffer_label_list_shm = SharedMemory(name=self.__BUFFER_LABEL_LIST_SHM_NAME, create=True, size=buffer_label_list_shm_size)
    self.__buffer_image_label_list_shm = SharedMemory(name=self.__BUFFER_IMAGE_LABEL_LIST_SHM_NAME, create=True, size=buffer_image_label_list_shm_size)

  def _init_shm(self):
    super()._init_shm()

    self.__buffer_label_list_ary[:] = 0
    self.__buffer_image_label_list_ary[:] = 0

  def close(self):
    super().close()

    self.__buffer_label_list_shm.close()
    self.__buffer_image_label_list_shm.close()

  def unlink(self):
    super().unlink()

    if not self.isCopy():
      self.__buffer_label_list_shm.unlink()
      self.__buffer_image_label_list_shm.unlink()
  
  @property
  def buffer_label_list_shm(self):
    return self.__buffer_label_list_shm
  
  @property
  def buffer_image_label_list_shm(self):
    return self.__buffer_image_label_list_shm
  
  @property
  def buffer_label_list_ary(self):
    return self.__buffer_label_list_ary
  
  @property
  def buffer_image_label_list_ary(self):
    return self.__buffer_image_label_list_ary

def test():
  setting = Setting((244, 244, 3), 100, 100)
  storage = SharedStorageWithLabel("test", setting)
  storage_copy = copy.copy(storage)
  storage_copy.init()
  
  storage.init()
  print("🟩 SharedStorageWithLabel test passed")

  storage_copy = copy.copy(storage)
  storage_copy.init()
  print("🟩 SharedStorageWithLabel copy test passed")

  temp = storage.buffer_label_list_ary
  temp = storage.buffer_image_label_list_ary

  storage.close()
  storage.unlink()
  storage_copy.close()
  print("🟩 SharedStorageWithLabel test passed")

if __name__ == "__main__":
  test()