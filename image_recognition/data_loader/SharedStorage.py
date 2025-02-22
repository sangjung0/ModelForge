import copy
import numpy as np
from multiprocessing.shared_memory import SharedMemory
import multiprocessing

from data_loader.Setting import Setting
from data_loader.Env import Env

class SharedStorage:
  # FLAG_SHM_NAME = "flag_shm"
  PATH_INDEX_SHM_NAME = "path_index_shm"
  LAST_PATH_INDEX_SHM_NAME = "last_path_index_shm"
  BUFFER_PATH_LIST_SHM_NAME = "buffer_path_list_shm"
  INDEX_SHM_NAME = "index_shm"
  BUFFER_IMAGE_LIST_SHM_NAME = "buffer_image_list_shm"
  BUFFER_STATUS_SHM_NAME = "buffer_status_shm"

  def __init__(self, name:str, setting:Setting):
    self._NAME = name
    self._SIZE_OF_PATH = setting.size_of_path
    self._MAX_PATH_LENGTH = setting.max_path_length
    self._SIZE_OF_LABEL = setting.size_of_label
    self._PATH_BUFFER_SIZE = setting.path_buffer_size
    self._IMAGE_BUFFER_SIZE = setting.image_buffer_size
    self._SHAPE = setting.shape

    self.__isCopy = False
    self.__number_of_processor = setting.number_of_processor
    self.__SETTING = setting

    # self.__FLAG_SHM_NAME = f"{self._NAME}_{SharedStorage.FLAG_SHM_NAME}"
    self.__PATH_INDEX_SHM_NAME = f"{self._NAME}_{SharedStorage.PATH_INDEX_SHM_NAME}"
    self.__LAST_PATH_INDEX_SHM_NAME = f"{self._NAME}_{SharedStorage.LAST_PATH_INDEX_SHM_NAME}"
    self.__BUFFER_PATH_LIST_SHM_NAME = f"{self._NAME}_{SharedStorage.BUFFER_PATH_LIST_SHM_NAME}"
    self.__INDEX_SHM_NAME = f"{self._NAME}_{SharedStorage.INDEX_SHM_NAME}"
    self.__BUFFER_IMAGE_LIST_SHM_NAME = f"{self._NAME}_{SharedStorage.BUFFER_IMAGE_LIST_SHM_NAME}"
    self.__BUFFER_STATUS_SHM_NAME = f"{self._NAME}_{SharedStorage.BUFFER_STATUS_SHM_NAME}"

  def __copy__(self):
    new = SharedStorage.__new__(SharedStorage)
    new.__init__(self._NAME, self.__SETTING)
    new.__isCopy = True
    new.__path_lock = self.__path_lock
    new.__index_lock = self.__index_lock
    # new.__flag_lock = self.__flag_lock
    return new

  def init(self):
    try:
      if self.__isCopy:
        self._allocate_shared_memory()
        self._allocate_ary()
        return
      self._create_shared_memory()
      self._allocate_ary()
      self._init_shm()
    except Exception as e:
      self.close()
      self.unlink()
      raise e

  def _allocate_shared_memory(self):
    # self.__flag_shm = SharedMemory(name=self.__FLAG_SHM_NAME)
    self.__path_index_shm = SharedMemory(name=self.__PATH_INDEX_SHM_NAME)
    self.__last_path_index_shm = SharedMemory(name=self.__LAST_PATH_INDEX_SHM_NAME)
    self.__buffer_path_list_shm = SharedMemory(name=self.__BUFFER_PATH_LIST_SHM_NAME)
    self.__index_shm = SharedMemory(name=self.__INDEX_SHM_NAME)
    self.__buffer_image_list_shm = SharedMemory(name=self.__BUFFER_IMAGE_LIST_SHM_NAME)
    self.__buffer_status_shm = SharedMemory(name=self.__BUFFER_STATUS_SHM_NAME)
    
  def _allocate_ary(self):
    # self.__flag_ary = np.ndarray(buffer=self.__flag_shm.buf[0:INT_SIZE], shape=(4,), dtype=np.uint8)
    # self.__processor_ary = np.ndarray(buffer=self.__flag_shm.buf[INT_SIZE:INT_SIZE*3], shape=(2,), dtype=np.uint32)
    self.__path_index_ary = np.ndarray(buffer=self.__path_index_shm.buf, shape=(1,), dtype=self._SIZE_OF_PATH)
    self.__last_path_index_ary = np.ndarray(buffer=self.__last_path_index_shm.buf, shape=(1,), dtype=self._SIZE_OF_PATH)
    self.__buffer_path_list_ary = np.ndarray(buffer=self.__buffer_path_list_shm.buf, shape=(self._PATH_BUFFER_SIZE,), dtype=f'S{self._MAX_PATH_LENGTH}')
    self.__index_ary = np.ndarray(buffer=self.__index_shm.buf, shape=(1,), dtype=np.uint32)
    self.__buffer_image_list_ary = np.ndarray(buffer=self.__buffer_image_list_shm.buf, shape=(self._IMAGE_BUFFER_SIZE, *self._SHAPE), dtype=np.uint8)
    self.__buffer_status_ary = np.ndarray(buffer=self.__buffer_status_shm.buf, shape=(self._IMAGE_BUFFER_SIZE* 2,), dtype=np.uint8)

  def _create_shared_memory(self):
    # flag_shm_size = INT_SIZE*3 # flag[terminate, pause, sync](4), processor counter (4), current processor (4)
    path_index_shm_size = np.dtype(self._SIZE_OF_PATH).itemsize
    buffer_path_list_shm_size = self._MAX_PATH_LENGTH* self._PATH_BUFFER_SIZE
    last_path_index_shm_size = np.dtype(self._SIZE_OF_PATH).itemsize
    index_shm_size = Env.INT_SIZE.value
    buffer_image_list_shm_size = np.prod(self._SHAPE) * np.dtype(np.uint8).itemsize * self._IMAGE_BUFFER_SIZE
    buffer_status_shm_size = self._IMAGE_BUFFER_SIZE * 2
    
    # self.__flag_shm = SharedMemory(name=self.__FLAG_SHM_NAME, create=True, size=flag_shm_size)
    self.__path_index_shm = SharedMemory(name=self.__PATH_INDEX_SHM_NAME, create=True, size=path_index_shm_size)
    self.__last_path_index_shm = SharedMemory(name=self.__LAST_PATH_INDEX_SHM_NAME, create=True, size=last_path_index_shm_size)
    self.__buffer_path_list_shm = SharedMemory(name=self.__BUFFER_PATH_LIST_SHM_NAME, create=True, size=buffer_path_list_shm_size)
    self.__index_shm = SharedMemory(name=self.__INDEX_SHM_NAME, create=True, size=index_shm_size)
    self.__buffer_image_list_shm = SharedMemory(name=self.__BUFFER_IMAGE_LIST_SHM_NAME, create=True, size=buffer_image_list_shm_size)
    self.__buffer_status_shm = SharedMemory(name=self.__BUFFER_STATUS_SHM_NAME, create=True, size=buffer_status_shm_size)

    self.__barrier = multiprocessing.Barrier(self.__number_of_processor)
    self.__manager = multiprocessing.Manager()

    self.__path_lock = self.__manager.Lock()
    self.__index_lock = self.__manager.Lock()
    # self.__flag_lock = self.__manager.Lock()

    self.__kill_event = self.__manager.Event()
    self.__pause_event = self.__manager.Event()
    self.__resume_event = self.__manager.Event()
      
  def _init_shm(self):
    # self.__flag_shm.buf[:] = 0
    self.__path_index_ary[:] = 0
    self.__last_path_index_ary[:] = 0
    self.__buffer_path_list_ary[:] = 0
    self.__index_ary[:] = 0
    # self.__buffer_image_list_ary[:] = 0
    self.__buffer_status_ary[:] = 0
    
  def isCopy(self):
    return self.__isCopy
      
  def close(self):
    # self.__flag_shm.close()
    self.__path_index_shm.close()
    self.__last_path_index_shm.close()
    self.__buffer_path_list_shm.close()
    self.__index_shm.close()
    self.__buffer_image_list_shm.close()
    self.__buffer_status_shm.close()

  def unlink(self):
    if not self.__isCopy:
      # self.__flag_shm.unlink()
      self.__path_index_shm.unlink()
      self.__last_path_index_shm.unlink()
      self.__buffer_path_list_shm.unlink()
      self.__index_shm.unlink()
      self.__buffer_image_list_shm.unlink()
      self.__buffer_status_shm.unlink
      self.__manager.shutdown()
      self.__manager.join()()
    
  # @property
  # def flag_shm(self):
  #   return self.__flag_shm

  @property
  def path_index_shm(self):
    return self.__path_index_shm
  
  @property
  def last_path_index_shm(self):
    return self.__last_path_index_shm
  
  @property
  def buffer_path_list_shm(self):
    return self.__buffer_path_list_shm
  
  @property
  def index_shm(self):
    return self.__index_shm
  
  @property
  def buffer_image_list_shm(self):
    return self.__buffer_image_list_shm
  
  @property
  def buffer_status_shm(self):
    return self.__buffer_status_shm

  # @property
  # def flag_ary(self):
  #   return self.__flag_ary

  # @property
  # def processor_ary(self):
  #   return self.__processor_ary

  @property
  def path_index_ary(self):
    return self.__path_index_ary
  
  @property
  def last_path_index_ary(self):
    return self.__last_path_index_ary

  @property
  def buffer_path_list_ary(self):
    return self.__buffer_path_list_ary
  
  @property
  def index_ary(self):
    return self.__index_ary
  
  @property
  def buffer_image_list_ary(self):
    return self.__buffer_image_list_ary
  
  @property
  def buffer_status_ary(self):
    return self.__buffer_status_ary

  @property
  def index_lock(self):
    return self.__index_lock

  @property
  def path_lock(self):
    return self.__path_lock
  
  # @property
  # def flag_lock(self):
  #   return self.__flag_lock
  
  @property
  def barrier(self):
    return self.__barrier
  
  @property
  def kill_event(self):
    return self.__kill_event
  
  @property
  def pause_event(self):
    return self.__pause_event
  
  @property
  def resume_event(self):
    return self.__resume_event
  
def test():
  setting = Setting((244, 244, 3), 100, 100)
  storage = SharedStorage("test", setting)
  storage_copy = copy.copy(storage)
  storage_copy.init()
  
  storage.init()
  print("🟩 SharedStorage test passed")

  storage_copy = copy.copy(storage)
  storage_copy.init()
  print("🟩 SharedStorage copy test passed")

  # temp = storage.flag_ary
  # temp = storage.processor_ary
  temp = storage.path_index_ary
  temp = storage.last_path_index_ary
  temp = storage.buffer_path_list_ary
  temp = storage.index_ary
  temp = storage.buffer_image_list_ary
  temp = storage.buffer_status_ary

  storage.close()
  storage.unlink()
  storage_copy.close()
  print("🟩 SharedStorage test done")

if __name__ == "__main__":
  test()