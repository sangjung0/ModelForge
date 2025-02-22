import numpy as np
import time
from tensorflow.keras.utils import Sequence
from multiprocessing.shared_memory import SharedMemory

from data_loader import Setting, SharedStorage
from data_loader.Env import Env

class Consumer(Sequence):
  def __init__(
      self, 
      setting:Setting,
      sharedStorage:SharedStorage,
    ):

    self._SHARED_STORAGE = sharedStorage
    self._BATCH_SIZE = setting.batch_size

    self.__LEN = int(np.floor(setting.number_of_image / self._BATCH_SIZE))
    self.__IMAGE_BUFFER_SIZE = setting.image_buffer_size
    
    self._SHARED_STORAGE.init()
    self._allocate_ary()
    self.__allocate_event()

    self._current_index = None
    self.__init_process()
    
  def __init_process(self):
    self.__kill_event.clear()
    self.__resume_event.clear()
    self.__bairrer.wait()
    self.__resume_event.set()

  def close(self):
    if self.__kill_event:
      self.__kill_event.set()

  def __len__(self):
    return self.__LEN
  
  def _allocate_ary(self):
    # self.__flag_ary = self.__SHARED_STORAGE.flag_ary
    # self.__processor_ary = self.__SHARED_STORAGE.processor_ary
    self.__path_index_ary = self._SHARED_STORAGE.path_index_ary
    self.__last_path_index_ary = self._SHARED_STORAGE.last_path_index_ary
    self.__index_ary = self._SHARED_STORAGE.index_ary
    self.__buffer_image_list_ary = self._SHARED_STORAGE.buffer_image_list_ary
    self.__buffer_status_ary = self._SHARED_STORAGE.buffer_status_ary
  
  def __allocate_event(self):
    self.__kill_event = self._SHARED_STORAGE.kill_event
    self.__resume_event = self._SHARED_STORAGE.resume_event
    self.__bairrer = self._SHARED_STORAGE.barrier
  
  def __clear_prev_item(self):
    if self._current_index is None: 
      self._current_index = 0
      return
    
    for check_index in range(self._current_index, self._current_index+self._BATCH_SIZE):
      if check_index >= self.__IMAGE_BUFFER_SIZE:
        check_index = check_index % self.__IMAGE_BUFFER_SIZE
        
      scaled_index = check_index * 2
      self.__buffer_status_ary[scaled_index+1] = 0
    
    self._current_index += 1

  def __restart_provider(self, image_index:int):
    self.__resume_event.clear()
    self.__bairrer.reset()
    self.__bairrer.wait()
  
    self.__path_index_ary[0] = image_index
    self.__last_path_index_ary[0] = image_index
    self.__index_ary[0] = 0

    self.__buffer_status_ary.fill(0)
    self.__buffer_status_ary.fill(0)

    self.__resume_event.set()

  def _getitem(self, index:int):
    return self.__buffer_image_list_ary[index:index+self._BATCH_SIZE]    

  def __getitem__(self, index:int):
    if self._current_index < 0:
      raise Exception("이미지 불러오기가 종료되었습니다.")
    self.__clear_prev_item()

    if index != self._current_index:
      self.__restart_provider(index*self._BATCH_SIZE)

    buffer_index = index * self._BATCH_SIZE % self.__IMAGE_BUFFER_SIZE

    for check_index in range(buffer_index, buffer_index+self._BATCH_SIZE):
      if check_index >= self.__IMAGE_BUFFER_SIZE:
        check_index = check_index % self.__IMAGE_BUFFER_SIZE

      sleep_time = Env.MIN_SLEEP_TIME.value
      while self.__buffer_status_ary[check_index*2] == 0 or self.__buffer_status_ary[check_index*2+1] == 0:
        time.sleep(sleep_time)
        sleep_time += sleep_time
        if sleep_time > Env.MAX_SLEEP_TIME.value: sleep_time = Env.MAX_SLEEP_TIME.value
      
      self.__buffer_status_ary[check_index*2] = 0
    
    return self._getitem(buffer_index)