import numpy as np
import time

from data_loader.PathProvider import PathProvider
from data_loader import SharedStorage, Worker
from data_loader.Setting import Setting
from data_loader.Env import Env

class Provider:
  def __init__(
      self, 
      setting:Setting,
      sharedStorage:SharedStorage,
      pathProvider:PathProvider,
      BaseWorker,
    ):
    if not issubclass(BaseWorker, Worker):
      raise Exception("🟥 BaseWorker는 Worker의 서브클래스여야 합니다.")

    self._PATH_BUFFER_SIZE = setting.path_buffer_size
    self._SHARED_STORAGE = sharedStorage
    self._worker = BaseWorker()
    
    self._IMAGE_REQUEST_COUNT = self._worker.image_request_count
    self._BUFFER_REQUEST_COUNT = self._worker.buffer_request_count

    self.__PATH_PROVIDER = pathProvider
    
    self._buffers = []
    
  def _allocate_ary(self):
    # self.__flag_ary = self.__SHARED_STORAGE.flag_ary
    # self.__processor_ary = self.__SHARED_STORAGE.processor_ary
    self.__path_index_ary = self._SHARED_STORAGE.path_index_ary
    self.__last_path_index_ary = self._SHARED_STORAGE.last_path_index_ary
    self.__index_ary = self._SHARED_STORAGE.index_ary

    self._buffer_path_list_ary = self._SHARED_STORAGE.buffer_path_list_ary
    self._buffer_image_list_ary = self._SHARED_STORAGE.buffer_image_list_ary
    self._buffer_status_ary = self._SHARED_STORAGE.buffer_status_ary

  def __allocate_lock(self):
    self.__index_lock = self._SHARED_STORAGE.index_lock
    self.__path_lock = self._SHARED_STORAGE.path_lock
    # self.__flag_lock = self.__SHARED_STORAGE.flag_lock

  def __allocate_event(self):
    self.__kill_event = self._SHARED_STORAGE.kill_event
    self.__pause_event = self._SHARED_STORAGE.pause_event
    self.__resume_event = self._SHARED_STORAGE.resume_event
    self.__bairrer = self._SHARED_STORAGE.barrier
    
  def _update(self, index:int, new_data: list[str]):
    self._buffer_path_list_ary.fill(0)
    for path in enumerate(new_data, start = index):
      index = index % self._PATH_BUFFER_SIZE
      self._buffer_path_list_ary[index] = path.encode('utf-8')
    
  def _get_source(self, index:int):
    return self._buffer_path_list_ary[index].rstrip(b'\x00').decode('utf-8')

  def __get_source(self, count:int):
    sources = []
    
    with self.__path_lock:
      current_index = self.__path_index_ary[0]
      last_index = self.__last_path_index_ary[0]

      for index in range(current_index, current_index+count):
        if index == last_index:
          new_paths = self.__PATH_PROVIDER.get_paths(current_index, self._PATH_BUFFER_SIZE)
          if len(new_paths) == 0:
            self.__path_index_ary[0] = index
            return sources
          self._update(index, new_paths)
        sources.append(self._get_source(index))

      self.__path_index_ary[0] = current_index + count

    return sources
    
  def __get_buffers(self, buffer_count:int):
    buffers = []

    with self.__index_lock:
      current_index = self.__index_ary[0]
    
      while len(buffers) < buffer_count:
        if current_index == self._buffer_image_list_ary.shape[0]:
          current_index = 0
        
        sleep_time = Env.MIN_SLEEP_TIME.value
        scale_index = current_index * 2
        while self._buffer_status_ary[scale_index] == 1 or self._buffer_status_ary[scale_index+1] == 1:
          time.sleep(sleep_time)
          sleep_time += sleep_time
          if sleep_time > Env.MAX_SLEEP_TIME.value: sleep_time = Env.MAX_SLEEP_TIME.value

        self._buffer_status_ary[scale_index] = 1
        buffers.append(current_index)
        current_index += 1

      self.__index_ary[0] = current_index

    return buffers
    
  def __get_source_and_buffer(self):
    if len(self._buffers) != 0:
      raise Exception("🟥 이미 버퍼를 가지고 있습니다.")

    source_size = self._IMAGE_REQUEST_COUNT
    buffer_size = self._BUFFER_REQUEST_COUNT

    sources = self.__get_source(source_size)
    buffer_size = round(buffer_size/source_size * len(sources))
    buffers = self.__get_buffers(buffer_size)        
          
    self._buffers = buffers.copy()
    return sources, buffers

  def _set_dest(self, dest:np.ndarray, buffer_index:int):
    if buffer_index not in self._buffers:
      raise Exception("🟥 해당 버퍼에 이미지를 넣을 수 없습니다.")
    
    self._buffer_image_list_ary[buffer_index] = dest
    self._buffer_status_ary[buffer_index*2+1] = 1
    self._buffers.remove(buffer_index)

  def runner(self):
    self._SHARED_STORAGE.init()
    self._allocate_ary()
    self.__allocate_lock()
    self.__allocate_event()
    
    self.__bairrer.wait()
    self.__resume_event.wait()

    while True:
      if self.__kill_event.is_set():
        break
      elif self.__pause_event.is_set():
        self.__pause_event.wait()
      elif not self.__bairrer.broken:
        self.__bairrer.wait()
        self.__resume_event.wait()
      else:
        sources, buffers = self.__get_source_and_buffer()
        dests = self._worker.work(sources, len(buffers))
        for dest, buffer_index in zip(dests, buffers):
          self._set_dest(dest, buffer_index)
          if self.__kill_event.is_set():
            break
    self.close()
        
  def close(self):
    self._SHARED_STORAGE.close()
