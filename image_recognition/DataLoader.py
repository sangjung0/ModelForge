from tensorflow.keras.utils import Sequence
import numpy as np
import cv2
from multiprocessing.shared_memory import SharedMemory
import multiprocessing
import time

BUFFER = 1000
INT_SIZE = 4
MIN_SLEEP_TIME = 0.1
MAX_SLEEP_TIME = 1

class SharedStorage:
  __FLAG_SHM = 'flag_shm'
  __CURRENT_IMAGE_INDEX_SHM = 'current_image_index_shm'
  __IMAGE_PATH_LIST_SHM = 'image_path_list_shm'
  __IMAGE_LABEL_LIST_SHM = 'image_label_list_shm'
  __LABEL_LIST_SHM = 'label_list_shm'
  __BUFFER_STATUS_SHM = 'buffer_status_shm'
  __BUFFER_INDEX_SHM = 'buffer_index_shm'
  __BUFFER_IMAGE_LIST_SHM = 'buffer_image_list_shm'
  __BUFFER_LABEL_LIST_SHM = 'buffer_label_list_shm'

  def __init__(self, image_paths:list[str], image_labels:list[int], label_list:list[str], shape:tuple, buffer_size:int):
    self._image_paths = [path.encode('utf-8') for path in image_paths]
    self._image_labels = image_labels
    self._label_list = [label.encode('utf-8') for label in label_list]

    self.__SHAPE = shape
    self.__MAX_PATH_LENGTH = max(self._image_paths, key=len)
    # self.__MAX_PATH_LENGTH = max(map(len, self._image_paths))
    self.__MAX_LABEL_LENGTH = max(self._label_list, key=len)
    self.__BUFFER_SIZE = buffer_size 

    self.__create_shared_memory()
    self.__init_flag()
    self.__init_current_index()
    self.__store_image_paths_in_IPLS()
    self.__store_image_labels_in_ILLS()
    self.__store_labels_in_LLS()
    self.__init_buffer_status()

  def __create_shared_memory(self):
    flag_shm_size = INT_SIZE*3
    current_image_index_shm_size = INT_SIZE
    image_path_list_shm_size = len(self._image_paths) * self.__MAX_PATH_LENGTH
    image_label_list_shm_size = len(self._image_labels) * INT_SIZE
    label_list_shm_size = len(self._label_list) * self.__MAX_LABEL_LENGTH
    buffer_status_shm_size = self.__BUFFER_SIZE * 2
    buffer_index_shm_size = INT_SIZE
    buffer_image_list_shm_size = np.prod(self.__SHAPE) * np.dtype(np.uint8).itemsize * self.__BUFFER_SIZE
    buffer_label_list_shm_size = INT_SIZE * self.__BUFFER_SIZE

    self._flag_shm = SharedMemory(name=self.__FLAG_SHM, create=True, size=flag_shm_size)
    self._current_image_index_shm = SharedMemory(name=self.__CURRENT_IMAGE_INDEX_SHM, create=True, size=current_image_index_shm_size)
    self._image_path_list_shm = SharedMemory(name=self.__IMAGE_PATH_LIST_SHM, create=True, size=image_path_list_shm_size)
    self._image_label_list_shm= SharedMemory(name=self.__IMAGE_LABEL_LIST_SHM, create=True, size=image_label_list_shm_size)
    self._label_list_shm = SharedMemory(name=self.__LABEL_LIST_SHM, create=True, size=label_list_shm_size)
    self._buffer_status_shm = SharedMemory(name=self.__BUFFER_STATUS_SHM, create=True, size=buffer_status_shm_size)
    self._buffer_index_shm = SharedMemory(name=self.__BUFFER_INDEX_SHM, create=True, size=buffer_index_shm_size)
    self._buffer_image_list_shm = SharedMemory(name=self.__BUFFER_IMAGE_LIST_SHM, create=True, size=buffer_image_list_shm_size)
    self._buffer_label_list_shm = SharedMemory(name=self.__BUFFER_LABEL_LIST_SHM, create=True, size=buffer_label_list_shm_size)
    
    self._index_lock = multiprocessing.Lock()
      
  def __init_flag(self):
    self._flag_shm.buf[:] = 0
      
  def __init_current_index(self):
    self._current_image_index_shm.buf[:] = 0

  def __store_image_paths_in_IPLS(self):
    self._image_path_list_shm.buf[:] = 0
    for i, path in enumerate(self._image_paths):
      self._image_path_list_shm.buf[i * self.__MAX_PATH_LENGTH:(i + 1) * self.__MAX_PATH_LENGTH] = path

  def __store_image_labels_in_ILLS(self):
    self._image_label_list_shm.buf[:] = 0
    for i, label in enumerate(self._image_labels):
      self._image_label_list_shm.buf[i*INT_SIZE:(i+1)*INT_SIZE] = label.to_bytes(INT_SIZE, byteorder='little')
      
  def __store_labels_in_LLS(self):
    self._label_list_shm.buf[:] = 0
    for i, label in enumerate(self._label_list):
      self._label_list_shm.buf[i * self.__MAX_LABEL_LENGTH:(i + 1) * self.__MAX_LABEL_LENGTH] = label

  def __init_buffer_status(self):
    self._buffer_status_shm.buf[:] = 0
    self._buffer_index_shm.buf[:] = 0
      
  def close(self):
    self._flag_shm.close()
    self._current_image_index_shm.close()
    self._image_path_list_shm.close()
    self._image_label_list_shm.close()
    self._label_list_shm.close()
    self._buffer_status_shm.close()
    self._buffer_index_shm.close()
    self._buffer_image_list_shm.close()
    self._buffer_label_list_shm.close()

  def unlink(self):
    self._flag_shm.unlink()
    self._current_image_index_shm.unlink()
    self._image_path_list_shm.unlink()
    self._image_label_list_shm.unlink()
    self._label_list_shm.unlink()
    self._buffer_status_shm.unlink()
    self._buffer_index_shm.unlink()
    self._buffer_image_list_shm.unlink()
    self._buffer_label_list_shm.unlink()

  @property
  def flag_shm(self):
    return self._flag_shm

  @property
  def current_image_index_shm(self):
    return self._current_image_index_shm
  
  @property
  def image_path_list_shm(self):
    return self._image_path_list_shm
  
  @property
  def image_label_list_shm(self):
    return self._image_label_list_shm
  
  @property
  def label_list_shm(self):
    return self._label_list_shm
  
  @property
  def buffer_status_shm(self):
    return self._buffer_status_shm
  
  @property
  def buffer_index_shm(self):
    return self._buffer_index_shm
  
  @property
  def buffer_image_list_shm(self):
    return self._buffer_image_list_shm
  
  @property
  def buffer_label_list_shm(self):
    return self._buffer_label_list_shm
  
  @property
  def index_lock(self):
    return self._index_lock

  @property
  def max_path_length(self):
    return self.__MAX_PATH_LENGTH

  @property
  def max_label_length(self):
    return self.__MAX_LABEL_LENGTH

class Worker:
  def work(self, paths:list[str], count:int):
    images = [cv2.imread(path) for path in paths]
    return images

class Provider:
  def __init__(
      self, 
      flag_shm_name:str,
      CIIS_name:str,
      IPLS_name:str,
      ILLS_name:str,
      # LLS_name:str,
      BSS_name:str,
      BIS_name:str,
      BILS_name:str,
      BLLS_name:str,
      index_lock,
      path_length:int, 
      # label_length:int, 
      number_of_image:int,
      shape:tuple, 
      buffer_size:int,
      BaseWorker
    ):
    if not issubclass(BaseWorker, Worker):
      raise Exception("BaseWorker는 Worker의 서브클래스여야 합니다.")
    
    self.__PATH_LENGTH = path_length
    # self.__LABEL_LENGTH = label_length
    # self.__SHAPE = shape
    self.__BUFFER_SIZE = buffer_size
    self.__NUMBER_OF_IMAGE = number_of_image
    self.__IMAGE_SIZE = np.prod(shape) * np.dtype(np.uint8).itemsize

    self._flag_shm = SharedMemory(name=flag_shm_name)
    self._current_image_index_shm = SharedMemory(name=CIIS_name)
    self._image_path_list_shm = SharedMemory(name=IPLS_name)
    self._image_label_list_shm = SharedMemory(name=ILLS_name)
    # self._label_list_shm = SharedMemory(name=LLS_name)
    self._buffer_status_shm = SharedMemory(name=BSS_name)
    self._buffer_index_shm = SharedMemory(name=BIS_name)
    self._buffer_image_list_shm = SharedMemory(name=BILS_name)
    self._buffer_label_list_shm = SharedMemory(name=BLLS_name)
    
    self.__index_lock = index_lock

    self._worker = BaseWorker()
    
    self.buffers = []
    
    with self.__index_lock:
      temp = int.from_bytes(self._flag_shm.buf[8:12], byteorder='little') + 1
      self._flag_shm.buf[8:12] = temp.to_bytes(INT_SIZE, byteorder='little')
    
  def _getPathIndex(self, image_count:int):
    """
    반드시 lock과 함께 사용
    """
    path_indeces = []
    
    current_index = int.from_bytes(self._current_image_index_shm.buf, byteorder='little')

    for index in range(current_index, current_index+image_count):
      if index == len(self.__NUMBER_OF_IMAGE):
        self._current_image_index_shm.buf[0:INT_SIZE] = index.to_bytes(INT_SIZE, byteorder='little')
        return path_indeces
      path_indeces.append(index)
    
    self._current_image_index_shm.buf[0:INT_SIZE] = (current_index + image_count).to_bytes(INT_SIZE, byteorder='little')
    return path_indeces

  def _getBuffers(self, buffer_count:int):
    """
    반드시 lock과 함께 사용
    """
    buffers = []

    buffer_index = int.from_bytes(self._buffer_index_shm.buf, byteorder='little')
    
    while len(buffers) < buffer_count:
      if buffer_index == self.__BUFFER_SIZE:
        buffer_index = 0
      
      sleep_time = MIN_SLEEP_TIME
      while self._buffer_status_shm.buf[buffer_index*2] == 1 or self._buffer_status_shm.buf[buffer_index*2+1] == 1:
        time.sleep(sleep_time)
        sleep_time += sleep_time
        if sleep_time > MAX_SLEEP_TIME: sleep_time = MAX_SLEEP_TIME

      self._buffer_status_shm.buf[buffer_index*2] = 1
      buffers.append(buffer_index)
      buffer_index += 1
    self._buffer_index_shm.buf[0:INT_SIZE] = buffer_index.to_bytes(INT_SIZE, byteorder='little')
    return buffers
    
  def getPathAndBuffer(self, image_count:int=1, buffer_count:int=1):
    if len(self.buffers) != 0:
      raise Exception("이미 버퍼를 가지고 있습니다.")

    buffers=[]
    path_indeces=[]
    
    with self.__index_lock:
      path_indeces = self._getPathIndex(image_count)
      buffer_count = round(buffer_count/len(path_indeces))
      buffers = self._getBuffers(buffer_count)        
          
    paths = [
      self._image_path_list_shm.buf[index * self.__PATH_LENGTH:(index + 1) * self.__PATH_LENGTH].rstrip(b'\x00').decode('utf-8')
      for index in path_indeces
    ]
    labels = [
      int.from_bytes(self._image_label_list_shm.buf[index*INT_SIZE:(index+1)*INT_SIZE], byteorder='little')
      for index in path_indeces
    ]
    self.buffers = buffers
    return paths, labels, buffers

  def setImage(self, image:np.ndarray, label:int, buffer_index:int):
    if buffer_index not in self.buffers:
      raise Exception("해당 버퍼에 이미지를 넣을 수 없습니다.")
    
    self._buffer_image_list_shm.buf[buffer_index * self.__IMAGE_SIZE:(buffer_index + 1) * self.__IMAGE_SIZE] = image.tobytes()
    self._buffer_label_list_shm.buf[buffer_index*INT_SIZE:(buffer_index+1)*INT_SIZE] = label.to_bytes(INT_SIZE, byteorder='little')
    self._buffer_status_shm.buf[buffer_index*2+1] = 1
    self.buffers.remove(buffer_index)
  
  def runner(self):
    while True:
      if self._flag_shm.buf[0] == 1:
        break
      elif self._flag_shm.buf[1] == 1:
        sleep_time = MIN_SLEEP_TIME
        while self._flag_shm.buf[1] == 1:
          time.sleep(sleep_time)
          sleep_time += sleep_time
          if sleep_time > MAX_SLEEP_TIME: sleep_time = MAX_SLEEP_TIME
      elif self._flag_shm.buf[2] == 1:
        with self.__index_lock:
          number_of_pause_processor = int.from_bytes(self._flag_shm.buf[4:8], byteorder='little')
          self._flag_shm.buf[4:8] = (number_of_pause_processor+1).to_bytes(INT_SIZE, byteorder='little')
        while self._flag_shm.buf[2] == 1:
          time.sleep(MIN_SLEEP_TIME)
      else:
        paths, labels, buffers = self.getPathAndBuffer()
        images = self._worker.work(paths, labels, len(buffers))
        for image, label, buffer in zip(images, labels, buffers):
          self.setImage(image, label, buffer)
          if self._flag_shm.buf[2] == 1:
            break
    self.close()
        
  def close(self):
    self._flag_shm.close()
    self._current_image_index_shm.close()
    self._image_path_list_shm.close()
    self._image_label_list_shm.close()
    # self._label_list_shm.close()
    self._buffer_status_shm.close()
    self._buffer_index_shm.close()
    self._buffer_image_list_shm.close()
    self._buffer_label_list_shm.close()

class Consumer(Sequence):
  def __init__(
      self, 
      flag_shm_name:str,
      CIIS_name:str,
      # IPLS_name:str,
      # ILLS_name:str,
      LLS_name:str,
      BSS_name:str,
      BIS_name:str,
      BILS_name:str,
      BLLS_name:str,
      index_lock,
      # path_length:int, 
      # label_length:int, 
      number_of_image:int,
      shape:tuple, 
      buffer_size:int,
      batch_size:int
    ):
    
    # self.__PATH_LENGTH = path_length
    # self.__LABEL_LENGTH = label_length
    self.__SHAPE = shape
    self.__BUFFER_SIZE = buffer_size
    self.__BATCH_SIZE = batch_size
    self.__NUMBER_OF_IMAGE = number_of_image
    self.__IMAGE_SIZE = np.prod(shape) * np.dtype(np.uint8).itemsize

    self._flag_shm = SharedMemory(name=flag_shm_name)
    self._current_image_index_shm = SharedMemory(name=CIIS_name)
    # self._image_path_list_shm = SharedMemory(name=IPLS_name)
    # self._image_label_list_shm = SharedMemory(name=ILLS_name)
    self._label_list_shm = SharedMemory(name=LLS_name)
    self._buffer_status_shm = SharedMemory(name=BSS_name)
    self._buffer_index_shm = SharedMemory(name=BIS_name)
    self._buffer_image_list_shm = SharedMemory(name=BILS_name)
    self._buffer_label_list_shm = SharedMemory(name=BLLS_name)
    
    self.__index_lock = index_lock
    
    self._current_index = None

  def __len__(self):
    return int(np.floor(self.__NUMBER_OF_IMAGE / self.__BATCH_SIZE))
  
  def __clear_prev_item(self):
    if self._current_index is None: 
      self._current_index = 0
      return
    
    for check_index in range(self._current_index, self._current_index+self.__BATCH_SIZE):
      if check_index >= self.__NUMBER_OF_IMAGE:
        check_index = check_index % self.__NUMBER_OF_IMAGE
        
      self._buffer_status_shm.buf[check_index*2+1] = 0
    
    self._current_index += 1

  def __restart_provider(self, image_index:int):
    self._flag_shm.buf[2] = 1
    while self._flag_shm.buf[4:8] != self._flag_shm.buf[8:12]:
      time.sleep(MIN_SLEEP_TIME)
  
    with self.__index_lock:
      self._current_image_index_shm.buf[0] = image_index.to_bytes(INT_SIZE, byteorder='little')
      self._buffer_index_shm.buf[0] = (image_index % self.__BUFFER_SIZE).to_bytes(INT_SIZE, byteorder='little')
      for i in range(self.__BUFFER_SIZE):
        self._buffer_status_shm.buf[i*2] = 0
        self._buffer_status_shm.buf[i*2+1] = 0
    self._flag_shm.buf[2] = 0

  def __getitem__(self, index):
    if self._current_index < 0:
      raise Exception("이미지 불러오기가 종료되었습니다.")
    self.__clear_prev_item()

    if index != self._current_index:
      self.__restart_provider(index*self.__BATCH_SIZE)

    buffer_index = index * self.__BATCH_SIZE % self.__BUFFER_SIZE

    for check_index in range(buffer_index, buffer_index+self.__BATCH_SIZE):
      if check_index >= self.__BUFFER_SIZE:
        check_index = check_index % self.__BUFFER_SIZE

      sleep_time = MIN_SLEEP_TIME
      while self._buffer_status_shm.buf[check_index*2] == 0 or self._buffer_status_shm.buf[check_index*2+1] == 0:
        time.sleep(sleep_time)
        sleep_time += sleep_time
        if sleep_time > MAX_SLEEP_TIME: sleep_time = MAX_SLEEP_TIME
      
      self._buffer_status_shm.buf[check_index*2] = 0
        
    images = np.ndarray(
      (self.__BATCH_SIZE, *self.__SHAPE), 
      dtype=np.uint8, 
      buffer=self._buffer_image_list_shm.buf[buffer_index * self.__IMAGE_SIZE:(buffer_index + self.__BATCH_SIZE) * self.__IMAGE_SIZE]
    )
    labels = np.ndarray((self.__BATCH_SIZE,), dtype=np.int32, buffer=self._buffer_label_list_shm.buf[buffer_index*INT_SIZE:(buffer_index+self.__BATCH_SIZE)*INT_SIZE])

    return images, labels
  
  def close(self):
    self._current_index = -1
    self._flag_shm.buf[0] = 1

    self._flag_shm.close()
    self._current_image_index_shm.close()
    # self._image_path_list_shm.close()
    # self._image_label_list_shm.close()
    self._label_list_shm.close()
    self._buffer_status_shm.close()
    self._buffer_index_shm.close()
    self._buffer_image_list_shm.close()
    self._buffer_label_list_shm.close()
      
class SharedMemoryManager:
  def __init__(self, image_paths:list[str], image_labels:list[int], label_list:list[str], shape:tuple, buffer_size:int = BUFFER, batch_size:int = 1):
    self._SAHPE = shape
    self._BUFFER_SIZE = buffer_size
    self._BATCH_SIZE = batch_size
    
    self._image_paths = image_paths
    self._image_labels = image_labels
    self._label_list = label_list
    
    self.__shared_storage = SharedStorage(image_paths, image_labels, label_list, shape, buffer_size)
    self.__consumer = Consumer(
      self.__shared_storage.flag_shm.name,
      self.__shared_storage.current_image_index_shm.name,
      # self.__shared_storage.image_path_list_shm.name,
      # self.__shared_storage.image_label_list_shm.name,
      self.__shared_storage.label_list_shm.name,
      self.__shared_storage.buffer_status_shm.name,
      self.__shared_storage.buffer_index_shm.name,
      self.__shared_storage.buffer_image_list_shm.name,
      self.__shared_storage.buffer_label_list_shm.name,
      self.__shared_storage.index_lock,
      # self.__shared_storage.max_path_length,
      # self.__shared_storage.max_label_length,
      len(self._image_paths),
      self._SAHPE,
      self._BUFFER_SIZE,
      self._BATCH_SIZE
    ) 
    self.__processes = []
    
  def provider_runner(self, *args):
    Provider(*args).runner()

  def start(self, n:int):
    if len(self.__processes) != 0:
      raise Exception("이미 시작되었습니다.")

    if n < 1:
      n = multiprocessing.cpu_count()

    for _ in range(n):
      p = multiprocessing.Process(
        target=self.provider_runner,
        args=(
            self.__shared_storage.flag_shm.name,
            self.__shared_storage.current_image_index_shm.name,
            self.__shared_storage.image_path_list_shm.name,
            self.__shared_storage.image_label_list_shm.name,
            self.__shared_storage.label_list_shm.name,
            self.__shared_storage.buffer_status_shm.name,
            self.__shared_storage.buffer_index_shm.name,
            self.__shared_storage.buffer_image_list_shm.name,
            self.__shared_storage.buffer_label_list_shm.name,
            self.__shared_storage.index_lock,
            self.__shared_storage.max_path_length,
            len(self._image_paths),
            self._SAHPE,
            self._BUFFER_SIZE,
            Worker
        )
      )
      p.start()
      self.__processes.append(p)

    return self.__consumer

  def close(self):
    for p in self.__processes:
      if p.is_alive():
          p.terminate()
    for p in self.__processes:
      p.join()

    self.__processes.clear()
    self.__consumer.close()
    self.__shared_storage.close()
    
  def unlink(self):
    self.__shared_storage.unlink()

  @property
  def shared_image_storage(self):
    return self.__shared_storage
  
  @property
  def consumer(self):
    return self.__consumer
  
def test():
    from ImageNet1kReader import ImageNet1kReader
    
    reader = ImageNet1kReader()
    paths = list(reader.path_to_label.keys())
    labels = list(reader.path_to_label.values())
    label_list = reader.label
    shape = (227, 227, 3)
    
    manager = SharedMemoryManager(
        image_paths=paths,
        image_labels=labels,
        label_list=label_list,
        shape=shape,
        buffer_size=10,
        batch_size=2
    )

    consumer = manager.start(n=6)

    # Consumer로부터 첫 배치를 읽어오는 예시
    if len(consumer) > 0:
        images, batch_labels = consumer[0]
        print("Got batch images shape:", images.shape)
        print("Got batch labels:", batch_labels)
    else:
        print("No batches available.")

    # 종료
    manager.close()
    manager.unlink()
    print("Test finished.")

if __name__ == "__main__":
    test()