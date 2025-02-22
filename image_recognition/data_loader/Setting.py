import multiprocessing
from data_loader.Env import Env

class Setting:
  def __init__(
    self,
    shape:tuple[int, int, int],
    max_path_length:int,
    number_of_image:int,
    batch_size:int,
    number_of_processor:int = 0,
    path_buffer_size:int = Env.PATH_BUFFER_SIZE.value,
    image_buffer_size:int = Env.IMAGE_BUFFER_SIZE.value,
    size_of_path = Env.SIZE_OF_PATH.value,
    size_of_label = Env.SIZE_OF_LABEL.value
  ):
    self.__SHAPE = shape
    self.__MAX_PATH_LENGTH = max_path_length
    self.__NUMBER_OF_IMAGE = number_of_image
    self.__BATCH_SIZE = batch_size
    
    self.__NUMBER_OF_PROCESSOR = (multiprocessing.cpu_count() -1) if number_of_processor == 0 else number_of_processor
    
    self.__PATH_BUFFER_SIZE = path_buffer_size
    self.__IMAGE_BUFFER_SIZE = image_buffer_size
    self.__SIZE_OF_PATH = size_of_path
    self.__SIZE_OF_LABEL = size_of_label

  @property
  def shape(self):
    return self.__SHAPE
  
  @property
  def max_path_length(self):
    return self.__MAX_PATH_LENGTH
  
  @property
  def number_of_image(self):
    return self.__NUMBER_OF_IMAGE
  
  @property
  def batch_size(self):
    return self.__BATCH_SIZE
  
  @property
  def number_of_processor(self):
    return self.__NUMBER_OF_PROCESSOR
  
  @property
  def path_buffer_size(self):
    return self.__PATH_BUFFER_SIZE
  
  @property
  def image_buffer_size(self):
    return self.__IMAGE_BUFFER_SIZE
  
  @property
  def size_of_path(self):
    return self.__SIZE_OF_PATH
  
  @property
  def size_of_label(self):
    return self.__SIZE_OF_LABEL