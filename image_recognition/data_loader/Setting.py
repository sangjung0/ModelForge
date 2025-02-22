import multiprocessing
from data_loader.env import *

class Setting:
  def __init__(
    self,
    shape:tuple[int, int, int],
    max_path_length:int,
    number_of_image:int,
    batch_size:int,
    with_label:bool,
    number_of_processor:int = 0,
    path_buffer_size:int = PATH_BUFFER_SIZE,
    image_buffer_size:int = IMAGE_BUFFER_SIZE,
    size_of_path = SIZE_OF_PATH,
    size_of_label = SIZE_OF_LABEL
  ):
    self.__SHAPE = shape
    self.__MAX_PATH_LENGTH = max_path_length
    self.__NUMBER_OF_IMAGE = number_of_image
    self.__BATCH_SIZE = batch_size
    self.__WITH_LABEL = with_label
    
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
  def with_label(self):
    return self.__WITH_LABEL
  
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