import cv2

class Worker:
  def __init__(self, image_request_count:int=1, buffer_requeest_scale:int=1):
    self.__IMAGE_REQUEST_COUNT = image_request_count
    self.__BUFFER_REQUEST_COUNT = image_request_count * buffer_requeest_scale

  def work(self, paths:list[str], count:int):
    images = [cv2.imread(path, cv2.IMREAD_UNCHANGED) for path in paths]
    return images

  @property
  def image_request_count(self):
    return self.__IMAGE_REQUEST_COUNT
  
  @property
  def buffer_request_count(self):
    return self.__BUFFER_REQUEST_COUNT