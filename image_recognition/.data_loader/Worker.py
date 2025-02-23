import cv2

class Worker:
  def __init__(self, image_request_count:int=1, buffer_requeest_scale:int=1):
    """
    image_request_count: int
      Number of images to be loaded at once
      image_request_count는 적당한 값을 찾아야 합니다.
    buffer_requeest_scale: int
      The number of buffers to be allocated for each image
      만약, image_request_count가 1일 때, 데이터 증강을 통해서 이미지를 4배로 늘린다면, buffer_requeest_scale를 4로 설정해야 합니다.
    """
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