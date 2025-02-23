import cv2

from data_loader import Worker


class WorkerWithLabel(Worker):
  def __init__(self, image_requeest_count:int = 1, buffer_requeest_scale:int = 1):
    super().__init__(image_requeest_count, buffer_requeest_scale)

  def work(self, paths:list[tuple[str, int]], count:int):
    result = []
    for path, label in paths:
      image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
      result.append((image, label))
    return result