from typing import Union
import numpy as np
import cv2
import random

class Position:
  def apply(self, background:np.ndarray, objects:list[np.ndarray], scale_factor:int = 10, max_try:int = 100, padding: Union[int, tuple[int, int, int, int]] = 0):
    """
      padding: int or tuple[int, int, int, int](top, left, bottom, right)
    """
    if background is None or objects is None or len(objects) == 0:
      raise Exception('Background and objects must be provided')
    if any([obj.shape[0] > background.shape[0] or obj.shape[1] > background.shape[1] for obj in objects]):
      raise Exception('Objects must be smaller than background')
    if any([obj.shape[2] != 4 for obj in objects]):
      raise Exception('Objects must have alpha channels')
    if isinstance(padding, int):
      padding = (padding, padding, padding, padding)
    
    self._objects = objects
    self.__SCALE_FACTOR = scale_factor

    shape = background.shape
    shape = (shape[0] - padding[0] - padding[2], shape[1] - padding[1] - padding[3])
    self._mask = np.zeros((shape[0]//self.__SCALE_FACTOR, shape[1]//self.__SCALE_FACTOR), dtype=np.bool)
    self._scaled_objects = [cv2.resize(obj, (obj.shape[1]//self.__SCALE_FACTOR, obj.shape[0]//self.__SCALE_FACTOR), interpolation=cv2.INTER_NEAREST) for obj in self._objects]

    for _ in range(max_try):
      candidate_position = self._random_position()
      if candidate_position is not None:
        break 
    else:
      return None

    return [(row*self.get_scale_factor() + padding[0], col*self.get_scale_factor() + padding[1]) for row, col in candidate_position]
    
  def get_scale_factor(self):
    return self.__SCALE_FACTOR

  def _random_position(self):
    raise Exception('Not implemented')
  
class RandomPosition(Position):
  def _random_position(self):
    scaled_position = []

    for obj in self._scaled_objects:
      candidate_positions = self.__candidate_position(self._mask, obj)
      if not candidate_positions:
        return None
      row, col = candidate_positions[np.random.randint(len(candidate_positions))]
      scaled_position.append((row, col))
      self._mask[row:row+obj.shape[0], col:col+obj.shape[1]] = obj[:, :, 3] > 0

    return scaled_position

  def __candidate_position(self, mask:np.ndarray, img:np.ndarray):
    positions = []
    for row in range(mask.shape[0] - img.shape[0]):
      for col in range(mask.shape[1] - img.shape[1]):
        if not mask[row:row+img.shape[0], col:col+img.shape[1]].any():
          positions.append((row, col))
    return positions
  
class AdaptiveQuadtreeNode(Position):
  def apply(self, background: np.ndarray, objects: list[np.ndarray], scale_factor: int = 10, max_try: int = 100, padding: int = 0, bias: bool = False):
      self.__bias = bias
      return super().apply(background, objects, scale_factor, max_try, padding)
  
  def _random_position(self):
    self.__min_shape = (
      min((obj.shape[0] for obj in self._scaled_objects)), 
      min((obj.shape[1] for obj in self._scaled_objects))
    )
    
    shape = self._mask.shape
    self.__candidate_position = [(0, 0, shape[0], shape[1])]
    scaled_posistion = []

    for obj in self._scaled_objects:
      row, col, candidate = self.__get_position(obj)
      if candidate is None:
        return None
      scaled_posistion.append((row + candidate[0], col + candidate[1]))
        
    if len(scaled_posistion) != len(self._scaled_objects):
      return None

    return scaled_posistion

  def __get_position(self, obj:np.ndarray):
    if self.__bias: self.__candidate_position.sort(key=lambda x: x[2] * x[3], reverse=True)
    else: random.shuffle(self.__candidate_position)
    
    while self.__candidate_position:
      candidate = self.__candidate_position.pop()
      if candidate[2] < obj.shape[0] or candidate[3] < obj.shape[1]:
        return None, None, None
      row, col = self.__find_position(obj, candidate)
      if row is not None and col is not None:
        return row, col, candidate
      self.__candidate_position.append(candidate)
    return None, None, None

  def __find_position(self, object_:np.ndarray, candidate:tuple[int, int, int, int]):
    r_row, r_col = candidate[0], candidate[1]

    h, w, _ = object_.shape
    ch, cw = candidate[2], candidate[3]

    if h > ch or w > cw:
      return None, None

    row = np.random.randint(0, ch - h + 1)
    col = np.random.randint(0, cw - w + 1)

    x_1, y_1, w_1, h_1 = r_row, r_col, row, col+w
    x_2, y_2, w_2, h_2 = r_row, r_col + col + w, row + h, cw - col - w
    x_3, y_3, w_3, h_3 = r_row + row + h, r_col + col, ch - row - h, cw - col
    x_4, y_4, w_4, h_4 = r_row + row, r_col, ch - row, col

    if w_1 > self.__min_shape[0] and h_1 > self.__min_shape[1]:
      self.__candidate_position.append([x_1, y_1, w_1, h_1])
    if w_2 > self.__min_shape[0] and h_2 > self.__min_shape[1]:
      self.__candidate_position.append([x_2, y_2, w_2, h_2])
    if w_3 > self.__min_shape[0] and h_3 > self.__min_shape[1]:
      self.__candidate_position.append([x_3, y_3, w_3, h_3])
    if w_4 > self.__min_shape[0] and h_4 > self.__min_shape[1]:
      self.__candidate_position.append([x_4, y_4, w_4, h_4])

    return row, col
  
def generate_position(
  background:np.ndarray, 
  objects:list[np.ndarray], 
  scaling:int = 10, 
  max_try:Union[int, tuple[int, int]] = 100, 
  padding:Union[int, tuple[int, int, int, int]] = 0, 
  bias=False
):
  if isinstance(max_try, int):
    max_try = (max_try, max_try)
  
  random_position = RandomPosition()
  positions = random_position.apply(background, objects, scaling, max_try[0], padding)
  if positions:
    return positions
  adaptive_quadtree_node = AdaptiveQuadtreeNode()
  positions = adaptive_quadtree_node.apply(background, objects, scaling, max_try[1], padding, bias = bias)
  if positions:
    return positions
  return None

def draw_at_position(background:np.ndarray, objects:list[np.ndarray], positions:list[tuple[int, int]]):
  draw_background = background.copy()
  
  for obj, (row, col) in zip(objects, positions):
    h, w = obj.shape[:2]
    roi_h = min(h, draw_background.shape[0] - row)
    roi_w = min(w, draw_background.shape[1] - col)
    obj = obj[:roi_h, :roi_w]
    roi = draw_background[row:row+h, col:col+w]
    alpha = obj[:, :, 3] / 255.0
    alpha_inv = 1.0 - alpha 

    mask = (alpha > 0)

    for c in range(3): 
      roi[:, :, c] = np.where(mask, alpha * obj[:, :, c] + alpha_inv * roi[:, :, c], roi[:, :, c])

    roi[:, :, 3] = np.where(mask, alpha * 255 + alpha_inv * roi[:, :, 3], roi[:, :, 3])

  return draw_background

def draw_segementation_at_contours(background:np.ndarray, contours:list[np.ndarray], colors:Union[tuple[int, int, int], list[tuple[int, int, int]]] = (255, 0, 0), alpha:float = 0.2):
  
  background = background.copy()
  overlay = np.zeros_like(background)
  mask = np.zeros(background.shape[:2], dtype=np.uint8)

  if isinstance(colors, tuple):
    colors = [colors for _ in contours]
  
  for contour, color in zip(contours, colors):
    cv2.drawContours(overlay, [contour], -1, color, thickness=cv2.FILLED)
    cv2.drawContours(mask, [contour], -1, 1, thickness=cv2.FILLED)

  blended = cv2.addWeighted(overlay, alpha, background, 1 - alpha, 0, background)
  background[mask > 0] = blended[mask > 0]
  
  return background
    

def mask_at_position(background:np.ndarray, objects:list[np.ndarray], positions:list[tuple[int, int]]):
  mask = np.zeros(background.shape[:2], dtype=np.uint32)
  for index, (obj, (row, col)) in enumerate(zip(objects, positions), start = 1):
    h, w = obj.shape[:2]
    roi_h = min(h, background.shape[0] - row)
    roi_w = min(w, background.shape[1] - col)
    mask[row:row+roi_h, col:col+roi_w][obj[:roi_h, :roi_w, 3] > 0] = index
  return mask

def rotate(image: np.ndarray, angle: float):
    h, w, c = image.shape

    diagonal = int(np.sqrt(w**2 + h**2))
    padded_img = np.zeros((diagonal, diagonal, c), dtype=np.uint8)

    x_offset = (diagonal - w) // 2
    y_offset = (diagonal - h) // 2
    padded_img[y_offset:y_offset+h, x_offset:x_offset+w] = image

    center = (diagonal // 2, diagonal // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(
        padded_img, M, (diagonal, diagonal),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0)
    )

    mask = rotated[:, :, 3] < 128
    rotated[mask, :] = 0 
    rotated[~mask, 3] = 255

    y, x = np.where(rotated[:, :, 3] > 0)
    rotated = rotated[y.min():y.max(), x.min():x.max()]

    return rotated
  
def centroid_at_position(background:np.ndarray, objects:list[np.ndarray], positions:list[tuple[int, int]]):
  centroids = []
  for obj, (row, col) in zip(objects, positions):
    h, w = obj.shape[:2]
    roi_h = min(h, background.shape[0] - row)
    roi_w = min(w, background.shape[1] - col)
    mask = obj[:roi_h, :roi_w, 3] > 0
    y, x = np.where(mask)
    y = (y + row).mean()
    x = (x + col).mean()
    centroids.append((y, x))
  return centroids

def centroid_at_mask(mask:np.ndarray):
  centroids = []
  for index in np.unique(mask):
    if index == 0:
      continue
    y, x = np.where(mask == index)
    y = y.mean()
    x = x.mean()
    centroids.append((y, x))
  return centroids

def bbox_at_mask(mask:np.ndarray):
  bboxes = []
  for index in np.unique(mask):
    if index == 0:
      continue
    y, x = np.where(mask == index)
    bboxes.append((y.min(), x.min(), y.max(), x.max()))
  return bboxes

def bbox_at_position(background:np.ndarray, objects:list[np.ndarray], positions:list[tuple[int, int]]):
  bboxes = []
  for obj, (row, col) in zip(objects, positions):
    h, w = obj.shape[:2]
    roi_h = min(h, background.shape[0] - row)
    roi_w = min(w, background.shape[1] - col)
    bboxes.append((row, col, row + roi_h, col + roi_w))
  return bboxes

def contours_at_mask(mask:np.ndarray):
  contours = []
  for index in np.unique(mask):
    if index == 0:
      continue
    obj = np.uint8(mask == index)*255
    contour, _ = cv2.findContours(obj.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours.append(max(contour, key=cv2.contourArea))

  return contours

def random_color_background(size:tuple[int], color:tuple[int] = None):
  if color is None:
    color = np.random.randint(0, 255, 3)
    color = np.append(color, 255)
  background = np.empty(size, dtype=np.uint8)
  background[:, :] = color
  return background

def random_noise(image, mean=0, std=10):
  noise = np.random.normal(mean, std, image.shape).astype(np.uint8)
  noisy = cv2.add(image, noise)
  return noisy

def feather_edges(image, background, mask, feather_amount=11):
  mask = cv2.GaussianBlur(mask.astype(np.float32), (feather_amount, feather_amount), 0)
  blended = (image * mask[:, :, np.newaxis] + background * (1 - mask[:, :, np.newaxis])).astype(np.uint8)

  return blended

def feather_edges(image, background, mask, feather_amount=11):
  mask = cv2.GaussianBlur(mask.astype(np.float32), (feather_amount, feather_amount), 0)
  blended = (image * mask[:, :, np.newaxis] + background * (1 - mask[:, :, np.newaxis])).astype(np.uint8)

  return blended

def gaussian_blur(image, ksize=5):
  return cv2.GaussianBlur(image, (ksize, ksize), 0)

def adjust_saturation_rgba(img, scale):
    """ RGBA 이미지의 채도를 조정 (scale > 1: 증가, scale < 1: 감소) """
    rgb = img[:, :, :3]  
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * scale, 0, 255)
    rgb_adjusted = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return np.dstack((rgb_adjusted, img[:, :, 3])) 

def adjust_exposure_rgba(img, gamma):
    """ RGBA 이미지의 노출(감마 보정) 조정 (gamma > 1: 밝게, gamma < 1: 어둡게) """
    inv_gamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** inv_gamma * 255 for i in range(256)]).astype("uint8")

    rgb = img[:, :, :3]
    rgb_adjusted = cv2.LUT(rgb, table)  # LUT를 사용해 감마 보정 적용
    return np.dstack((rgb_adjusted, img[:, :, 3]))

def add_directional_light(
    img:np.ndarray, 
    quadrilaterals: list[tuple[tuple[int, int], tuple[int, int], tuple[int, int], tuple[int, int]]],
    ksize= 151,
    intensity=150,
    increase_factor=1.5
):
    h, w, _ = img.shape
    mask = np.zeros((h, w), dtype=np.uint8)

    for quad in quadrilaterals:
        pts = np.array(quad, np.int32)
        temp_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [pts], 1)
        mask += temp_mask
        
    mask[mask > 0] = ((mask[mask > 0]**increase_factor) * intensity).astype(np.uint8)
    mask = cv2.GaussianBlur(mask, (ksize, ksize), 0)  

    light_effect = cv2.merge([mask, mask, mask])
    result = cv2.add(img[:, :, 0:3], light_effect)
    img[:, :, 0:3] = np.clip(result, 0, 255)

    return img

def random_bright_spots(shape:tuple[int, int], min:int = 1, max:int = 10):
  h, w = shape
  bright_spot = []
  for _ in range(random.randint(min, max)):
    x1, y1 = random.randint(0, w), random.randint(0, h)
    x2, y2 = random.randint(0, w), random.randint(0, h)
    x3, y3 = random.randint(0, w), random.randint(0, h)
    x4, y4 = random.randint(0, w), random.randint(0, h)
    bright_spot.append(((x1, y1), (x2, y2), (x3, y3), (x4, y4)))
  return bright_spot