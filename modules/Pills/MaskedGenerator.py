from Pills import *
from Pills.Utils import *

class MaskedGenerator(RandomEffectGenerator):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)

    self.__MIN_DISTANCE = 30
    self.__MAX_REPITATION = 3

  def generate_masked(self, *args, **kargs):
    img, _, (_, _, _, bboxes, _) = self.generate_with_random_effect(*args, **kargs)
    img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)

    mean_color = np.mean(img, axis=(0, 1))

    contaminated_img = img.copy()
    masked_bbox = []
    for bbox in bboxes:
      if self._RANDOM.random() < 0.5:
        continue
      for _ in range(self._RANDOM.randint(1, self.__MAX_REPITATION)):
        ly, lx, ry, rx = bbox
        ly, lx = random.randint(ly, ry), random.randint(lx, rx)
        ry, rx = (
          random.randint(min(ly + self.__MIN_DISTANCE, ry), ry), 
          random.randint(min(lx + self.__MIN_DISTANCE, rx), rx)
        )
        contaminated_img[ly:ry, lx:rx] = mean_color
        masked_bbox.append((ly, lx, ry, rx))
    
    contaminated_img = gaussian_blur(contaminated_img,  5)
    masked_img = img.copy()
    for ly, lx, ry, rx in masked_bbox:
      masked_img[ly:ry, lx:rx] = contaminated_img[ly:ry, lx:rx]

    return masked_img, img

if __name__ == "__main__":
  pass
  