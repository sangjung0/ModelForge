from enum import Enum
import numpy as np

class Env(Enum):
  PATH_BUFFER_SIZE = 10000
  IMAGE_BUFFER_SIZE = 1000
  SIZE_OF_LABEL = np.int32
  SIZE_OF_PATH = np.int32
  INT_SIZE = 4
  MIN_SLEEP_TIME = 0.1
  MAX_SLEEP_TIME = 1
  READ_BUFFER_SIZE = 1_073_741_824