from pathlib import Path
from itertools import islice
from tqdm import tqdm
import csv

from data_loader.env import READ_BUFFER_SIZE

class PathProvider:
  def __init__(self, path:Path, read_buffer_size:int = READ_BUFFER_SIZE):
    if not path.exists():
      raise FileNotFoundError(f"{path} not found")

    if path.suffix != ".csv":
      raise Exception(f"{path} is not a csv file")
    
    self.__PATH = path
    self.__READ_BUFFER_SIZE = read_buffer_size
    
    self._max_index = None

  def __check_max_index(self, path: Path):
    with open(path, "r", buffering=self.__READ_BUFFER_SIZE) as f:
        max_index = sum(1 for _ in tqdm(f, desc=f"🟩 Check max index"))
    return max_index

  def _parse_data(self, data:list[str]):
    return data[0]
  
  def get_paths(self, index:int, length:int):
      result = []
      
      with open(self.__PATH, 'r', buffering=self.__READ_BUFFER_SIZE) as f:
        reader = csv.reader(f)
        for row in islice(reader, index, index + length):
          result.append(self._parse_data(row))

      return result

  @property
  def max_index(self):
    if self._max_index is None:
      self._max_index = self.__check_max_index(self.__PATH)
    return self._max_index

def test():
  PathProvider()
  print("🟩 PathProvider test passed")

if __name__ == "__main__":
  test()