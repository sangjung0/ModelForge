from pathlib import Path
from itertools import islice
import csv

from data_loader.Env import Env

class PathProvider:
  def __init__(self, path:Path, read_buffer_size:int = Env.READ_BUFFER_SIZE.value):
    if not path.exists():
      raise FileNotFoundError(f"{path} not found")

    if path.suffix != ".csv":
      raise Exception(f"{path} is not a csv file")
    
    self.__PATH = path
    self.__READ_BUFFER_SIZE = read_buffer_size
    
  def _parse_data(self, data:list[str]):
    return data[0]
  
  def get_paths(self, index:int, length:int):
      result = []
      
      with open(self.__PATH, 'r', buffering=self.__READ_BUFFER_SIZE) as f:
        reader = csv.reader(f)
        for row in islice(reader, index, index + length):
          result.append(self._parse_data(row))

      return result

def test():
  PathProvider()
  print("🟩 PathProvider test passed")

if __name__ == "__main__":
  test()