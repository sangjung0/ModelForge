from pathlib import Path
from itertools import islice
import csv

class PathProvider:
  def __init__(self, path:Path):
    if not path.exists():
      raise FileNotFoundError(f"{path} not found")

    if path.suffix != ".csv":
      raise Exception(f"{path} is not a csv file")
    
    self.__PATH = path
    
  def _parse_data(self, data:list[str]):
    return data[0]
  
  def get(self, index:int, length:int):
      result = []
      
      with open(self.__PATH, 'r') as f:
        reader = csv.reader(f)
        for row in islice(reader, index, index + length):
          result.append(self._parse_data(row))

      return result

  def get_all(self):
    result = []
    
    with open(self.__PATH, 'r') as f:
      reader = csv.reader(f)
      for row in reader:
        result.append(self._parse_data(row))
    
    return result

def test():
  PathProvider()
  print("🟩 PathProvider test passed")

if __name__ == "__main__":
  test()