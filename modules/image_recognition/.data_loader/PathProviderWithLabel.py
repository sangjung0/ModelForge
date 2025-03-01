from pathlib import Path
from data_loader.PathProvider import PathProvider
from data_loader.Env import Env


class PathProviderWithLabel(PathProvider):
  def __init__(self, path:Path, read_buffer_size:int = Env.READ_BUFFER_SIZE.value):
    super().__init__(path, read_buffer_size)
    
  def _parse_data(self, data:list[str]):
    return (data[0], int(data[1]))