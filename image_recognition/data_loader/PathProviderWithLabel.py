from pathlib import Path
from data_loader.PathProvider import PathProvider
from data_loader.env import READ_BUFFER_SIZE


class PathProviderWithLabel(PathProvider):
  def __init__(self, path:Path, read_buffer_size:int = READ_BUFFER_SIZE):
    super().__init__(path, read_buffer_size)
    
  def _parse_data(self, data:list[str]):
    return (data[0], int(data[1]))