from pathlib import Path
from .PathProvider import PathProvider


class PathProviderWithLabel(PathProvider):
  def __init__(self, path:Path):
    super().__init__(path)
    
  def _parse_data(self, data:list[str]):
    return (data[0], int(data[1]))