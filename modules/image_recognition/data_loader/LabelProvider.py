from pathlib import Path

class LabelProvider:
  def __init__(self, path:Path):
    self.__PATH = path

    self._index_label_dict = None

    self.__load_label()
    
  def __load_label(self):
    with open(self.__PATH, 'r') as f:
      lines = f.readlines()
    lines = list(map(lambda x: tuple(map(lambda x: x.strip() , x.strip().split(", ", 1))), lines))
    self._index_label_dict = {int(index):label for label, index in lines}

  def __len__(self):
    return len(self._index_label_dict)

  def __getitem__(self, index:int):
    return self._index_label_dict[index]

