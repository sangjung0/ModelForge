from pathlib import Path
from .env import ILSVRC_PATH
from image_recognition.data_loader import LabelProvider, PathProvider, PathProviderWithLabel


class ILSVRC:
  path = Path(ILSVRC_PATH)
  
  @staticmethod
  def train_paths_and_labels():
    provider = PathProviderWithLabel(ILSVRC.path / "train.csv")
    result = provider.get_all()
    return zip(*result)

  @staticmethod
  def val_paths_and_labels():
    provider = PathProviderWithLabel(ILSVRC.path / "val.csv")
    result = provider.get_all()
    return zip(*result)
  
  @staticmethod
  def test_paths():
    provider = PathProvider(ILSVRC.path / "test.csv")
    return provider.get_all()

  @staticmethod
  def label_dict():
    provider = LabelProvider(ILSVRC.path / "label.csv")
    return provider