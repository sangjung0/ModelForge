from pathlib import Path
from data_loader import Manager, WorkerWithLabel, Setting, Utils
from ILSVRC import ILSVRC_PATH

def test():
  ILSVRC_path = Path(ILSVRC_PATH)
  train_path = ILSVRC_path / "train.csv"
  val_path = ILSVRC_path / "val.csv"
  test_path = ILSVRC_path / "test.csv"

  number_of_train = Utils.get_max_index(train_path)
  number_of_val = Utils.get_max_index(val_path)
  number_of_test = Utils.get_max_index(test_path)
  
  setting = Setting(
    shape = (224, 224, 3),
    max_path_length = max(
      Utils.get_max_length_of_path(train_path),
      Utils.get_max_length_of_path(val_path),
      Utils.get_max_length_of_path(test_path)
    ),
    number_of_image = max(number_of_train, number_of_val, number_of_test),
    batch_size = 128,
    number_of_processor = 10
  )
  manager = Manager("ILSVRC", setting)
  try:
    train_set = manager.get_train_set(train_path, WorkerWithLabel)
    valid_set = manager.get_valid_set(val_path, WorkerWithLabel)
    test_set = manager.get_test_set(test_path)

    for i in range(number_of_train):
      train_set[i]
    for i in range(number_of_val):
      valid_set[i]
    for i in range(number_of_test):
      test_set[i]
  except Exception as e:
    print(e)
  manager.close()

test()