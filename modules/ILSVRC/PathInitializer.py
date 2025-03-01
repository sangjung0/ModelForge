import time
import xml.etree.ElementTree as ET
import random
from pathlib import Path
from tqdm import tqdm
from enum import Enum

from .utils import extract_number
from .env import ILSVRC_PATH

class Mode(Enum):
  ONLY_STRICT = "only_strict"
  STRICT_AND_TEST = "strict_and_test"
  ONLY_TEST = "only_test"
  NO_STRICT_AND_TEST = "no_strict_and_test"

class PathInitializer:
  def __init__(self, sort:bool = False, shuffle:bool = False, path:str = ILSVRC_PATH, mode:str = Mode.NO_STRICT_AND_TEST):
    self.__STRICT_MODE = True if mode in [Mode.ONLY_STRICT, Mode.STRICT_AND_TEST] else False
    self.__TEST_MODE = True if mode in [Mode.ONLY_TEST, Mode.STRICT_AND_TEST] else False

    self.__SORT = sort
    self.__SHUFFLE = shuffle
    
    self.path = Path(path)
    self.synset_mapping_path = self.path / "LOC_synset_mapping.txt"
    self.annotation_path = self.path / "ILSVRC" / "Annotations" / "CLS-LOC"
    self.data_path = self.path / "ILSVRC" / "Data" / "CLS-LOC"
    self.train_csv_path = self.path / "train.csv"
    self.val_csv_path = self.path / "val.csv"
    self.test_csv_path = self.path / "test.csv"
    self.label_csv_path = self.path / "label.csv"
    self.label_name_csv_path = self.path / "label_name.csv"
    
    self._label_number_list = None
    self._name_of_label_list = None
    self._label_index_dict = None
    # self._train_annotation = None
    self._val_annotation = None
    self._train_path_list = None
    self._val_path_list = None
    self._test_path_list = None

  def isExist(self):
    return self.train_csv_path.exists() and self.val_csv_path.exists() and self.test_csv_path.exists() and self.label_csv_path.exists() and self.label_name_csv_path.exists()
    
  def save(self):
    if self.__STRICT_MODE and self.__TEST_MODE:
      print(f"🟩 Strict & test mode")
    elif self.__TEST_MODE:
      print(f"🟩 Test start")
    elif self.__STRICT_MODE:
      print(f"🟩 Strict mode")
    
    cover_bar = tqdm(total=8, desc="🟩 Saveing..")
    time.sleep(1)
    cover_bar.set_description("🟩 save synset mapping")
    self.__parse_synset_mapping()
    cover_bar.update(1)

    cover_bar.set_description("🟩 save annotation")
    self.__parse_annotation()
    cover_bar.update(1)

    cover_bar.set_description("🟩 save image path")
    self.__get_image_path()
    cover_bar.update(1)

    cover_bar.set_description("🟩 save train csv")
    self.__save_csv_to_train()
    cover_bar.update(1)

    cover_bar.set_description("🟩 save val csv")
    self.__save_csv_to_val()
    cover_bar.update(1)
    
    cover_bar.set_description("🟩 save test csv")
    self.__save_csv_to_test()
    cover_bar.update(1)

    cover_bar.set_description("🟩 save label csv")
    self.__save_csv_to_label()
    cover_bar.update(1)

    cover_bar.set_description("🟩 save label name csv")
    self.__save_csv_to_label_name()
    cover_bar.update(1)

    cover_bar.set_description("🟩 save done")
    cover_bar.close()

    if self.__TEST_MODE:
      print(f"✅ Test succeeded")

  def __parse_synset_mapping(self):
    with open(self.synset_mapping_path, "r") as f:
      lines = f.readlines()
    
    lines = list(map(lambda x: tuple(map(lambda x: x.strip(), x.strip().split(" ", 1))), lines))
    lines.sort(key=lambda x: x[0])
    label_number_list, name_of_label_list = list(zip(*lines))
    label_index_dict = {label_number: index for index, label_number in enumerate(label_number_list)}

    self._label_number_list = label_number_list
    self._name_of_label_list = name_of_label_list
    self._label_index_dict = label_index_dict

  def __parse_annotation_val(self, path: Path):
    if self._label_index_dict is None:
      raise Exception("❌ Label is not parsed yet")
    
    temp = {}
    
    for xml_file in tqdm(list(path.iterdir()), desc=f"🟢 Parse annotation in {path}"):
      if not xml_file.is_file():
        if self.__STRICT_MODE:
          raise Exception(f"❌ Invalid file: {xml_file}")
        continue
      if xml_file.suffix != ".xml":
        if self.__STRICT_MODE:
          raise Exception(f"❌ Invalid file: {xml_file}")
        continue
      tree = ET.parse(xml_file)
      root = tree.getroot()
      filename = root.find("filename").text
      object_tag = root.find("object")
      name_tag = object_tag.find("name")
      name = name_tag.text
  
      temp[filename] = self._label_index_dict[name]
        
    return temp

  def __parse_annotation_(self, path: Path):
    if self._label_index_dict is None:
      raise Exception("❌ Label is not parsed yet")
    
    temp = {}
    
    for class_folder in tqdm(list(path.iterdir()), desc=f"🟢 find folder in {path}"):
      if not class_folder.is_dir():
        if self.__STRICT_MODE:
          raise Exception(f"❌ Invalid folder: {class_folder}")
        continue
      if not class_folder.name in self._label_index_dict:
        raise Exception(f"❌ Label not found: {class_folder.name}")
      for xml_file in tqdm(list(class_folder.iterdir()), desc=f"Parse annotation in {class_folder}"):
        if not xml_file.is_file():
          if self.__STRICT_MODE:
            raise Exception(f"❌ Invalid file: {xml_file}")
          continue
        if xml_file.suffix != ".xml":
          if self.__STRICT_MODE:
            raise Exception(f"❌ Invalid file: {xml_file}")
          continue
        tree = ET.parse(xml_file)
        root = tree.getroot()
        filename = root.find("filename").text
        object_tag = root.find("object")
        name_tag = object_tag.find("name")
        name = name_tag.text
    
        temp[filename] = self._label_index_dict[name]
        
      if self.__TEST_MODE:
        break

    return temp

  def __parse_annotation(self):
    # train_path = self.annotation_path / "train"
    val_path = self.annotation_path / "val"

    # self._train_annotation = self.__parse_annotation_(train_path)
    self._val_annotation = self.__parse_annotation_val(val_path)

  def __get_image_test_path(self, path:Path):
    temp = []
    for image_file in tqdm(list(path.iterdir()), desc=f"🟢 Get image path in {path}"):
      if not image_file.is_file():
        if self.__STRICT_MODE:
          raise Exception(f"❌ Invalid file: {image_file}")
        continue
      if image_file.suffix != ".JPEG":
        if self.__STRICT_MODE:
          raise Exception(f"❌ Invalid file: {image_file}")
        continue
      temp.append(image_file.resolve())
      
    if self.__SORT:
      temp.sort(key=lambda x: extract_number(x.stem))
    
    return temp

  def __get_image_val_path(self, path:Path, annotation_dict:dict[str, int]):
    if annotation_dict is None:
      raise Exception("❌ Annotation is not parsed yet")
    
    temp = []
    for image_file in tqdm(list(path.iterdir()), desc=f"🟢 Get image path in {path}"):
      if not image_file.is_file():
        if self.__STRICT_MODE:
          raise Exception(f"❌ Invalid file: {image_file}")
        continue
      if image_file.suffix != ".JPEG":
        if self.__STRICT_MODE:
          raise Exception(f"❌ Invalid file: {image_file}")
        continue
      temp.append((image_file.resolve(), annotation_dict[image_file.stem]))
    
    if self.__SORT:
      temp.sort(key=lambda x: x[1])
      temp.sort(key=lambda x: extract_number(x[0].stem))
    return temp

  def __get_image_path_(self, path:Path):
    if self._label_index_dict is None:
      raise Exception("❌ Label is not parsed yet")
    
    temp = []
    for class_folder in tqdm(list(path.iterdir()), desc=f"🟢 find folder in {path}"):
      if not class_folder.is_dir():
        if self.__STRICT_MODE:
          raise Exception(f"❌ Invalid folder: {class_folder}")
        continue
      if not class_folder.name in self._label_index_dict:
        raise Exception(f"❌ Label not found: {class_folder.name}")
      label_index = self._label_index_dict[class_folder.name]
      for image_file in tqdm(list(class_folder.iterdir()), desc=f"🟢 Get image path in {class_folder}"):
        if not image_file.is_file():
          continue
        if image_file.suffix != ".JPEG":
          continue
        temp.append((image_file.resolve(), label_index))
        
      if self.__TEST_MODE:
        break
    
    if self.__SORT:
      temp.sort(key=lambda x: x[1])
      temp.sort(key=lambda x: extract_number(x[0].stem))
    return temp

  def __get_image_path(self):
    test_path = self.data_path / "test"
    train_path = self.data_path / "train"
    val_path = self.data_path / "val"
    
    self._test_path_list = self.__get_image_test_path(test_path)
    # self._train_path_list = self.__get_image_path_(train_path, self._train_annotation)
    # self._val_path_list = self.__get_image_path_(val_path, self._val_annotation)
    self._train_path_list = self.__get_image_path_(train_path)
    self._val_path_list = self.__get_image_val_path(val_path, self._val_annotation)

  def __save_csv_to_train(self):
    if self.__SHUFFLE: random.shuffle(self._train_path_list)
    with open(self.train_csv_path, "w") as f:
      for path, label in self._train_path_list:
        f.write(f"{path},{label}\n")

  def __save_csv_to_val(self):
    with open(self.val_csv_path, "w") as f:
      for path, label in self._val_path_list:
        f.write(f"{path},{label}\n")
  
  def __save_csv_to_test(self):
    with open(self.test_csv_path, "w") as f:
      for path in self._test_path_list:
        f.write(f"{path}\n")
  
  def __save_csv_to_label(self):
    with open(self.label_csv_path, "w") as f:
      for label, index in self._label_index_dict.items():
        f.write(f"{label}, {index}\n")
        
  def __save_csv_to_label_name(self):
    with open(self.label_name_csv_path, "w") as f:
      for label, name in zip(self._label_number_list, self._name_of_label_list):
        f.write(f"{label}, {name}\n")

        
def test():
  initializer = PathInitializer(shuffle=True, mode=Mode.STRICT_AND_TEST)
  initializer.save()
  print("🟩 Test Done")

def main():
  initializer = PathInitializer(mode=Mode.NO_STRICT_AND_TEST)
  if initializer.isExist():
    print("🟩 Already exist")
    return
  initializer.save()
  print("🟩 Done")
  
if __name__ == "__main__":
  # test()
  main()