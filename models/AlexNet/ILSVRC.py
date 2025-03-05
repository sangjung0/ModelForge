from pathlib import Path

from image_recognition.ILSVRC import ILSVRC
from image_recognition.data_loader import DataGenerator

from AlexNet import AlexNet

BATCH_SIZE = 128
INPUT_SHAPE = (277, 277, 3)

def main():
  label_map = ILSVRC.label_dict()
  train_paths, train_labels = ILSVRC.train_paths_and_labels()
  train_data = DataGenerator(train_paths, train_labels, BATCH_SIZE, INPUT_SHAPE)

  alexnet = AlexNet(input_shape=INPUT_SHAPE, num_classes=len(label_map))
  alexnet.train(train_data.dataset)
  alexnet.save(Path("./"))

if __name__ == "__main__":
  main()