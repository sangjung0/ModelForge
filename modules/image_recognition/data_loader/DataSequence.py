import tensorflow as tf
import numpy as np
import cv2

class DataSequence(tf.keras.utils.Sequence):
  def __init__(self, image_paths:list[str], labels:list[int], batch_size:int, input_size:tuple[int], augment:bool=False, shuffle:bool=True):
    self.__IMAGE_PATHS = image_paths
    self.__LABELS = labels
    self.__BATCH_SIZE = batch_size
    self.__INPUT_SIZE = input_size
    self.__AUGMENT = augment
    self.__SHUFFLE = shuffle
    
    self.__INDEXES = np.arange(len(self.__IMAGE_PATHS))
    self.__LEN = int(np.floor(len(self.__IMAGE_PATHS) / self.__BATCH_SIZE))

    self.on_epoch_end()

  def __len__(self):
    """한 epoch에 필요한 batch 개수"""
    return self.__LEN

  def __getitem__(self, index):
    """index에 해당하는 batch 데이터를 로드"""
    batch_index = index * self.__BATCH_SIZE
    indexes = self.__INDEXES[batch_index:batch_index + self.__BATCH_SIZE]

    image_paths = [self.__IMAGE_PATHS[k] for k in indexes]
    labels = [self.__LABELS[k] for k in indexes]

    X, y = self.__data_generation(image_paths, labels)
    return X, y

  def __data_generation(self, image_paths, labels):
    """데이터 로드 및 증강"""
    X = np.empty((self.__BATCH_SIZE, *self.__INPUT_SIZE))
    y = np.array(labels)

    for i, path in enumerate(image_paths):
        img = cv2.imread(path)
        img = cv2.resize(img, self.__INPUT_SIZE[:2])
        img = img / 255.0  # Normalize

        if self.__AUGMENT:
            img = self.augment_image(img)

        X[i,] = img

    return X, y

  def augment_image(self, img):
    """이미지 증강 (필요하면 추가 가능)"""
    if np.random.rand() > 0.5:
        img = cv2.flip(img, 1)  # 수평 뒤집기
    return img

  def on_epoch_end(self):
    """epoch 종료 후 데이터 섞기"""
    if self.__SHUFFLE:
        np.random.shuffle(self.__INDEXES)
