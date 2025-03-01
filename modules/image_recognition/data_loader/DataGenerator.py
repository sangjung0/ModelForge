import tensorflow as tf
import numpy as np
import cv2

class DataGenerator:
    def __init__(self, image_paths: list[str], labels: list[int], batch_size: int, input_shape: tuple[int], augment_scale = 2, shuffle: bool = True):
        self.__IMAGE_PATH = np.array(image_paths)
        self.__LABELS = np.array(labels).astype(np.int32)
        self.__BATCH_SIZE = batch_size
        self.__INPUT_SHAPE = input_shape
        self.__AUGMENT_SCALE = augment_scale
        self.__SHUFFLE = shuffle

        self.__LEN = len(self.__IMAGE_PATH)

        self.dataset = self.__create_dataset()

    def _load_and_preprocess_image(self, path, label):
        """이미지를 로드하고 전처리"""
        path = path.numpy().decode("utf-8")
        img = cv2.imread(path, flags = cv2.IMREAD_COLOR_RGB)  # 이미지 로드
        
        img = cv2.resize(img, self.__INPUT_SHAPE[:2])  # 크기 조정
        img = img / 255.0  # 정규화

        return img.astype(np.float32), label

    def _augment_image(self, img):
        """데이터 증강 (필요 시 추가 가능)"""
        flipped = tf.image.flip_left_right(img)  # 좌우 반전
        augmented_images = tf.stack([img, flipped], axis=0)  # (2, H, W, C)
        return augmented_images

    def __parse_function(self, path, label):
        """Dataset에 사용할 데이터 변환 함수"""
        img, label = tf.py_function(self._load_and_preprocess_image, [path, label], [tf.float32, tf.int32])
        img.set_shape(self.__INPUT_SHAPE)  # 고정된 크기 설정

        aug_images = tf.py_function(self._augment_image, [img], tf.float32)
        aug_images.set_shape((self.__AUGMENT_SCALE, *self.__INPUT_SHAPE))

        labels = tf.repeat(tf.expand_dims(label, 0), self.__AUGMENT_SCALE)
        
        return tf.data.Dataset.from_tensor_slices((aug_images , labels))

    def __create_dataset(self):
        """멀티프로세싱을 활용한 tf.data.Dataset 생성"""
        dataset = tf.data.Dataset.from_tensor_slices((self.__IMAGE_PATH, self.__LABELS))

        if self.__SHUFFLE:
            dataset = dataset.shuffle(buffer_size=self.__LEN)

        dataset = dataset.interleave(self.__parse_function, num_parallel_calls=tf.data.AUTOTUNE)  # 멀티프로세싱
        dataset = dataset.batch(self.__BATCH_SIZE).prefetch(tf.data.AUTOTUNE)  # GPU 훈련 속도 최대화

        return dataset