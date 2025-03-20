import tensorflow as tf

class EpochTracker(tf.keras.callbacks.Callback):
    """현재 Epoch을 추적하는 Callback"""
    def __init__(self):
        super().__init__()
        self.__current_epoch = 0  # 현재 Epoch 저장 변수

    def on_epoch_begin(self, epoch, logs=None):
        self.__current_epoch = epoch  # `epoch` 값 업데이트

    def get_epoch(self):
        """현재 Epoch 값을 반환"""
        return self.__current_epoch
