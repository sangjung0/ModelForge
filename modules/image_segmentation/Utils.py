import tensorflow as tf

# 1. IoU Metric (Mean IoU)
class IoUMetric(tf.keras.metrics.Metric):
    def __init__(self, name="iou_metric", smooth=1e-6, **kwargs):
        super(IoUMetric, self).__init__(name=name, **kwargs)
        self.smooth = smooth
        self.iou = self.add_weight(name="iou", initializer="zeros")
        self.count = self.add_weight(name="count", initializer="zeros")  # 배치 개수 저장

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.cast(y_pred > 0.5, tf.float32)  # Threshold 적용
        intersection = tf.reduce_sum(y_true * y_pred)
        union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - intersection
        iou_score = (intersection + self.smooth) / (union + self.smooth)

        self.iou.assign_add(iou_score)  # 값 누적
        self.count.assign_add(1.0)  # 배치 개수 증가

    def result(self):
        return self.iou / (self.count + self.smooth)  # 배치 평균 반환

    def reset_states(self):
        self.iou.assign(0.0)
        self.count.assign(0.0)


# 2. Dice Coefficient Metric
class DiceMetric(tf.keras.metrics.Metric):
    def __init__(self, name="dice_metric", smooth=1e-6, **kwargs):
        super(DiceMetric, self).__init__(name=name, **kwargs)
        self.smooth = smooth
        self.dice = self.add_weight(name="dice", initializer="zeros")
        self.count = self.add_weight(name="count", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.cast(y_pred > 0.5, tf.float32)
        intersection = tf.reduce_sum(y_true * y_pred)
        dice_score = (2 * intersection + self.smooth) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + self.smooth)

        self.dice.assign_add(dice_score)
        self.count.assign_add(1.0)

    def result(self):
        return self.dice / (self.count + self.smooth)  # 배치 평균 반환

    def reset_states(self):
        self.dice.assign(0.0)
        self.count.assign(0.0)


# 3. Pixel Accuracy Metric
class PixelAccuracy(tf.keras.metrics.Metric):
    def __init__(self, name="pixel_accuracy", **kwargs):
        super(PixelAccuracy, self).__init__(name=name, **kwargs)
        self.accuracy = self.add_weight(name="accuracy", initializer="zeros")
        self.count = self.add_weight(name="count", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.cast(y_pred > 0.5, tf.float32)
        correct = tf.reduce_sum(tf.cast(y_true == y_pred, tf.float32))
        total = tf.size(y_true, out_type=tf.float32)  # 전체 픽셀 개수
        accuracy = correct / total

        self.accuracy.assign_add(accuracy)
        self.count.assign_add(1.0)

    def result(self):
        return self.accuracy / (self.count + 1e-6)  # 배치 평균 반환

    def reset_states(self):
        self.accuracy.assign(0.0)
        self.count.assign(0.0)
