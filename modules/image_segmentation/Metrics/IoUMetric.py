import tensorflow as tf

#  IoU Metric (Mean IoU)
# Intersection over Union (IoU) = TP / (TP + FP + FN)
class IoUMetric(tf.keras.metrics.Metric):
  def __init__(self, name="iou_metric", smooth=1e-6, **kwargs):
    super(IoUMetric, self).__init__(name=name, **kwargs)
    self.smooth = smooth
    self.total_iou = self.add_weight(name="total_iou", initializer="zeros")
    self.count = self.add_weight(name="count", initializer="zeros")  # 배치 개수 저장

  def update_state(self, y_true, y_pred, sample_weight=None):
    y_pred = tf.cast(y_pred > 0.5, tf.float32)  # Threshold 적용

    intersection = tf.reduce_sum(y_true * y_pred, axis=[1, 2, 3])
    union = tf.reduce_sum(y_true, axis=[1, 2, 3]) + tf.reduce_sum(y_pred, axis=[1, 2, 3]) - intersection
    iou_score = (intersection + self.smooth) / (union + self.smooth)

    batch_mean_iou = tf.reduce_mean(iou_score)  # 배치 평균 계산
    self.total_iou.assign_add(batch_mean_iou)  # 값 누적
    self.count.assign_add(1.0)  # 배치 개수 증가

  def result(self):
    return self.total_iou / (self.count + self.smooth)  # 배치 평균 반환

  def reset_states(self):
    self.total_iou.assign(0.0)
    self.count.assign(0.0)

  def get_config(self):
    config = super(IoUMetric, self).get_config()
    config.update({"smooth": self.smooth})
    return config

  @classmethod
  def from_config(cls, config):
    return cls(**config)