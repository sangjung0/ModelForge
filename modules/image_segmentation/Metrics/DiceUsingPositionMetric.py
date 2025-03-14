import tensorflow as tf

# Dice Coefficient Metric
# 2 x |X ∩ Y| / (|X| + |Y|)
class DiceUsingPositionMetric(tf.keras.metrics.Metric):
  def __init__(self, name="dice_metric", smooth=1e-6, **kwargs):
    super(DiceUsingPositionMetric, self).__init__(name=name, **kwargs)
    self.smooth = smooth
    self.total_dice = self.add_weight(name="total_dice", initializer="zeros")
    self.count = self.add_weight(name="count", initializer="zeros")

  def update_state(self, y_true, y_pred, sample_weight=None):

    ly_true, lx_true, ry_true, rx_true = y_true[:, 0], y_true[:, 1], y_true[:, 2], y_true[:, 3]
    ly_pred, lx_pred, ry_pred, rx_pred = y_pred[:, 0], y_pred[:, 1], y_pred[:, 2], y_pred[:, 3]

    ly_inter = tf.maximum(ly_true, ly_pred)
    lx_inter = tf.maximum(lx_true, lx_pred)
    ry_inter = tf.minimum(ry_true, ry_pred)
    rx_inter = tf.minimum(rx_true, rx_pred)

    inter_height = tf.maximum(0.0, ry_inter - ly_inter)
    inter_width = tf.maximum(0.0, rx_inter - lx_inter)
    intersection_area = tf.maximum(0.0, inter_height * inter_width)

    area_true = (ry_true - ly_true) * (rx_true - lx_true)
    area_pred = (ry_pred - ly_pred) * (rx_pred - lx_pred)
    
    union_area = area_true + area_pred - intersection_area
    dice_score = (2 * intersection_area + self.smooth) / (union_area + self.smooth)

    batch_mean_dice = tf.reduce_mean(dice_score)
    self.total_dice.assign_add(batch_mean_dice)
    self.count.assign_add(1.0)

  def result(self):
    return self.total_dice / (self.count + self.smooth) # 0 ~ 1

  def reset_states(self):
    self.total_dice.assign(0.0)
    self.count.assign(0.0)

  def get_config(self):
    config = super(DiceUsingPositionMetric, self).get_config()
    config.update({"smooth": self.smooth})
    return config

  @classmethod
  def from_config(cls, config):
    return cls(**config)