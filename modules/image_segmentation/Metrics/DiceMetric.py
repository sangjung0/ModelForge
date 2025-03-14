import tensorflow as tf

# Dice Coefficient Metric
# 2 x |X ∩ Y| / (|X| + |Y|)
class DiceMetric(tf.keras.metrics.Metric):
  def __init__(self, name="dice_metric", smooth=1e-6, **kwargs):
    super(DiceMetric, self).__init__(name=name, **kwargs)
    self.smooth = smooth
    self.total_dice = self.add_weight(name="total_dice", initializer="zeros")
    self.count = self.add_weight(name="count", initializer="zeros")

  def update_state(self, y_true, y_pred, sample_weight=None):
    intersection = tf.reduce_sum(y_true * y_pred, axis=[1, 2, 3])
    total = tf.reduce_sum(y_true + y_pred, axis=[1, 2, 3])
    dice_score = (2 * intersection + self.smooth) / (total + self.smooth)

    batch_mean_dice = tf.reduce_mean(dice_score)
    self.total_dice.assign_add(batch_mean_dice)
    self.count.assign_add(1.0)

  def result(self):
    return self.total_dice / (self.count + self.smooth) # 0 ~ 1

  def reset_states(self):
    self.total_dice.assign(0.0)
    self.count.assign(0.0)

  def get_config(self):
    config = super(DiceMetric, self).get_config()
    config.update({"smooth": self.smooth})
    return config

  @classmethod
  def from_config(cls, config):
    return cls(**config)