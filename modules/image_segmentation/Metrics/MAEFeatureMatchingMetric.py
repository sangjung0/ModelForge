from math import pi
import tensorflow as tf

class MAEFeatureMatchingMetric(tf.keras.metrics.Metric):
  def __init__(self, name="mae_feature_matching_metric", **kwargs):
    super(MAEFeatureMatchingMetric, self).__init__(name=name, **kwargs)
    self.total_mae_feature_matching = self.add_weight(name="total_mae_feature_matching", initializer="zeros")
    self.count = self.add_weight(name="count", initializer="zeros")

  def update_state(self, y_true, y_pred, epoch, model, warmup:int=1000, scale:int = 1, sample_weight=None):
    mae = tf.reduce_mean(tf.reduce_sum(tf.abs(y_true - y_pred), axis=(1, 2, 3)))
    y_true = model(y_true)
    y_pred = model(y_pred)
    feature_matching = tf.reduce_mean(tf.reduce_sum(tf.abs(y_true - y_pred), axis=(1, 2, 3)))

    ratio = (epoch / warmup) * (pi / 2)
    # tf.print("\nratio:", ratio)
    # tf.print(epoch, pi)
    ratio = tf.sin(ratio)
    # tf.print("post ratio", ratio)
    sum = (mae * (1 - ratio) + feature_matching * ratio) * scale
    # tf.print(mae)
    # tf.print(feature_matching)

    self.total_mae_feature_matching.assign_add(sum)
    self.count.assign_add(1.0)
    
  def result(self):
    return self.total_mae_feature_matching / self.count

  def reset_states(self):
    self.total_mae_feature_matching.assign(0.0)
    self.count.assign(0.0)
    
  def get_config(self):
    config = super(MAEFeatureMatchingMetric, self).get_config()
    return config
  
  @classmethod
  def from_config(cls, config):
    return cls(**config)