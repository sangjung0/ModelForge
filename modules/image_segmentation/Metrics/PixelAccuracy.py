import tensorflow as tf

# Pixel Accuracy Metric
class PixelAccuracy(tf.keras.metrics.Metric):
    def __init__(self, name="pixel_accuracy", smooth=1e-6, **kwargs):
        super(PixelAccuracy, self).__init__(name=name, **kwargs)
        self.smooth = smooth
        self.total_accuracy = self.add_weight(name="total_accuracy", initializer="zeros")
        self.count = self.add_weight(name="count", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        correct = tf.reduce_sum(tf.cast(y_true == y_pred, tf.float32), axis=[1, 2, 3])
        total = tf.cast(tf.reduce_prod(tf.shape(y_true)[1:]), tf.float32)
        accuracy = correct / total

        batch_mean_accuracy = tf.reduce_mean(accuracy)
        self.total_accuracy.assign_add(batch_mean_accuracy)
        self.count.assign_add(1.0)

    def result(self):
        return self.total_accuracy / (self.count + self.smooth)  # 0 ~ 1

    def reset_states(self):
        self.total_accuracy.assign(0.0)
        self.count.assign(0.0)

    def get_config(self):
        config = super(PixelAccuracy, self).get_config()
        config.update({"smooth": self.smooth})
        return config
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)