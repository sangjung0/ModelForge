import tensorflow as tf


class FeatureMatchingMetric(tf.keras.metrics.Metric):
    def __init__(self, name="feature_matching_metric", **kwargs):
        super(FeatureMatchingMetric, self).__init__(name=name, **kwargs)
        self.total_feature_matching = self.add_weight(name="total_feature_matching", initializer="zeros")
        self.count = self.add_weight(name="count", initializer="zeros")

    def update_state(self, y_true, y_pred, model, sample_weight=None):
        y_true = model(y_true)
        y_pred = model(y_pred)
        feature_matching = tf.reduce_mean(tf.reduce_sum(tf.abs(y_true - y_pred), axis=(1, 2, 3)))

        self.total_feature_matching.assign_add(feature_matching)
        self.count.assign_add(1.0)

    def result(self):
        return self.total_feature_matching / self.count

    def reset_states(self):
        self.total_feature_matching.assign(0.0)
        self.count.assign(0.0)

    def get_config(self):
        config = super(FeatureMatchingMetric, self).get_config()
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)