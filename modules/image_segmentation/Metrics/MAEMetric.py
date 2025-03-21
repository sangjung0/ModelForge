import tensorflow as tf

class MAEMetric(tf.keras.metrics.Metric):
    def __init__(self, name='mae', **kwargs):
        super(MAEMetric, self).__init__(name=name, **kwargs)
        self.mae = self.add_weight(name='mae', initializer='zeros')
        self.count = self.add_weight(name='count', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):

        mae = tf.reduce_mean(tf.reduce_sum(tf.abs(y_true - y_pred), axis=(1, 2, 3)))
        self.mae.assign_add(mae)
        self.count.assign_add(1)

    def result(self):
        return self.mae / self.count

    def reset_states(self):
        self.mae.assign(0)
        self.count.assign(0)

    def get_config(self):
        config = super(MAEMetric, self).get_config()
        return config
      
    @classmethod
    def from_config(cls, config):
        return cls(**config)