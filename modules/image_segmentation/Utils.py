import tensorflow as tf


# 1. IoU (Intersection over Union)
def iou_metric(y_true, y_pred, smooth=1e-6):
  y_pred = tf.cast(y_pred > 0.5, tf.float32)  # Threshold 적용
  intersection = tf.reduce_sum(y_true * y_pred)  # 교집합
  union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - intersection  # 합집합
  return (intersection + smooth) / (union + smooth)  # IoU 계산

# 2. Dice Coefficient (F1-score 기반)
def dice_metric(y_true, y_pred, smooth=1e-6):
  y_pred = tf.cast(y_pred > 0.5, tf.float32)
  intersection = tf.reduce_sum(y_true * y_pred)
  return (2 * intersection + smooth) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + smooth)

# 3. Pixel-wise Accuracy
def pixel_accuracy(y_true, y_pred):
  y_pred = tf.cast(y_pred > 0.5, tf.float32)
  correct = tf.reduce_sum(tf.cast(y_true == y_pred, tf.float32))  # 맞춘 픽셀 수
  total = tf.reduce_sum(tf.ones_like(y_true, dtype=tf.float32))  # 전체 픽셀 수
  return correct / total