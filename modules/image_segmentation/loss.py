import tensorflow as tf

### 1. Dice Loss
def dice_loss(y_true, y_pred, smooth=1e-6):
    intersection = tf.reduce_sum(y_true * y_pred, axis=[1,2,3])  # 배치별 Intersection
    total = tf.reduce_sum(y_true + y_pred, axis=[1,2,3])  # 배치별 Total Pixels
    dice = (2. * intersection + smooth) / (total + smooth)  # Dice Score

    return tf.reduce_mean(1 - dice)  # Loss는 최소화해야 하므로 (1 - Dice Score)

### 2. IoU Loss
def iou_loss(y_true, y_pred, smooth=1e-6):
    intersection = tf.reduce_sum(y_true * y_pred, axis=[1,2,3])  # 배치별 Intersection
    union = tf.reduce_sum(y_true + y_pred, axis=[1,2,3]) - intersection  # 배치별 Union
    iou = (intersection + smooth) / (union + smooth)  # IoU Score

    return tf.reduce_mean(1 - iou)  # Loss는 최소화해야 하므로 (1 - IoU)

### 3. Pixel Accuracy Loss (1 - 픽셀 정확도)
def pixel_accuracy_loss(y_true, y_pred):
    correct = tf.reduce_sum(tf.cast(y_true == y_pred, tf.float32), axis=[1,2,3])  # 맞은 픽셀 개수
    total = tf.cast(tf.reduce_prod(tf.shape(y_true)[1:]), tf.float32)  # 전체 픽셀 개수
    accuracy = correct / total  # 배치별 픽셀 정확도

    return tf.reduce_mean(1 - accuracy)  # Loss는 최소화해야 하므로 (1 - Accuracy)


def dice_using_position_loss(y_true, y_pred, smooth=1e-6):
    ly_true, lx_true, ry_true, rx_true = tf.split(y_true, 4, axis=-1)
    ly_pred, lx_pred, ry_pred, rx_pred = tf.split(y_pred, 4, axis=-1)

    ly_inter = tf.maximum(ly_true, ly_pred)
    lx_inter = tf.maximum(lx_true, lx_pred)
    ry_inter = tf.minimum(ry_true, ry_pred)
    rx_inter = tf.minimum(rx_true, rx_pred)

    inter_height = tf.maximum(0.0, ry_inter - ly_inter)
    inter_width = tf.maximum(0.0, rx_inter - lx_inter)
    intersection_area = inter_height * inter_width

    area_true = (ry_true - ly_true) * (rx_true - lx_true)
    area_pred = (ry_pred - ly_pred) * (rx_pred - lx_pred)
    
    union_area = area_true + area_pred - intersection_area
    dice_score = (2 * intersection_area + smooth) / (union_area + smooth)

    return tf.reduce_mean(1 - dice_score)