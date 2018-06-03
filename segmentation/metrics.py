
import tensorflow as tf

def dice_coeff(seg_true, seg_pred):
    with tf.variable_scope("dice_coff_loss"):
        seg_true_flat = tf.layers.flatten(seg_true)
        seg_pred_flat = tf.layers.flatten(seg_pred)
        intersection = tf.multiply(seg_true_flat, seg_pred_flat, name="intersection")
        intersect = tf.reduce_sum(intersection)

        smooth = 0.01
        dice = tf.divide(2 * intersect + smooth, tf.reduce_sum(seg_true_flat) + tf.reduce_sum(seg_pred_flat) + smooth)
        return dice


def dice_loss(seg_true, seg_pred):
    return - dice_coeff(seg_true, seg_pred)