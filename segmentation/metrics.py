
import tensorflow as tf

def _get_class(seg, pred_class):
    size = [-1] * 5
    begin = [0] * 5

    size[pred_class] = 1
    begin[1] = pred_class

    _seg = tf.slice(seg, begin=begin, size=size)
    return _seg



def dice_coeff(seg_true, seg_pred, pred_class=1):
    with tf.variable_scope("dice_coff_loss"):
        tf.assert_equal(tf.shape(seg_true), tf.shape(seg_pred))

        seg_true_flat = tf.layers.flatten(_get_class(seg_true, pred_class))
        seg_pred_flat = tf.layers.flatten(_get_class(seg_pred, pred_class))

        intersection = tf.multiply(seg_true_flat, seg_pred_flat, name="intersection")
        intersect = tf.reduce_sum(intersection)

        smooth = 0.01
        dice = tf.divide(2 * intersect + smooth, tf.reduce_sum(seg_true_flat) + tf.reduce_sum(seg_pred_flat) + smooth)
        return dice


def dice_loss(seg_true, seg_pred):
    return - dice_coeff(seg_true, seg_pred)