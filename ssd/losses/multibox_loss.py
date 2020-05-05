import tensorflow as tf

_policy = tf.keras.mixed_precision.experimental.global_policy()


class LocLoss(tf.losses.Loss):
    def __init__(self, delta, **kwargs):
        super(LocLoss, self).__init__(reduction=tf.losses.Reduction.NONE,
                                      name='loc_loss',
                                      **kwargs)
#         self.smooth_l1 = tf.losses.Huber(delta=delta,
#                                          reduction=tf.losses.Reduction.NONE)

    def call(self, y_true, y_pred):
        #         loss = self.smooth_l1(y_true, y_pred)
        loss = tf.compat.v1.losses.huber_loss(labels=y_true, predictions=y_pred, reduction='none')
        loss = tf.reduce_sum(loss, axis=-1)
        loss = tf.cast(loss, dtype=_policy.compute_dtype)
        return loss


class ClsLoss(tf.losses.Loss):
    def __init__(self, negatives_ratio, **kwargs):
        super(ClsLoss, self).__init__(reduction=tf.losses.Reduction.NONE,
                                      name='cls_loss',
                                      **kwargs)
        self._negatives_ratio = negatives_ratio
        self._softmax_crossentropy = tf.losses.CategoricalCrossentropy(from_logits=True,
                                                                       reduction=tf.losses.Reduction.NONE)

    def _mine_hard_negatives(self, crossentropy, background_mask, negatives_to_keep):

        loss = tf.where(tf.equal(background_mask, 1.0), crossentropy, 0.0)
        sorted_idx = tf.argsort(loss, axis=-1, direction='DESCENDING')
        rank = tf.argsort(sorted_idx, axis=-1)
        negatives_to_keep = tf.expand_dims(negatives_to_keep, axis=1)
        loss = tf.where(tf.less(rank, negatives_to_keep), loss, 0.0)
        return loss

    def _mine_positives(self, crossentropy, background_mask):
        loss = tf.where(tf.not_equal(background_mask, 1.0), crossentropy, 0.0)
        return loss

    def call(self, y_true, y_pred):
        background_mask = tf.cast(tf.equal(y_true[:, :, 0], 1.0), dtype=_policy.compute_dtype)
        num_positives = tf.reduce_sum(1 - background_mask, axis=-1)
        negatives_to_keep = tf.cast(self._negatives_ratio * num_positives, dtype=tf.int32)

        crossentropy = tf.cast(self._softmax_crossentropy(tf.cast(y_true, dtype=tf.float32),
                                                          tf.cast(y_pred, dtype=tf.float32)),
                               dtype=_policy.compute_dtype)

        negative_loss = self._mine_hard_negatives(crossentropy, background_mask, negatives_to_keep)
        postive_loss = self._mine_positives(crossentropy, background_mask)

        loss = tf.reduce_sum(postive_loss + negative_loss, axis=-1)
        loss = tf.cast(loss, dtype=_policy.compute_dtype)
        return loss


class MultiBoxLoss(tf.losses.Loss):
    def __init__(self, config, **kwargs):
        super(MultiBoxLoss, self).__init__(reduction=tf.losses.Reduction.NONE,
                                           name='ssd_loss',
                                           **kwargs)
        self._num_classes = config['num_classes']
        self._cls_loss = ClsLoss(config['negatives_ratio'])
        self._loc_loss = LocLoss(config['smooth_l1_delta'])
        self._cls_loss_weight = config['cls_loss_weight']
        self._loc_loss_weight = config['loc_loss_weight']

    def call(self, y_true, y_pred):
        y_true_loc = y_true[:, :, :4]
        y_true_cls = tf.one_hot(tf.cast(y_true[:, :, 4], dtype=tf.int32),
                                depth=self._num_classes + 1,
                                dtype=_policy.compute_dtype)

        y_pred_loc = y_pred[:, :, :4]
        y_pred_cls = y_pred[:, :, 4:]

        background_mask = tf.cast(tf.equal(y_true_cls[:, :, 0], 1.0), dtype=_policy.compute_dtype)
        num_positives = tf.reduce_sum(1 - background_mask, axis=-1)

        cls_loss = self._cls_loss(y_true_cls, y_pred_cls)

        loc_loss = self._loc_loss(y_true_loc, y_pred_loc)
        loc_loss = tf.where(tf.not_equal(background_mask, 1.0), loc_loss, 0.0)
        loc_loss = tf.reduce_sum(loc_loss, axis=-1)

        cls_loss = tf.math.divide_no_nan(cls_loss, num_positives)
        loc_loss = tf.math.divide_no_nan(loc_loss, num_positives)

        cls_loss = cls_loss * self._cls_loss_weight
        loc_loss = loc_loss * self._loc_loss_weight
        return cls_loss, loc_loss
