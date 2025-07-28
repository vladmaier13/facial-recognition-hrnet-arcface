import tensorflow as tf
from math import pi

class ArcLoss(tf.keras.losses.Loss):
    def __init__(self, margin=0.5, scale=64, name="arcloss"):
        super().__init__(name=name)
        self.margin = margin
        self.scale = scale
        self.threshold = tf.math.cos(pi - margin)
        self.cos_m = tf.math.cos(margin)
        self.sin_m = tf.math.sin(margin)
        self.safe_margin = tf.constant(0.2, dtype=tf.float32)  # Valoare fixă pentru stabilitate

    @tf.function
    def call(self, y_true, y_pred):
        cos_t = y_pred
        sin_t = tf.math.sqrt(1 - tf.math.square(cos_t))
        cos_t_margin = tf.where(
            cos_t > self.threshold,
            cos_t * self.cos_m - sin_t * self.sin_m,
            cos_t - self.safe_margin
        )
        
        mask = y_true
        cos_t_onehot = cos_t * mask
        cos_t_margin_onehot = cos_t_margin * mask
        logits = (cos_t + cos_t_margin_onehot - cos_t_onehot) * self.scale
        return tf.nn.softmax_cross_entropy_with_logits(y_true, logits)

    def get_config(self):
        config = super().get_config()
        config.update({"margin": self.margin, "scale": self.scale})
        return config