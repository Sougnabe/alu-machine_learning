#!/usr/bin/env python3
"""Training utility for machine translation transformer."""
import tensorflow.compat.v2 as tf

Dataset = __import__('3-dataset').Dataset
create_masks = __import__('4-create_masks').create_masks
Transformer = __import__('5-transformer').Transformer


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    """Implements transformer warmup learning-rate schedule."""

    def __init__(self, dm, warmup_steps=4000):
        """Class constructor."""
        super(CustomSchedule, self).__init__()
        self.dm = tf.cast(dm, tf.float32)
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        """Returns learning rate for a training step."""
        step = tf.cast(step, tf.float32)
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)
        return tf.math.rsqrt(self.dm) * tf.math.minimum(arg1, arg2)


def loss_function(real, pred):
    """Computes sparse categorical crossentropy ignoring padding."""
    loss_obj = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True,
        reduction='none'
    )
    mask = tf.cast(tf.not_equal(real, 0), tf.float32)
    loss = loss_obj(real, pred)
    loss *= mask
    return tf.reduce_sum(loss) / tf.reduce_sum(mask)


def train_transformer(N, dm, h, hidden, max_len, batch_size, epochs):
    """Creates and trains a transformer for Portuguese to English."""
    dataset = Dataset(batch_size, max_len)

    input_vocab = dataset.tokenizer_pt.vocab_size + 2
    target_vocab = dataset.tokenizer_en.vocab_size + 2

    transformer = Transformer(
        N,
        dm,
        h,
        hidden,
        input_vocab,
        target_vocab,
        max_len,
        max_len
    )

    learning_rate = CustomSchedule(dm)
    optimizer = tf.keras.optimizers.Adam(
        learning_rate,
        beta_1=0.9,
        beta_2=0.98,
        epsilon=1e-9
    )

    for epoch in range(epochs):
        train_loss = tf.keras.metrics.Mean()
        train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()

        for batch, (inputs, target) in enumerate(dataset.data_train):
            target_inp = target[:, :-1]
            target_real = target[:, 1:]

            encoder_mask, combined_mask, decoder_mask = create_masks(
                inputs,
                target_inp
            )

            with tf.GradientTape() as tape:
                predictions = transformer(
                    inputs,
                    target_inp,
                    True,
                    encoder_mask,
                    combined_mask,
                    decoder_mask
                )
                loss = loss_function(target_real, predictions)

            gradients = tape.gradient(loss, transformer.trainable_variables)
            optimizer.apply_gradients(
                zip(gradients, transformer.trainable_variables)
            )

            train_loss.update_state(loss)
            mask = tf.cast(tf.not_equal(target_real, 0), tf.float32)
            train_accuracy.update_state(
                target_real,
                predictions,
                sample_weight=mask
            )

            if batch % 50 == 0:
                print('Epoch {}, batch {}: loss {} accuracy {}'.format(
                    epoch + 1,
                    batch,
                    train_loss.result().numpy(),
                    train_accuracy.result().numpy()
                ))

        print('Epoch {}: loss {} accuracy {}'.format(
            epoch + 1,
            train_loss.result().numpy(),
            train_accuracy.result().numpy()
        ))

    return transformer
