"""This module provides the implementation of training supervisor."""
import os
import tensorflow as tf
from tqdm import tqdm

class TrainingSupervisor(object):
    """A training supervisor will organize and monitor the training process."""
    def __init__(self, model, optimizer, loss, dataset, training_dir, save_freq, 
                 monitor, mode, name, val_dataset=None, lr_schedule=None, num_examples=None, batch_size=None, 
                 epochs=None, warmup_epochs=5, decay_type='linear', learning_rate=0.001, min_lr=1e-5) -> None:
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.loss_fun = loss
        self.dataset = dataset
        self.data_generator = iter(self.dataset)
        self.val_dataset = val_dataset
        self.save_freq = save_freq
        self.training_dir = training_dir
        self.name = name
        self.eval_freq = min(save_freq // 5, 200)

        # Metrics setup
        self.metrics = {
            'categorical_accuracy': tf.keras.metrics.CategoricalAccuracy(name='train_accuracy'),
            'loss': tf.keras.metrics.Mean(name="train_loss_mean")
        }
        self.val_metrics = None
        if val_dataset is not None:
            self.val_metrics = {
                'val_categorical_accuracy': tf.keras.metrics.CategoricalAccuracy(name='val_accuracy'),
                'val_loss': tf.keras.metrics.Mean(name='val_loss_mean')
            }

        # Monitoring setup
        self.monitor = self.metrics[monitor]
        self.mode = mode

        # Training state
        self.schedule = {
            'step': tf.Variable(0, trainable=False, dtype=tf.int64),
            'epoch': tf.Variable(0, trainable=False, dtype=tf.int64),
            'monitor_value': tf.Variable(-1.0, trainable=False, dtype=tf.float32),
            'learning_rate': tf.Variable(optimizer.lr.numpy(), trainable=False, dtype=tf.float32)
        }

        # Learning rate schedule setup (optional)
        self.lr_schedule = lr_schedule
        if lr_schedule is None and num_examples is not None and batch_size is not None and epochs is not None:
            total_steps = num_examples // batch_size * epochs
            warmup_steps = num_examples // batch_size * warmup_epochs
            total_decay_steps = max(total_steps - warmup_steps, 1)

            def lr_schedule_fn(step):
                step = tf.cast(step, tf.float32)
                if step < warmup_steps:
                    return (step / warmup_steps) * learning_rate
                else:
                    decay_progress = (step - warmup_steps) / total_decay_steps
                    if decay_type == "linear":
                        lr = learning_rate * (1 - decay_progress)
                    else:
                        lr = learning_rate * tf.math.exp(-decay_progress)
                    return tf.maximum(lr, min_lr)

            self.lr_schedule = lr_schedule_fn

        # Checkpoint setup
        self.checkpoint = tf.train.Checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            metrics=self.metrics,
            schedule=self.schedule,
            monitor=self.monitor,
            data_generator=self.data_generator
        )
        if self.val_metrics is not None:
            self.checkpoint.val_metrics = self.val_metrics

        self.manager = tf.train.CheckpointManager(
            self.checkpoint, 
            os.path.join(training_dir, 'checkpoints', name),
            max_to_keep=2
        )
        self.scout = tf.train.CheckpointManager(
            self.checkpoint,
            os.path.join(training_dir, 'model_scout', name),
            max_to_keep=1
        )
        self.clerk = tf.summary.create_file_writer(
            os.path.join(training_dir, 'logs', name)
        )

    def restore(self, weights_only=False, from_scout=False):
        checkpoint_path = self.scout.latest_checkpoint if from_scout else self.manager.latest_checkpoint
        if not checkpoint_path:
            print("No checkpoint found. Starting from scratch.")
            return
        print(f"Restoring from {'scout' if from_scout else 'regular'} checkpoint: {checkpoint_path}")
        if weights_only:
            tf.train.Checkpoint(model=self.model).restore(checkpoint_path)
        else:
            status = self.checkpoint.restore(checkpoint_path)
            status.expect_partial()
            self.optimizer.lr.assign(self.schedule['learning_rate'].numpy())
            print(f"Restored learning rate: {self.optimizer.lr.numpy()}")

    @tf.function
    def _train_step(self, x_batch, y_batch, step):
        if self.lr_schedule is not None:
            new_lr = self.lr_schedule(step)
            self.optimizer.lr.assign(new_lr)
            self.schedule['learning_rate'].assign(new_lr)

        with tf.GradientTape() as tape:
            logits = self.model(x_batch, training=True)
            loss = self.loss_fun(y_batch, logits) + sum(self.model.losses)

        grads = tape.gradient(loss, self.model.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
        return logits, loss

    @tf.function
    def _update_metrics(self, labels, logits, loss):
        self.metrics['categorical_accuracy'].update_state(labels, logits)
        self.metrics['loss'].update_state(loss)

    @tf.function
    def _evaluate_step(self, x, y):
        logits = self.model(x, training=False)
        loss = self.loss_fun(y, logits) + sum(self.model.losses)
        self.val_metrics['val_categorical_accuracy'].update_state(y, logits)
        self.val_metrics['val_loss'].update_state(loss)
        return loss

    def _reset_metrics(self, reset_val=True):
        for metric in self.metrics.values():
            metric.reset_states()
        if reset_val and self.val_metrics is not None:
            for metric in self.val_metrics.values():
                metric.reset_states()

    def _evaluate_model(self):
        if self.val_dataset is None:
            return None
        print(f"\n[Epoch {int(self.schedule['epoch']) + 1}] Starting validation...")
        self._reset_metrics(reset_val=True)
        for x_batch, y_batch in iter(self.val_dataset):
            self._evaluate_step(x_batch, y_batch)
        val_loss = self.val_metrics['val_loss'].result()
        val_acc = self.val_metrics['val_categorical_accuracy'].result()
        print(f"[Validation] Accuracy: {val_acc:.4f}, Loss: {val_loss:.2f}")
        return (val_loss, val_acc)

    def _log_to_tensorboard(self, evaluate=True):
        current_step = int(self.schedule['step'])
        train_loss = self.metrics['loss'].result()
        train_acc = self.metrics['categorical_accuracy'].result()
        lr = self.optimizer.lr.numpy()

        val_results = None
        if evaluate and self.val_metrics is not None:
            val_results = self._evaluate_model()

        with self.clerk.as_default():
            tf.summary.scalar("training/loss", train_loss, step=current_step)
            tf.summary.scalar("training/accuracy", train_acc, step=current_step)
            tf.summary.scalar("training/learning_rate", lr, step=current_step)
            if val_results:
                val_loss, val_acc = val_results
                tf.summary.scalar("validation/loss", val_loss, step=current_step)
                tf.summary.scalar("validation/accuracy", val_acc, step=current_step)

        print(f"\n[Step {current_step}]")
        print(f"LR: {lr:.7f} | Train Acc: {train_acc:.4f} | Train Loss: {train_loss:.2f}")
        if val_results:
            print(f"Val Acc: {val_acc:.4f} | Val Loss: {val_loss:.2f}")

    def _checkpoint(self):
        current = self.val_metrics['val_categorical_accuracy'].result()
        previous = self.schedule['monitor_value'].numpy()

        if previous < 0.0 or (current > previous and self.mode == 'max') or (current < previous and self.mode == 'min'):
            print(f"Monitor improved ({previous:.4f} → {current:.4f}). Saving best checkpoint...")
            self.schedule['monitor_value'].assign(current)
            self.scout.save()

        self.manager.save()
        print(f"Checkpoint saved at step {int(self.schedule['step'])}")

    def _save_model(self, export_dir):
        try:
            self.model.save(export_dir, save_format='tf')
        except Exception as e:
            print(f"Failed to save full model to {export_dir}: {e}")

    def train(self, epochs, steps_per_epoch):
        initial_epoch = int(self.schedule['epoch'].numpy())
        total_epochs = initial_epoch + epochs

        print(f"\n=== Training Summary ===")
        print(f"Starting from epoch: {initial_epoch + 1}")
        print(f"Training for {epochs} epochs (total: {total_epochs} epochs)")
        print(f"Steps per epoch: {steps_per_epoch}")
        print(f"Validation: {'enabled' if self.val_dataset else 'disabled'}\n")

        for epoch in range(initial_epoch, total_epochs):
            print(f"\n=== Epoch {epoch + 1}/{total_epochs} ===")
            progress_bar = tqdm(total=steps_per_epoch, 
                                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]',
                                colour='green',
                                dynamic_ncols=True,
                                leave=False)

            for step, (x_batch, y_batch) in enumerate(self.data_generator):
                logits, loss = self._train_step(x_batch, y_batch, self.schedule['step'])
                self._update_metrics(y_batch, logits, loss)
                self.schedule['step'].assign_add(1)

                progress_bar.update(1)
                progress_bar.set_postfix({
                    "loss": f"{loss.numpy():.2f}",
                    "acc": f"{self.metrics['categorical_accuracy'].result().numpy():.3f}",
                    "lr": f"{self.optimizer.lr.numpy():.6f}"
                })

                if (step + 1) % self.eval_freq == 0:
                    self._log_to_tensorboard()
                    self._checkpoint()
                if (step + 1) % self.save_freq == 0:
                    self._checkpoint()

            self.schedule['epoch'].assign_add(1)
            self.data_generator = iter(self.dataset)
            self._log_to_tensorboard()
            self._checkpoint()
            progress_bar.close()

        print("\n=== Training Completed ===")

    def export(self, model, export_dir):
        print(f"\nExporting model to {export_dir}...")
        model.save(export_dir, save_format='tf')
        print("Model exported successfully!")

    def override(self, step=None, epoch=None, monitor_value=None):
        if step is not None:
            self.schedule['step'].assign(step)
        if epoch is not None:
            self.schedule['epoch'].assign(epoch)
        if monitor_value is not None:
            self.schedule['monitor_value'].assign(monitor_value)