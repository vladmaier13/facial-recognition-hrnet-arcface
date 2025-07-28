"""The training module for ArcFace face recognition."""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
from argparse import ArgumentParser
from tensorflow import keras
from dataset import build_dataset
from losses import ArcLoss
from network import ArcLayer, L2Normalization, hrnet_v2
from training_supervisor import TrainingSupervisor

parser = ArgumentParser()
parser.add_argument("--softmax", default=False, type=bool, help="Training with softmax loss.")
parser.add_argument("--epochs", default=60, type=int, help="Number of training epochs.")
parser.add_argument("--batch_size", default=128, type=int, help="Training batch size.")
parser.add_argument("--export_only", default=False, type=bool, help="Save the model without training.")
parser.add_argument("--restore_weights_only", default=False, type=bool, help="Only restore the model weights from checkpoint.")
parser.add_argument("--override", default=False, type=bool, help="Manually override the training objects.")
parser.add_argument("--learning_rate", default=0.001, type=float, help="Initial learning rate.")
parser.add_argument("--use_lr_schedule", default=True, type=bool, help="Use learning rate schedule.")
parser.add_argument("--min_lr", default=0.00001, type=float, help="Minimum learning rate.")
parser.add_argument("--warmup_epochs", default=5, type=int, help="Number of warmup epochs.")
parser.add_argument("--decay_type", default="linear", choices=["linear", "exponential"], help="Decay type after warmup")
parser.add_argument("--skip_validation", default=False, type=bool, help="Skip validation during training.")
parser.add_argument("--val_batch_size", default=None, type=int, help="Batch size for validation.")
args = parser.parse_args()

if __name__ == "__main__":
    name = "hrnetv2_softmax+arcface"
    train_files = "D:\\Licenta\\arcface-main\\faces_emore\\split_dataset\\train.record"
    test_files = "D:\\Licenta\\arcface-main\\faces_emore\\split_dataset\\test.record"
    val_files = "D:\\Licenta\\arcface-main\\faces_emore\\split_dataset\\val.record"
    input_shape = (112, 112, 3)
    embedding_size = 512
    num_ids = 100
    num_examples = 4999
    training_dir = os.getcwd()
    export_dir = os.path.join(training_dir, 'exported', name)
    regularizer = keras.regularizers.L2(5e-4)
    frequency = 1000

    if args.restore_weights_only:
        checkpoint_path = tf.train.latest_checkpoint(os.path.join(training_dir, 'checkpoints', name))
        if checkpoint_path:
            checkpoint = tf.train.load_checkpoint(checkpoint_path)
            try:
                restored_lr = checkpoint.get_tensor('schedule/learning_rate/.ATTRIBUTES/VARIABLE_VALUE')
                print(f"Restored LR from checkpoint: {restored_lr}")
                args.learning_rate = float(restored_lr)
            except:
                print("Warning: Could not restore learning rate.")

    base_model = hrnet_v2(
        input_shape=input_shape,
        output_size=embedding_size,
        width=18,
        trainable=True,
        kernel_regularizer=regularizer,
        name="embedding_model"
    )

    if args.softmax:
        print("Building training model with softmax loss...")
        model = keras.Sequential([
            keras.Input(input_shape),
            base_model,
            keras.layers.Dense(num_ids, kernel_regularizer=regularizer),
            keras.layers.Softmax()], 
            name="training_model")
        loss_fun = keras.losses.CategoricalCrossentropy()
    else:
        print("Building training model with ARC loss...")
        model = keras.Sequential([
            keras.Input(input_shape),
            base_model,
            L2Normalization(),
            ArcLayer(num_ids, regularizer)], 
            name="training_model")
        loss_fun = ArcLoss()

    model.summary()

    optimizer = keras.optimizers.Adam(
        learning_rate=args.learning_rate,
        amsgrad=True,
        epsilon=0.001,
        clipvalue=1.0
    )

    dataset_train = build_dataset(
        train_files, 
        batch_size=args.batch_size,
        one_hot_depth=num_ids, 
        training=True, 
        buffer_size=4096
    )

    val_batch_size = args.val_batch_size or args.batch_size
    dataset_val = None
    if val_files and not args.skip_validation:
        try:
            dataset_val = build_dataset(
                val_files,
                batch_size=val_batch_size,
                one_hot_depth=num_ids,
                training=False,
                buffer_size=4096
            )
            val_iter = iter(dataset_val)
            next(val_iter)
        except Exception as e:
            print(f"Validation error: {e}")
            dataset_val = None

    supervisor = TrainingSupervisor(
        model=model,
        optimizer=optimizer,
        loss=loss_fun,
        dataset=dataset_train,
        training_dir=training_dir,
        save_freq=frequency,
        monitor="categorical_accuracy",
        mode='max',
        name=name,
        val_dataset=dataset_val,
        num_examples=num_examples,
        batch_size=args.batch_size,
        epochs=args.epochs,
        warmup_epochs=args.warmup_epochs,
        decay_type=args.decay_type,
        learning_rate=args.learning_rate,
        min_lr=args.min_lr
    )

    if args.export_only:
        supervisor.restore(args.restore_weights_only, from_scout=True)
        supervisor.export(model, export_dir)  # Exportă întregul model
        print("✅ Exportat modelul întreg cu capul final din best checkpoint.")
        quit()

    supervisor.restore(args.restore_weights_only)

    if not args.softmax and args.restore_weights_only:
        with supervisor.clerk.as_default():
            tf.summary.text("⚠️ Faza 2", "Începem faza ArcFace după Softmax", step=int(supervisor.schedule['step']))
            print("📌 Marker TensorBoard scris: Începem faza ArcFace")

        checkpoint_path = tf.train.latest_checkpoint(
            os.path.join(training_dir, 'checkpoints', name)
        )
        if checkpoint_path:
            checkpoint_reader = tf.train.load_checkpoint(checkpoint_path)
            saved_step = checkpoint_reader.get_tensor('schedule/step/.ATTRIBUTES/VARIABLE_VALUE')
            saved_epoch = checkpoint_reader.get_tensor('schedule/epoch/.ATTRIBUTES/VARIABLE_VALUE')
            print(f"Preluăm contoarele din checkpoint: Pas={saved_step}, Epoca={saved_epoch}")
            supervisor.override(step=saved_step, epoch=saved_epoch)

    if args.override:
        supervisor.override(0, 0)

    steps_per_epoch = num_examples // args.batch_size
    supervisor.train(args.epochs, steps_per_epoch)

    if args.softmax:
        print("Saving full Softmax model (with classification head)...")
        model.save(export_dir)
    else:
        print("Saving only ArcFace backbone (embeddings)...")
        supervisor.export(base_model, export_dir)
