import os
import tensorflow as tf
import argparse
from utils.voc_parser import load_dataset, create_tf_dataset, VOC_CLASSES
from models.ssd_mobilenet import SSDMobileNet

def train_ssd_mobilenet(args):
    # Set random seed for reproducibility
    tf.random.set_seed(42)
    
    # Load datasets
    print(f"Loading training data from {args.train_dir}...")
    train_dataset_raw = load_dataset(args.train_dir, split='trainval')
    
    print(f"Loading validation data from {args.val_dir}...")
    val_dataset_raw = load_dataset(args.val_dir, split='test')
    
    # Create TensorFlow datasets with enhanced augmentation
    train_dataset = create_tf_dataset(
        train_dataset_raw,
        input_size=(args.input_size, args.input_size),
        batch_size=args.batch_size,
        shuffle=True,
        augment=True
    )
    
    val_dataset = create_tf_dataset(
        val_dataset_raw,
        input_size=(args.input_size, args.input_size),
        batch_size=args.batch_size,
        shuffle=False,
        augment=False
    )
    
    # Create model with improved architecture
   
    ssd_model = SSDMobileNet(
        input_shape=(args.input_size, args.input_size, 3),
        num_classes=len(VOC_CLASSES)
    )
    model = ssd_model.build_model()
    
    # Initialize learning rate as a variable so we can modify it in callbacks
    init_lr = args.learning_rate
    
    # Use more advanced optimizer with gradient clipping
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=init_lr,
        clipnorm=1.0  # Gradient clipping to prevent exploding gradients
    )
    
    # Use the improved loss function with focal loss and class weighting
    model.compile(
        optimizer=optimizer,
        loss=ssd_model.ssd_loss(alpha=1.0, gamma=2.0),  # Use focal loss with alpha=1.0, gamma=2.0
        metrics=[['mae'], ['accuracy']]  # Metrics for each output: boxes and classes
    )
    
    # Print model summary
    model.summary()
    
    # Define learning rate schedule function for warm-up
    def lr_scheduler(epoch, lr):
        if epoch < 3:
            # Warmup for first 3 epochs
            return init_lr * min(1.0, (epoch + 1) / 3)
        return lr  # Return current lr for other epochs (handled by ReduceLROnPlateau)
    
    # Define enhanced callbacks with better learning rate scheduling
    callbacks = [
        # Save weights after each epoch
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(args.output_dir, 'ssd_mobilenet_weights_{epoch:02d}.weights.h5'),
            save_best_only=True,
            save_weights_only=True,
            monitor='val_loss'
        ),
        # Learning rate scheduler for warmup and other adjustments
        tf.keras.callbacks.LearningRateScheduler(lr_scheduler),
        # More aggressive learning rate reduction
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,  # More aggressive reduction
            patience=3,
            min_lr=1e-7,
            verbose=1
        ),
        # Early stopping
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        # TensorBoard logging
        tf.keras.callbacks.TensorBoard(
            log_dir=os.path.join(args.output_dir, 'logs'),
            write_graph=True,
            profile_batch=0  # Disable profiling for better performance
        )
    ]
    
    # Train model with more epochs
    print(f"Starting training for {args.epochs} epochs with improved model and training process...")
    
    # Calculate steps per epoch based on dataset size and batch size
    steps_per_epoch = len(train_dataset_raw) // args.batch_size
    validation_steps = len(val_dataset_raw) // args.batch_size
    
    # Ensure at least 1 step per epoch
    steps_per_epoch = max(1, steps_per_epoch)
    validation_steps = max(1, validation_steps)
    
    print(f"Training with {steps_per_epoch} steps per epoch and {validation_steps} validation steps")
    
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=args.epochs,
        callbacks=callbacks,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        verbose=1
    )
    
    # Save final model using the recommended Keras format
    final_model_path = os.path.join(args.output_dir, 'ssd_mobilenet_final.keras')
    print(f"Saving final model to {final_model_path}")
    model.save(final_model_path, save_format='keras')
    
    # Create and save inference model using the recommended Keras format
    inference_model = ssd_model.build_inference_model()
    inference_model_path = os.path.join(args.output_dir, 'ssd_mobilenet_inference.keras')
    print(f"Saving inference model to {inference_model_path}")
    inference_model.save(inference_model_path, save_format='keras')
    
    print("Training completed successfully!")

def main():
    parser = argparse.ArgumentParser(description='Train SSD MobileNet on PASCAL VOC dataset')
    parser.add_argument('--train-dir', type=str, default='archive/VOCtrainval_06-Nov-2007',
                        help='Directory containing training data')
    parser.add_argument('--val-dir', type=str, default='archive/VOCtest_06-Nov-2007',
                        help='Directory containing validation data')
    parser.add_argument('--output-dir', type=str, default='output',
                        help='Directory to save model weights and logs')
    parser.add_argument('--input-size', type=int, default=300,  # Increase input size for better detection
                        help='Input image size')
    parser.add_argument('--batch-size', type=int, default=8,
                        help='Training batch size')
    parser.add_argument('--epochs', type=int, default=100,  # Increase epochs
                        help='Number of epochs to train')
    parser.add_argument('--learning-rate', type=float, default=2e-4,  # Slightly higher learning rate
                        help='Initial learning rate')
    parser.add_argument('--workers', type=int, default=4,
                        help='Number of worker processes for data loading')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Start training
    train_ssd_mobilenet(args)

if __name__ == '__main__':
    main()