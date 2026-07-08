import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from tensorflow.keras.regularizers import l2
from data_split import split_dataset
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay


def train_gesture_cnn(data_dir='UA-dataset',
                      output_dir='dataset_split',
                      train_ratio=0.7,
                      val_ratio=0.15,
                      test_ratio=0.15,
                      img_size=(64, 64),
                      batch_size=32,
                      epochs=40):

    # Validate split ratios sum to 1
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1."

    # Split dataset if not done already
    split_dataset(data_dir, output_dir, train_ratio, val_ratio, test_ratio)

    # ---------------- Data generators ----------------
    # Data augmentation for training
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.15,
        zoom_range=0.15,
        brightness_range=(0.7, 1.3),
        horizontal_flip=True,
        fill_mode='nearest',
        preprocessing_function=lambda x: tf.image.random_crop(x, size=(64, 64, 3))  # random cropping
    )

    train_gen = train_datagen.flow_from_directory(
        os.path.join(output_dir, 'train'),
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True
    )

    val_test_datagen = ImageDataGenerator(rescale=1./255)

    val_gen = val_test_datagen.flow_from_directory(
        os.path.join(output_dir, 'val'),
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )

    # For test set only rescaling (no augmentation)
    test_datagen = ImageDataGenerator(rescale=1./255)
    test_gen = test_datagen.flow_from_directory(
        os.path.join(output_dir, 'test'),
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )

    print("Class indices:", train_gen.class_indices)
    num_classes = train_gen.num_classes

    # Map index->class name in correct order for reports/plots
    idx_to_class = {v: k for k, v in train_gen.class_indices.items()}
    ordered_class_names = [idx_to_class[i] for i in range(num_classes)]

    # --- Optional class weights (computed dynamically) ---
    class_weight = None
    counts = np.bincount(train_gen.classes)
    ratios = counts.max() / counts.min() if counts.min() > 0 else 1.0
    if ratios > 1.5:
        total = counts.sum()
        class_weight = {i: total / (len(counts) * c) for i, c in enumerate(counts)}
        print("Using class_weight:", class_weight)
    else:
        print("Classes close to balanced; not using class_weight.")

    # ---------------- Model ----------------
    model = Sequential([
        Input(shape=(64, 64, 3)),
        Conv2D(32, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        MaxPooling2D(),

        Conv2D(64, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        MaxPooling2D(),

        Conv2D(128, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        MaxPooling2D(),

        Conv2D(256, (3, 3), padding='same', activation='relu', kernel_regularizer=l2(5e-5)),
        BatchNormalization(),

        GlobalAveragePooling2D(),
        Dropout(0.4),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax', kernel_regularizer=l2(5e-5))
    ])

    model.compile(
        loss='categorical_crossentropy',
        optimizer=tf.keras.optimizers.Adam(learning_rate=3e-4),
        metrics=['accuracy']
    )

    # ---- Per-epoch test evaluation for plotting ----
    class TestEvalCallback(tf.keras.callbacks.Callback):
        def __init__(self, test_iterator):
            super().__init__()
            self.test_iterator = test_iterator
            self.test_loss = []
            self.test_acc = []

        def on_epoch_end(self, epoch, logs=None):
            loss, acc = self.model.evaluate(self.test_iterator, verbose=0)
            self.test_loss.append(loss)
            self.test_acc.append(acc)

    test_eval_cb = TestEvalCallback(test_gen)

    # --- Callbacks: always evaluate the best epoch ---
    ckpt_path = 'best_gesture_model.keras'
    callbacks = [
        ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=3, min_lr=1e-5, verbose=1),
        EarlyStopping(monitor='val_accuracy', patience=6, restore_best_weights=True, verbose=1),
        ModelCheckpoint(ckpt_path, monitor='val_accuracy', save_best_only=True, verbose=1),
        test_eval_cb, 
    ]

    # ---------------- Train ----------------
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs,
        class_weight=class_weight,
        callbacks=callbacks
    )

    # ---- Plots: Training vs Test error & Loss per epoch ----
    epochs_r = range(1, len(history.history['accuracy']) + 1) 
    train_err = 1.0 - np.array(history.history['accuracy'])
    val_err   = 1.0 - np.array(history.history['val_accuracy'])
    test_err  = 1.0 - np.array(test_eval_cb.test_acc)

    plt.figure()
    plt.plot(epochs_r, train_err, label='Train error')
    plt.plot(epochs_r, val_err,   label='Val error', linestyle='--')
    plt.plot(epochs_r, test_err,  label='Test error')
    plt.xlabel('Epoch'); plt.ylabel('Error (1 - accuracy)')
    plt.title('Training vs Validation vs Test Error per Epoch')
    plt.grid(True); plt.legend(); plt.tight_layout()
    plt.show()

    plt.figure()
    plt.plot(epochs_r, history.history['loss'], label='Train loss')
    plt.plot(epochs_r, history.history['val_loss'], label='Val loss', linestyle='--')
    plt.plot(epochs_r, test_eval_cb.test_loss, label='Test loss')
    plt.xlabel('Epoch'); plt.ylabel('Loss')
    plt.title('Loss per Epoch')
    plt.grid(True); plt.legend(); plt.tight_layout()
    plt.show()

    # ---------------- Save & final evaluate ----------------
    model.save('hand_gesture_cnn.keras')

    print("\nEvaluating on test set...")
    test_loss, test_acc = model.evaluate(test_gen, verbose=2)
    print(f"Test accuracy: {test_acc*100:.2f}%")

    # --------- Confusion Matrix & Report ---------
    y_pred_probs = model.predict(test_gen)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = test_gen.classes

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=ordered_class_names, digits=4))

    cm = confusion_matrix(y_true, y_pred, normalize='true') * 100.0  # percentages
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=ordered_class_names)
    disp.plot(cmap=plt.cm.Blues, values_format=".1f")
    plt.title("Confusion Matrix - Test Set (%)")
    plt.tight_layout()
    plt.show()

    return model, train_gen.class_indices


if __name__ == '__main__':
    # Example usage:
    model, class_indices = train_gesture_cnn()
