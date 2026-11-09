"""Train a compact 1D-CNN on labeled vibration recordings."""
from pathlib import Path
import json
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from preprocess import prepare_data, WINDOW

np.random.seed(42)
tf.random.set_seed(42)
(splits, mean, std) = prepare_data()
(x_train, y_train), (x_val, y_val), (x_test, y_test) = splits
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(WINDOW, 1)),
    tf.keras.layers.Conv1D(16, 7, activation='relu', padding='same'),
    tf.keras.layers.MaxPooling1D(4),
    tf.keras.layers.Conv1D(32, 5, activation='relu', padding='same'),
    tf.keras.layers.MaxPooling1D(4),
    tf.keras.layers.Conv1D(32, 3, activation='relu', padding='same'),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(1, activation='sigmoid'),
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()
Path('models').mkdir(exist_ok=True)
model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=20, batch_size=32,
          callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True)])
probabilities = model.predict(x_test, verbose=0).ravel()
predictions = (probabilities >= 0.5).astype(int)
print('\nHELD-OUT TEST RESULTS (synthetic if demo generator used)')
print(classification_report(y_test, predictions, target_names=['normal', 'abnormal'], zero_division=0))
print('Confusion matrix:\n', confusion_matrix(y_test, predictions))
model.save('models/hackedge_vibration_cwru.keras')
Path('models/normalization_cwru.json').write_text(json.dumps({'mean': mean, 'std': std, 'window': WINDOW}, indent=2))
print('Saved model and normalization settings in models/')
