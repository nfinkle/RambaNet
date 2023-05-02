from tensorflow import keras
# from keras import layers
from preprocess import get_sample
import tensorflow as tf
import matplotlib.pyplot as plt
import os

OUTPUTS = len(os.listdir("./raw_dataset/Talmud/organized"))
INPUT_SIZE = 256
MIN_RATIO = 0.65
ALPHABET = 'אבגדהוזחטיכךלמםנןסעפףצץקרשת "\''
ONE_HOT_LEN = len(ALPHABET) + 1

ds_sefaria = tf.data.Dataset.from_generator(lambda: get_sample(input_size=INPUT_SIZE, alphabet=ALPHABET, dataset_directory = "./raw_dataset/Talmud/organized", min_ratio=MIN_RATIO), args=(), output_types=(tf.int8, tf.int8), output_shapes = ( [INPUT_SIZE, ONE_HOT_LEN], [OUTPUTS] ) )
ds_sefaria = ds_sefaria.cache()
counts = [0] * OUTPUTS
for _, label in ds_sefaria:
  o = int(tf.argmax(label).numpy())
  counts[o] += 1
DATASET_SIZE = sum(counts)
#TODO: shapes and arguments should be parametric
# print(len(list(ds_sefaria)))

# inputs = keras.Input(shape=(29, 1024), name='characters')
inputs = keras.Input(shape=(INPUT_SIZE, ONE_HOT_LEN), name='characters')
x = keras.layers.Flatten()(inputs) 
x = keras.layers.Dense(64, activation='relu', name='dense_1')(x)
x = keras.layers.Dense(64, activation='relu', name='dense_2')(x)
x = keras.layers.Dense(OUTPUTS, name='predictions')(x)
outputs = keras.layers.Softmax(name='softmax')(x)

model = keras.Model(inputs=inputs, outputs=outputs)

train_size = int(0.7 * DATASET_SIZE)
val_size = int(0.15 * DATASET_SIZE)
test_size = int(0.15 * DATASET_SIZE)

model.compile(optimizer=keras.optimizers.AdamW(),  # Optimizer
              # Loss function to minimize
              loss='categorical_crossentropy',
              # List of metrics to monitor
              metrics=['accuracy'])

model.summary()


full_dataset = ds_sefaria.shuffle(buffer_size=DATASET_SIZE)
train_dataset = full_dataset.take(train_size)
test_dataset = full_dataset.skip(train_size)
val_dataset = test_dataset.skip(val_size)
test_dataset = test_dataset.take(test_size)

train_dataset = train_dataset.batch(64)
test_dataset = test_dataset.batch(64)
val_dataset = val_dataset.batch(64)

output = model.fit(train_dataset, validation_data=val_dataset, validation_freq=4, epochs=16)

test_predicted_labels = model.predict(test_dataset)
true_labels = []
for batch_of_examples, batch_of_true_labels in test_dataset:
  true_labels.append(batch_of_true_labels)

test_true_labels = tf.argmax(tf.concat(true_labels, axis=0), axis=1)

print(test_true_labels, tf.argmax(test_predicted_labels,axis=1))
plt.imshow(tf.math.confusion_matrix(
    test_true_labels,
    tf.argmax(test_predicted_labels,axis=1),
    num_classes=OUTPUTS,
))

print('\n# Evaluate')
result = model.evaluate(test_dataset)
print(result)
# dict(zip(model.metrics_names, result))