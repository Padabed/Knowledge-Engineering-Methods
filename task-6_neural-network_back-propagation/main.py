from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()

animal_vehicle_classes = [2, 3, 4, 5, 6, 7, 8, 9]
animal_vehicle_train_indices = []
animal_vehicle_test_indices = []

for i in range(len(train_labels)):
    if train_labels[i][0] in animal_vehicle_classes:
        animal_vehicle_train_indices.append(i)

for i in range(len(test_labels)):
    if test_labels[i][0] in animal_vehicle_classes:
        animal_vehicle_test_indices.append(i)

train_images = train_images[animal_vehicle_train_indices]
train_labels = train_labels[animal_vehicle_train_indices]
test_images = test_images[animal_vehicle_test_indices]
test_labels = test_labels[animal_vehicle_test_indices]

# Normalize to range [0, 1]
train_images = train_images.astype('float32') / 255.0
test_images = test_images.astype('float32') / 255.0

# binary conv -> 0 for animal, 1 for vehicle
train_labels = (train_labels >= 4).astype(int)
test_labels = (test_labels >= 4).astype(int)


classifiers = [
    (1, [32]),
    (2, [32, 64]),
    (3, [32, 64, 128])
]

# train
for num_layers, filters in classifiers:
    print(f"Training classifier with {num_layers} convolutional layer(s)...")

    model = Sequential()
    model.add(Conv2D(filters[0], (3, 3), activation='relu', input_shape=(32, 32, 3)))
    model.add(MaxPooling2D((2, 2)))

    for i in range(1, num_layers):
        model.add(Conv2D(filters[i], (3, 3), activation='relu'))
        model.add(MaxPooling2D((2, 2)))

    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))  # Binary classification

    # Compile and train the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(train_images, train_labels, epochs=10, batch_size=64, verbose=1)

    # Evaluate the model on test data
    test_loss, test_accuracy = model.evaluate(test_images, test_labels, verbose=0)
    print(f"Test accuracy with {num_layers} convolutional layer(s): {test_accuracy:.4f}")

    print()
