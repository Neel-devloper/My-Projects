import sqlite3
import numpy as np
from tensorflow import keras
import random


def load_from_db():
    db_file = 'neural_net.db'
    try:
        conn = sqlite3.connect(db_file, timeout=10)
        cursor = conn.cursor()

        # Load hidden weights (128 x 784)
        cursor.execute("SELECT row, col, value FROM hidden_weights")
        hidden_weights_data = cursor.fetchall()
        hidden_weights = np.zeros((128, 784))
        for row, col, value in hidden_weights_data:
            hidden_weights[row, col] = value

        # Load hidden biases (128)
        cursor.execute("SELECT idx, value FROM hidden_biases")
        hidden_biases_data = cursor.fetchall()
        hidden_biases = np.zeros(128)
        for idx, value in hidden_biases_data:
            hidden_biases[idx] = value

        # Load output weights (128 x 10)
        cursor.execute("SELECT row, col, value FROM output_weights")
        output_weights_data = cursor.fetchall()
        output_weights = np.zeros((128, 10))
        for row, col, value in output_weights_data:
            output_weights[row, col] = value

        # Load output biases (10)
        cursor.execute("SELECT idx, value FROM output_biases")
        output_biases_data = cursor.fetchall()
        output_biases = np.zeros(10)
        for idx, value in output_biases_data:
            output_biases[idx] = value

        return hidden_weights, hidden_biases, output_weights, output_biases

    except sqlite3.DatabaseError as e:
        print(f"Database error: {e}")
        if "disk I/O error" in str(e):
            print("Disk I/O error detected. Check disk space, permissions, or file locks.")
        return (np.random.uniform(-1, 1, (128, 784)),
                np.random.uniform(-1, 1, 128),
                np.random.uniform(-1, 1, (128, 10)),
                np.random.uniform(-1, 1, 10))
    finally:
        conn.close()


def read_img(img_idx: int) -> tuple:
    (x_train, y_train), (_, _) = keras.datasets.mnist.load_data()
    image = x_train[img_idx]
    label = y_train[img_idx]
    image = image / 255.0
    vector = image.flatten().tolist()
    return vector, label


def softmax(prediction_vector: list) -> list:
    exp_vec = [np.exp(x) for x in prediction_vector]
    sum_exp = sum(exp_vec)
    return [x / sum_exp for x in exp_vec]


def front_prop(hidden_layer_weights, hidden_layer_biases, output_layer_weights, output_layer_biases, img_vector):
    img_vector = np.array(img_vector)
    hidden_weights = np.array(hidden_layer_weights)
    hidden_biases = np.array(hidden_layer_biases)
    output_weights = np.array(output_layer_weights)
    output_biases = np.array(output_layer_biases)

    hidden_input = np.dot(hidden_weights, img_vector) + hidden_biases
    hidden_output = np.maximum(0, hidden_input)
    output_input = np.dot(output_weights.T, hidden_output) + output_biases
    output = softmax(output_input.tolist())

    return output


# Load weights and biases from database
hidden_weights, hidden_biases, output_weights, output_biases = load_from_db()

# Load MNIST dataset
(x_train, y_train), (_, _) = keras.datasets.mnist.load_data()

# Select 20 random image indices
num_images = 20
correct = 0
img_indices = random.sample(range(len(x_train)), num_images)

# Process and predict for each image
for idx, img_idx in enumerate(img_indices):
    img_vector, true_label = read_img(img_idx)
    predicted = front_prop(hidden_weights, hidden_biases, output_weights, output_biases, img_vector)

    # Display predictions
    print(f"\nImage {idx + 1} (Index: {img_idx})")
    print(f"True Label: {true_label}")
    print("Predictions (probabilities for digits 0-9):")
    for digit, prob in enumerate(predicted):
        print(f"Digit {digit}: {prob:.4f}")
    print(f"Predicted Digit: {np.argmax(predicted)}")
    
    if int(np.argmax(predicted)) == int(true_label):
        correct += 1

print(f'Accuracy : {(correct/num_images)*100}%')
