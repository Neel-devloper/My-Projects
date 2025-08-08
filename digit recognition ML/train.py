import sqlite3
from tensorflow import keras
import numpy as np


def load_from_db():
    db_file = 'neural_net.db'
    try:
        conn = sqlite3.connect(db_file)
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
        return (np.random.uniform(-1, 1, (128, 784)),
                np.random.uniform(-1, 1, 128),
                np.random.uniform(-1, 1, (128, 10)),
                np.random.uniform(-1, 1, 10))
    finally:
        conn.close()


def read_img(img_idx: int) -> list:
    (x_train, _), (_, _) = keras.datasets.mnist.load_data()
    image = x_train[img_idx]
    image = image / 255.0
    vector = image.flatten().tolist()
    return vector


def softmax(prediction_vector: list) -> list:
    exp_vec = [np.exp(x) for x in prediction_vector]
    sum_exp = sum(exp_vec)
    return [x / sum_exp for x in exp_vec]


def cost_function(actual, predicted, epsilon=1e-10):
    actual = np.array(actual)
    predicted = np.array(predicted)
    assert actual.shape == predicted.shape, "Shapes of actual and predicted must match"
    predicted = np.clip(predicted, epsilon, 1 - epsilon)
    if actual.ndim == 1:
        loss = -np.sum(actual * np.log(predicted))
    else:
        loss = -np.mean(np.sum(actual * np.log(predicted), axis=1))
    return loss


def store_to_db(hidden_layer_weights, hidden_layer_biases, output_layer_weights, output_layer_biases):
    db_file = 'neural_net.db'
    try:
        conn = sqlite3.connect(db_file)
        cursor = conn.cursor()

        # Convert lists to NumPy arrays
        hidden_layer_weights = np.array(hidden_layer_weights)
        hidden_layer_biases = np.array(hidden_layer_biases)
        output_layer_weights = np.array(output_layer_weights)
        output_layer_biases = np.array(output_layer_biases)

        # Clear existing data
        cursor.execute("DELETE FROM hidden_weights")
        cursor.execute("DELETE FROM hidden_biases")
        cursor.execute("DELETE FROM output_weights")
        cursor.execute("DELETE FROM output_biases")

        # Prepare data for insertion
        hidden_weights_data = [(i, j, hidden_layer_weights[i, j])
                               for i in range(hidden_layer_weights.shape[0])
                               for j in range(hidden_layer_weights.shape[1])]
        hidden_biases_data = [(i, hidden_layer_biases[i])
                              for i in range(hidden_layer_biases.shape[0])]
        output_weights_data = [(i, j, output_layer_weights[i, j])
                               for i in range(output_layer_weights.shape[0])
                               for j in range(output_layer_weights.shape[1])]
        output_biases_data = [(i, output_layer_biases[i])
                              for i in range(output_layer_biases.shape[0])]

        # Insert new data
        cursor.executemany("INSERT INTO hidden_weights (row, col, value) VALUES (?, ?, ?)", hidden_weights_data)
        cursor.executemany("INSERT INTO hidden_biases (idx, value) VALUES (?, ?)", hidden_biases_data)
        cursor.executemany("INSERT INTO output_weights (row, col, value) VALUES (?, ?, ?)", output_weights_data)
        cursor.executemany("INSERT INTO output_biases (idx, value) VALUES (?, ?)", output_biases_data)

        # Commit changes
        conn.commit()

    except sqlite3.DatabaseError as e:
        print(f"Database error: {e}")
    finally:
        conn.close()


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

    return output, hidden_output


def back_prop(hidden_layer_weights, hidden_layer_biases, output_layer_weights, output_layer_biases, actual_vector,
              predicted_vector, img_vector, hidden_output, learning_rate=0.01):
    hidden_weights = np.array(hidden_layer_weights)
    hidden_biases = np.array(hidden_layer_biases)
    output_weights = np.array(output_layer_weights)
    output_biases = np.array(output_layer_biases)
    actual = np.array(actual_vector)
    predicted = np.array(predicted_vector)
    img_vector = np.array(img_vector)
    hidden_output = np.array(hidden_output)

    output_error = predicted - actual
    output_weights_grad = np.dot(hidden_output.reshape(-1, 1), output_error.reshape(1, -1))
    output_biases_grad = output_error

    hidden_error = np.dot(output_weights, output_error)
    hidden_error = hidden_error * (hidden_output > 0)

    hidden_weights_grad = np.dot(hidden_error.reshape(-1, 1), img_vector.reshape(1, -1))
    hidden_biases_grad = hidden_error

    hidden_weights -= learning_rate * hidden_weights_grad
    hidden_biases -= learning_rate * hidden_biases_grad
    output_weights -= learning_rate * output_weights_grad
    output_biases -= learning_rate * output_biases_grad

    return (hidden_weights.tolist(), hidden_biases.tolist(),
            output_weights.tolist(), output_biases.tolist())


# Load weights and biases from database
hidden_weights, hidden_biases, output_weights, output_biases = load_from_db()

# Load MNIST dataset
(x_train, y_train), (_, _) = keras.datasets.mnist.load_data()

print('This training may take time')
print('2-3 hours')

# Process only the first 40000 images and update weights/biases
for img_idx in range(40000):
    img_vector = read_img(img_idx)
    actual = np.zeros(10)
    actual[y_train[img_idx]] = 1.0

    # Forward propagation
    predicted, hidden_output = front_prop(hidden_weights, hidden_biases, output_weights, output_biases, img_vector)

    # Backpropagation
    hidden_weights, hidden_biases, output_weights, output_biases = back_prop(
        hidden_weights, hidden_biases, output_weights, output_biases,
        actual, predicted, img_vector, hidden_output
    )

    # Store updated weights and biases
    store_to_db(hidden_weights, hidden_biases, output_weights, output_biases)
    print(f'Gone through {img_idx} images')
