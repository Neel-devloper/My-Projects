import sqlite3
import numpy as np
import os

# Database file
db_file = 'neural_net.db'

# Remove the database file if it exists and is corrupted
if os.path.exists(db_file):
    try:
        conn = sqlite3.connect(db_file)
        conn.close()
    except sqlite3.DatabaseError:
        print("Database is malformed, deleting and creating a new one...")
        os.remove(db_file)

# Connect to SQLite database (creates a new one if deleted or doesn't exist)
try:
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()

    # Drop existing tables to ensure a fresh start
    cursor.execute("DROP TABLE IF EXISTS hidden_weights")
    cursor.execute("DROP TABLE IF EXISTS hidden_biases")
    cursor.execute("DROP TABLE IF EXISTS output_weights")
    cursor.execute("DROP TABLE IF EXISTS output_biases")

    # Create tables with appropriate schemas
    cursor.execute("""
    CREATE TABLE hidden_weights (
        row INTEGER,
        col INTEGER,
        value REAL,
        PRIMARY KEY (row, col)
    )
    """)

    cursor.execute("""
    CREATE TABLE hidden_biases (
        idx INTEGER PRIMARY KEY,
        value REAL
    )
    """)

    cursor.execute("""
    CREATE TABLE output_weights (
        row INTEGER,
        col INTEGER,
        value REAL,
        PRIMARY KEY (row, col)
    )
    """)

    cursor.execute("""
    CREATE TABLE output_biases (
        idx INTEGER PRIMARY KEY,
        value REAL
    )
    """)

    # Generate random weights and biases from uniform distribution between -1 and 1
    hidden_weights = np.random.uniform(-1, 1, (128, 784))  # 128 x 784 matrix
    hidden_biases = np.random.uniform(-1, 1, 128)          # 128 vector
    output_weights = np.random.uniform(-1, 1, (128, 10))   # 128 x 10 matrix
    output_biases = np.random.uniform(-1, 1, 10)           # 10 vector

    # Prepare data as lists of tuples for efficient insertion
    hidden_weights_data = [(i, j, hidden_weights[i, j]) for i in range(128) for j in range(784)]
    hidden_biases_data = [(i, hidden_biases[i]) for i in range(128)]
    output_weights_data = [(i, j, output_weights[i, j]) for i in range(128) for j in range(10)]
    output_biases_data = [(i, output_biases[i]) for i in range(10)]

    # Insert data into tables using executemany for efficiency
    cursor.executemany("INSERT INTO hidden_weights (row, col, value) VALUES (?, ?, ?)", hidden_weights_data)
    cursor.executemany("INSERT INTO hidden_biases (idx, value) VALUES (?, ?)", hidden_biases_data)
    cursor.executemany("INSERT INTO output_weights (row, col, value) VALUES (?, ?, ?)", output_weights_data)
    cursor.executemany("INSERT INTO output_biases (idx, value) VALUES (?, ?)", output_biases_data)

    # Commit changes
    conn.commit()

except sqlite3.DatabaseError as e:
    print(f"Database error: {e}")
finally:
    # Close the connection
    conn.close()
