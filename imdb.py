# Step 1: Load the IMDB dataset
from keras.datasets import imdb
from keras.preprocessing.sequence import pad_sequences

# Load the IMDB dataset with the top 10,000 most frequent words
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=10000)

# Step 2: Pad or truncate the sequences to a fixed length of 250 words
max_len = 250
X_train = pad_sequences(X_train, maxlen=max_len)
X_test = pad_sequences(X_test, maxlen=max_len)

# Step 3: Define the deep neural network architecture
from keras.models import Sequential
from keras.layers import Embedding, Bidirectional, LSTM, Dense
embedding_dim = 128
model = Sequential()
model.add(Embedding(input_dim=10000, output_dim=embedding_dim, input_length=max_len))
model.add(Bidirectional(LSTM(64, return_sequences=True)))
model.add(Bidirectional(LSTM(64)))
model.add(Dense(1, activation='sigmoid'))

# Step 4: Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Step 5: Train the model
history = model.fit(X_train, y_train, batch_size=128, epochs=3, validation_split=0.2)

# Step 6: Evaluate the trained model
loss, accuracy = model.evaluate(X_test, y_test,batch_size=128)
print("Test Loss:", loss)
print("Test Accuracy:", accuracy)

# Example
# Get word index mapping
word_index = imdb.get_word_index()

# Reverse the word index mapping
index_to_word = {index: word for word, index in word_index.items()}

# Choose a subset of the test data
import numpy as np
subset_size = 10
subset_indices = np.random.choice(len(X_test), size=subset_size, replace=False)
subset_X = X_test[subset_indices]
subset_y_true = y_test[subset_indices]

# Make predictions and convert probabilities to binary predictions
threshold = 0.5
subset_y_pred = (model.predict(subset_X).flatten() > threshold).astype(int)

# Print sample reviews along with their classification
for i in range(subset_size):
    review = " ".join(index_to_word.get(idx - 3, '?') for idx in subset_X[i] if idx != 0)
    true_label = "Positive" if subset_y_true[i] == 1 else "Negative"
    pred_label = "Positive" if subset_y_pred[i] == 1 else "Negative"
    print("Review:", review)
    print("True Label:", true_label)
    print("Predicted Label:", pred_label)
    print()
