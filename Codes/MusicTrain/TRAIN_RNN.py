#Furkan ÇOLAK


import numpy as np
import pretty_midi
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense, Activation
from tensorflow.keras.utils import to_categorical
import pickle

# Function to extract notes from MIDI files
def get_notes(midi_files):
    notes = []
    for file in midi_files:
        midi = pretty_midi.PrettyMIDI(file)  # Load MIDI file
        for instrument in midi.instruments:
            for note in instrument.notes:
                notes.append(note.pitch)  # Append note pitch to the list
    return notes

# Function to prepare input and output sequences for the network
def prepare_sequences(notes, n_vocab):
    sequence_length = 10  # Length of note sequences
    note_to_int = dict((note, number) for number, note in enumerate(set(notes)))

    network_input = []
    network_output = []

    # Create input sequences and the corresponding outputs
    for i in range(0, len(notes) - sequence_length, 1):
        sequence_in = notes[i:i + sequence_length]
        sequence_out = notes[i + sequence_length]
        network_input.append([note_to_int[char] for char in sequence_in])
        network_output.append(note_to_int[sequence_out])

    n_patterns = len(network_input)

    # Reshape the input into a format compatible with LSTM layers
    network_input = np.reshape(network_input, (n_patterns, sequence_length, 1))
    # Normalize input
    network_input = network_input / float(n_vocab)

    # One hot encode the output vectors
    network_output = to_categorical(network_output)

    return (network_input, network_output, note_to_int)

# Function to create the LSTM model
def create_model(network_input, n_vocab):
    model = Sequential()
    model.add(LSTM(512, input_shape=(network_input.shape[1], network_input.shape[2]), return_sequences=True))
    model.add(Dropout(0.3))  # Avoid overfitting
    model.add(LSTM(512, return_sequences=True))
    model.add(Dropout(0.3))
    model.add(LSTM(512))
    model.add(Dense(256))
    model.add(Dropout(0.3))
    model.add(Dense(n_vocab))  # Output layer
    model.add(Activation('softmax'))  # Softmax for classification
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')  # Compile model
    return model

# Main Process for Training
#midi_files = ['disgust.mid','disgust2.mid','disgust3.mid']  # Replace with your MIDI files for each emotion
midi_files = ['angry.mid','angry2.mid','angry3.mid']
#midi_files = ['happy.mid','happy2.mid','happy3.mid']
#midi_files = ['fear.mid','fear2.mid','fear3.mid']
#midi_files = ['sad.mid','sad2.mid','sad3.mid']
#midi_files = ['suprise.mid','suprise2.mid','suprise3.mid']
#midi_files = ['neutral.mid','neutral2.mid']

notes = get_notes(midi_files)  # Extract notes from MIDI files
n_vocab = len(set(notes))  # Get the number of unique notes

network_input, network_output, note_to_int = prepare_sequences(notes, n_vocab)
model = create_model(network_input, n_vocab)

# Train the model
model.fit(network_input, network_output, epochs=10, batch_size=128)

# Save the trained model
model.save('angry.h5')

# Save the note_to_int dictionary for later use
with open('angry_note_to_int.pkl', 'wb') as f:
    pickle.dump(note_to_int, f)
