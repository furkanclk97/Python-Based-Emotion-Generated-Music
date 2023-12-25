#Furkan ÇOLAK


import os
import cv2
import numpy as np
from keras.preprocessing import image
import warnings
warnings.filterwarnings("ignore")
from keras.preprocessing.image import load_img, img_to_array 
from keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np
import pretty_midi
from tensorflow.keras.models import load_model
import pickle

# Load the pre-trained facial emotion recognition model
model = load_model("facial_emotion_model.h5")

# Initialize the Haar Cascade for face detection
face_haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Start webcam capture
cap = cv2.VideoCapture(0)

# Function to generate music notes using a trained model
def generate_notes(model, note_to_int, n_vocab, length=500, diversity=1.0):
    sequence_length = 10
    int_to_note = {number: note for note, number in note_to_int.items()}

    # Generate a random start sequence
    start_sequence = np.random.randint(0, n_vocab, size=(sequence_length,))
    pattern = list(start_sequence)

    prediction_output = []

    # Generate notes using the model
    for note_index in range(length):
        # Prepare the input sequence
        prediction_input = np.reshape(pattern, (1, len(pattern), 1))
        prediction_input = prediction_input / float(n_vocab)

        # Predict the next note
        prediction = model.predict(prediction_input, verbose=0)

        # Apply diversity to the prediction
        prediction = np.log(prediction + 1e-7) / diversity
        exp_preds = np.exp(prediction)
        prediction = exp_preds / np.sum(exp_preds)

        # Choose the most probable next note
        prediction_probs = prediction[0] / np.sum(prediction[0])
        probas = np.random.multinomial(1, prediction_probs, 1)
        index = np.argmax(probas)

        result = int_to_note[index]
        prediction_output.append(result)

        # Update the pattern
        pattern.append(index)
        pattern = pattern[1:len(pattern)]

    return prediction_output

# Function to convert the generated notes into a MIDI file
def create_midi(prediction_output, output_file='generated_melody.mid'):
    offset = 0
    output_notes = []

    # Create PrettyMIDI notes and add them to the MIDI pattern
    for pattern in prediction_output:
        note = pretty_midi.Note(
            velocity=100, pitch=pattern, start=offset, end=offset + 0.5)
        output_notes.append(note)
        offset += 0.5

    midi_pattern = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=0)
    instrument.notes.extend(output_notes)
    midi_pattern.instruments.append(instrument)
    midi_pattern.write(output_file)

# Main loop for processing webcam frames
while True:
    ret, test_img = cap.read()  # Capture frame from webcam
    if not ret:
        continue
    gray_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces_detected = face_haar_cascade.detectMultiScale(gray_img, 1.32, 5)

    for (x, y, w, h) in faces_detected:
        # Draw rectangle around detected faces
        cv2.rectangle(test_img, (x, y), (x + w, y + h), (255, 0, 0), thickness=7)
        roi_gray = gray_img[y:y + w, x:x + h]  # Crop the face area
        roi_gray = cv2.resize(roi_gray, (48, 48))
        img_pixels = image.img_to_array(roi_gray)
        img_pixels = np.expand_dims(img_pixels, axis=0)
        img_pixels /= 255

        # Predict the emotion of the detected face
        predictions = model.predict(img_pixels)
        max_index = np.argmax(predictions[0])
        emotions = ('angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'suprise')
        predicted_emotion = emotions[max_index]

        # Display the predicted emotion on the webcam feed
        cv2.putText(test_img, predicted_emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    resized_img = cv2.resize(test_img, (1000, 700))
    cv2.imshow('Facial emotion analysis ', resized_img)

    # Generate and save a MIDI file when 'g' key is pressed
    if cv2.waitKey(10) == ord('g'):
        model = load_model(predicted_emotion+'.h5')
        with open(predicted_emotion+'_note_to_int.pkl', 'rb') as f:
            note_to_int = pickle.load(f)

        n_vocab = len(note_to_int)
        prediction_output = generate_notes(model, note_to_int, n_vocab)
        create_midi(prediction_output, predicted_emotion+'.mid')

    # Break the loop when 'q' key is pressed
    if cv2.waitKey(10) == ord('q'):
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
