from flask import Flask, request, render_template
import pickle
import numpy as np
import io
import librosa

app = Flask(__name__)

# Load your trained model
with open('df (1).pkl', 'rb') as file:
    model = pickle.load(file)

@app.route('/', methods=['GET'])
def upload_form():
    return render_template('upload_up.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return render_template('upload_up.html', error='No file part')
    
    file = request.files['file']
    if file.filename == '':
        return render_template('upload_up.html', error='No selected file')
    
    # Process the file and get the transcription
    audio_data = file.read()  # Read file contents
    transcription = process_audio(audio_data)  # Process audio data
    
    return render_template('result_up.html', transcription=transcription)

def process_audio(audio_data):
    # Use librosa to load the audio data
    y, sr = librosa.load(io.BytesIO(audio_data), sr=None)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    
    # Reshape MFCCs to match the input shape expected by the model
    mfccs_reshaped = mfccs.T.reshape(1, -1)
    
    # Use your model to predict text
    prediction = model.predict(mfccs_reshaped)
    
    # Assuming the prediction is in the form of text or can be converted to text
    transcription = "".join(map(str, prediction)) if isinstance(prediction, np.ndarray) else str(prediction)
    
    return transcription

if __name__ == '__main__':
    app.run(debug=True)
