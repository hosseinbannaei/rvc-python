from flask import Flask, request, jsonify, send_file
from pydub import AudioSegment
import os
import io
from rvc_python.infer import infer_file, load_rvc_model

app = Flask(__name__)

hu_model = None

tts_file = "outputs/tts/tts.wav"
rvc_file = "outputs/rvc/rvc.wav"

def change_voice(input_file, rvc_model, rvc_pitch):
    result = infer_file(
        input_path=input_file,
        model_path=f"models/rvc/{rvc_model}.pth",
        index_path="",  # Optional: specify path to index file if available
        device="cuda:0", # Use cpu or cuda
        f0method="rmvpe",  # Choose between 'harvest', 'crepe', 'rmvpe', 'pm'
        f0up_key=rvc_pitch,  # Transpose setting
        opt_path=rvc_file, # Output file path
        index_rate=0.5,
        filter_radius=3,
        resample_sr=0,  # Set to desired sample rate or 0 for no resampling.
        rms_mix_rate=0.25,
        protect=0.33,
        version="v2"
    )
    return rvc_file


@app.route('/voice-change', methods=['POST'])
def voice_change():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Extracting form data
    rvc_model = request.form.get('model')
    rvc_pitch = request.form.get('pitch')

    if rvc_model is None or rvc_pitch is None:
        return jsonify({'error': 'Missing settings in JSON'}), 400
    
    audio_data = file.read()
    audio_segment = AudioSegment.from_raw(io.BytesIO(audio_data), sample_width=2, frame_rate=22050, channels=1)
    wav_io = io.BytesIO()
    audio_segment.export(wav_io, format="wav")
    wav_io.seek(0)

    with open(tts_file, "wb") as f:
        f.write(wav_io.read())
    
    change_voice(tts_file, rvc_model, rvc_pitch)

    try:
        # Ensure the file exists
        if not os.path.isfile(rvc_file):
            return jsonify({'error': 'File not found'}), 404

        # Return the file
        return send_file(rvc_file, as_attachment=True, mimetype='audio/wav')

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    hu_model = load_rvc_model()
    app.run(debug=True, port=5000, host='0.0.0.0')