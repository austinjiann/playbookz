from flask import Flask, request, render_template, send_from_directory, jsonify
import os
import subprocess
import uuid

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
PROCESSED_FOLDER = 'processed'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER

# Create directories if they don't exist
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
if not os.path.exists(PROCESSED_FOLDER):
    os.makedirs(PROCESSED_FOLDER)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400
    file = request.files['video']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file:
        # Save the uploaded file
        _, ext = os.path.splitext(file.filename)
        video_filename = str(uuid.uuid4()) + ext
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], video_filename)
        file.save(video_path)

        # Process the video
        output_filename = f"highlights_{video_filename}"
        output_path = os.path.join(app.config['PROCESSED_FOLDER'], output_filename)

        try:
            command = [
                'python',
                'gen_highlights.py',
                '--video',
                video_path,
                '--mode',
                'audio',
                '--output',
                output_path
            ]
            subprocess.run(command, check=True, capture_output=True, text=True)
            return jsonify({'downloadUrl': f'/download/{output_filename}'})
        except subprocess.CalledProcessError as e:
            return jsonify({'error': 'Error processing video', 'details': e.stderr}), 500

@app.route('/download/<filename>')
def download_file(filename):
    return send_from_directory(app.config['PROCESSED_FOLDER'], filename, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True, port=5001)
