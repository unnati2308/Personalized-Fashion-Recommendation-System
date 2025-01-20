from flask import Flask, request, jsonify, render_template, redirect, url_for, send_file
from flask_cors import CORS
import os
import similarity

app = Flask(__name__)
CORS(app)

os.makedirs('uploads', exist_ok=True)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'GET':
        return render_template('login.html')
    elif request.method == 'POST':
        phone = request.form.get('phone')
        name = request.form.get('name')
        age = request.form.get('age')
        return redirect(url_for('chatbot'))

@app.route('/chatbot', methods=['GET', 'POST'])
def chatbot():
    if request.method == 'GET':
        return render_template('chatbot.html')
    elif request.method == 'POST':
        return redirect(url_for('upload'))

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    error_message = None

    if request.method == 'POST':
        if 'image' not in request.files or request.files['image'].filename == '':
            error_message = "No file selected. Please choose an image to upload."
        else:
            file = request.files['image']
            upload_path = os.path.join("uploads", file.filename)
            file.save(upload_path)

            similar_images = similarity.find_similar_images(upload_path)

            return render_template('results.html', similar_images=similar_images)

    return render_template('upload.html', error_message=error_message)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
