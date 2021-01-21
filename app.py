from flask import Flask, render_template, url_for, request, redirect
from flask_bootstrap import Bootstrap


import os
import inference

app = Flask(__name__)
Bootstrap(app)

"""
Routes
"""
# Testing URL
@app.route('/hello/', methods=['GET', 'POST'])
def hello_world():
    return("Hello World!")

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        uploaded_file = request.files['file']

        # blank filename, button pressed without image
        if uploaded_file.filename != '':
            # should validate file is image
            image_path = os.path.join('static', uploaded_file.filename) 
            # save file
            uploaded_file.save(image_path)
            # perform image preprocessing and get inference
            class_name = inference.get_preds(image_path)
            print('CLASS NAME=', class_name)
            result = {
                'class_name': class_name,
                'image_path': image_path,
            }
            return render_template('show.html', result=result)

    return render_template('index.html')

# complete Flask app
if __name__ == '__main__':
    app.run(debug=True)
