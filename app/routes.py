from flask import render_template, flash, redirect, request, jsonify
from app import app
from models import Classifier
import json

classifier = Classifier()

@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html', title='Index')

@app.route('/emotion/', methods=['GET', 'POST'])
def emotion():
    if request.method == 'POST':
        data = json.loads(request.data)
        value = classifier.classify_message(data['value'])
        return jsonify(value)