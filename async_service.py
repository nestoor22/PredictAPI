import pandas as pd
from flask import Flask, request
from predict_data.celery import *

app = Flask(__name__)
app.config['CELERY_BROKER_URL'] = 'redis://localhost:6379/0'


@app.route('/predictPrice/', methods=['GET'])
def predict_price():
    data = request.json
    predict_price_from_data.delay(data)
    return 'OK'


@app.route('/predictArea/', methods=['GET'])
def predict_area():
    data = request.json
    predict_area_from_data.delay(data)
    return 'OK'


@app.route('/predictDistance/', methods=['GET'])
def predict_distance_to_center():
    data = request.json
    predict_distance_to_center_from_data.delay(data)
    return 'OK'


@app.route('/predictRooms/', methods=['GET'])
def predict_rooms():
    data = request.json
    predict_rooms_number_from_data.delay(data)
    return 'OK'


if __name__ == '__main__':
    app.run(host='localhost', debug=True)