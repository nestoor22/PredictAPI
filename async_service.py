from flask import Flask, request
from predict_data.celery import predict_price_from_data, predict_rooms_number_from_data,\
    predict_area_from_data, predict_distance_to_center_from_data

app = Flask(__name__)                                                       # Create app
app.config['CELERY_BROKER_URL'] = 'redis://localhost:6380/0'                # Add celery settings to application

# Create flask routes. All functions get data as json and run celery task
@app.route('/predictPrice/', methods=['POST'])
def predict_price():
    data = request.json
    predict_price_from_data.delay(data)
    return 'OK'


@app.route('/predictArea/', methods=['POST'])
def predict_area():
    data = request.json
    predict_area_from_data.delay(data)
    return 'OK'


@app.route('/predictDistance/', methods=['POST'])
def predict_distance_to_center():
    data = request.json
    predict_distance_to_center_from_data.delay(data)
    return 'OK'


@app.route('/predictRooms/', methods=['POST'])
def predict_rooms():
    data = request.json
    predict_rooms_number_from_data.delay(data)
    return 'OK'


if __name__ == '__main__':
    app.run(host='localhost', debug=True)