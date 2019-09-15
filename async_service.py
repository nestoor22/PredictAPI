import ast
from flask import Flask, request
from celery import Celery
from keras.models import model_from_json
from keras.utils import CustomObjectScope
from keras.initializers import glorot_uniform

app = Flask(__name__)
app.config['CELERY_BROKER_URL'] = 'redis://localhost:6380/0'
app.config['CELERY_RESULT_BACKEND'] = 'redis://localhost:6379/0'

celery = Celery(app.name, broker=app.config['CELERY_BROKER_URL'])
celery.conf.update(app.config)

app = Flask(__name__)
app.config['CELERY_BROKER_URL'] = 'redis://localhost:6379/0'
app.config['CELERY_RESULT_BACKEND'] = 'redis://localhost:6379/0'

with CustomObjectScope({'GlorotUniform': glorot_uniform()}):

    price_prediction_model = model_from_json(open('models/price_prediction_model.json', 'r').read())
    price_prediction_model.load_weights('models/price_prediction_weights.h5')

    area_prediction_model = model_from_json(open('models/area_prediction_model.json', 'r').read())
    area_prediction_model.load_weights('models/area_prediction_weights.h5')

    distance_to_center_prediction_model = model_from_json(open('models/distance_to_center_prediction_model.json', 'r').read())
    distance_to_center_prediction_model.load_weights('models/distance_to_center_weights.h5')

    room_prediction_model = model_from_json(open('models/rooms_prediction_model.json', 'r').read())
    room_prediction_model.load_weights('models/rooms_prediction_weights.h5')


@celery.task
def predict_price_from_data(data):
    data_to_predict = ast.literal_eval(data)


@app.route('/predictPrice/', methods=['GET'])
def predict_price():
    data = request.json
    return 'Sebek'


if __name__ == '__main__':
    app.run(host='localhost', debug=True)