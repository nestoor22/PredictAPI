import os
from celery import Celery
import pandas as pd
import tensorflow as tf
from keras.models import model_from_json
from keras.utils import CustomObjectScope
from keras.initializers import glorot_uniform
from data_processor import return_original_price, return_original_area, return_original_distance,\
    return_original_rooms_number, scaling_data_to_good_view


celery = Celery('', broker='redis://localhost:6379/0')


def create_connection():
    import mysql.connector
    connection = mysql.connector.connect(host='127.0.0.1', port=3306, user='root', password='asdfghjkl228',
                                         use_pure=True)
    cursor = connection.cursor()
    return connection, cursor


def load_models(model_predict=None):
    if model_predict is not None:
        with CustomObjectScope({'GlorotUniform': glorot_uniform()}):
            model = model_from_json(open('models/{}_prediction_model.json'.format(model_predict), 'r').read())
            model.load_weights('models/{}_prediction_weights.h5'.format(model_predict))
        return model
    else:
        return None


@celery.task
def predict_price_from_data(data):
    connection, cursor = create_connection()

    price_prediction_model = load_models('price')
    data_to_predict = scaling_data_to_good_view(pd.DataFrame([data]))

    graph = tf.get_default_graph()
    with graph.as_default():
        predicted_price = price_prediction_model.predict(data_to_predict)

    original_price = return_original_price(predicted_price)
    cursor.execute(f"""INSERT INTO dream_house.prices(price) VALUES ({original_price})""")
    connection.commit()
    connection.close()
    return 1


@celery.task
def predict_area_from_data(data):
    connection, cursor = create_connection()

    area_prediction_model = load_models('area')
    data_to_predict = scaling_data_to_good_view(pd.DataFrame([data]))

    graph = tf.get_default_graph()
    with graph.as_default():
        predicted_area = area_prediction_model.predict(data_to_predict)

    original_area = return_original_area(predicted_area)
    cursor.execute(f"""INSERT INTO dream_house.prices(price) VALUES ({original_area})""")
    connection.commit()
    connection.close()
    return 1


@celery.task
def predict_distance_to_center_from_data(data):
    connection, cursor = create_connection()

    distance_to_center_prediction_model = load_models('distance_to_center')
    data_to_predict = scaling_data_to_good_view(pd.DataFrame([data]))

    graph = tf.get_default_graph()
    with graph.as_default():
        predicted_distance_to_center = distance_to_center_prediction_model.predict(data_to_predict)

    original_distance_to_center = return_original_distance(predicted_distance_to_center)
    cursor.execute(f"""INSERT INTO dream_house.prices(price) VALUES ({original_distance_to_center})""")
    connection.commit()
    connection.close()
    return 1


@celery.task
def predict_rooms_number_from_data(data):
    connection, cursor = create_connection()

    rooms_prediction_model = load_models('rooms')
    data_to_predict = scaling_data_to_good_view(pd.DataFrame([data]))

    graph = tf.get_default_graph()
    with graph.as_default():
        predicted_rooms = rooms_prediction_model.predict(data_to_predict)

    original_rooms_number = return_original_rooms_number(predicted_rooms)
    cursor.execute(f"""INSERT INTO dream_house.prices(price) VALUES ({original_rooms_number})""")
    connection.commit()
    connection.close()
    return 1

