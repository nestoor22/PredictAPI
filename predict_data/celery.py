import pandas as pd
import tensorflow as tf
from celery import Celery
from keras import backend as K
from helpers import load_models, create_connection, write_to_db
from data_processor import return_original_price, return_original_area, return_original_distance,\
    return_original_rooms_number, scaling_data_to_good_view

#  celery -A predict_data.celery worker -l info -P gevent           -- for running celery


celery = Celery('', broker='redis://localhost:6379/0')


@celery.task
def predict_price_from_data(data: dict):
    user_id = data.pop('user_id')
    price_prediction_model = load_models('price')
    data_to_predict = scaling_data_to_good_view(pd.DataFrame([data]))

    graph = tf.get_default_graph()
    with graph.as_default():
        predicted_price = price_prediction_model.predict(data_to_predict)

    K.clear_session()
    data['price'] = return_original_price(predicted_price)
    data['user_id'] = user_id

    connection, cursor = create_connection()
    write_to_db(data, cursor)
    connection.commit()

    return 1


@celery.task
def predict_area_from_data(data: dict):
    user_id = data['user_id']

    area_prediction_model = load_models('area')
    data_to_predict = scaling_data_to_good_view(pd.DataFrame([data]))

    graph = tf.get_default_graph()
    with graph.as_default():
        predicted_area = area_prediction_model.predict(data_to_predict)

    data['area'] = return_original_area(predicted_area)
    data['user_id'] = user_id

    connection, cursor = create_connection()
    write_to_db(data, cursor)
    connection.commit()

    return 1


@celery.task
def predict_distance_to_center_from_data(data: dict):
    user_id = data.pop('user_id')

    distance_to_center_prediction_model = load_models('distance_to_center')
    data_to_predict = scaling_data_to_good_view(pd.DataFrame([data]))

    graph = tf.get_default_graph()
    with graph.as_default():
        predicted_distance_to_center = distance_to_center_prediction_model.predict(data_to_predict)

    data['distance_to_center'] = return_original_distance(predicted_distance_to_center)
    data['user_id'] = user_id

    connection, cursor = create_connection()
    write_to_db(data, cursor)
    connection.commit()

    return 1


@celery.task
def predict_rooms_number_from_data(data: dict):
    user_id = data.pop('user_id')

    rooms_prediction_model = load_models('rooms')
    data_to_predict = scaling_data_to_good_view(pd.DataFrame([data]))

    graph = tf.get_default_graph()
    with graph.as_default():
        predicted_rooms = rooms_prediction_model.predict(data_to_predict)

    data['rooms'] = return_original_rooms_number(predicted_rooms)
    data['user_id'] = user_id

    connection, cursor = create_connection()
    write_to_db(data, cursor)
    connection.commit()

    return 1

