import pandas as pd
import tensorflow as tf
from celery import Celery
from keras import backend as K
from helpers import load_models, create_connection, update_predicted_data, get_data_from_db
from data_processor import return_original_price, return_original_area, return_original_distance,\
    return_original_rooms_number, scaling_data_to_good_view

#  celery -A predict_data.celery worker --loglevel=debug --concurrency=4 -P gevent           -- for running celery

celery = Celery('', broker='redis://localhost:6380/0')


@celery.task
def predict_price_for_user(user_id):
    connection, cursor = create_connection()
    data = get_data_from_db(user_id, cursor, 'cost')
    if not data:
        return
    del data['cost']

    price_prediction_model = load_models('price')

    data_to_solve_id = data.pop('id')
    data_to_predict = scaling_data_to_good_view(pd.DataFrame([data]))

    graph = tf.get_default_graph()
    with graph.as_default():
        predicted_price = price_prediction_model.predict(data_to_predict)

    original_price = return_original_price(predicted_price[0])
    K.clear_session()

    update_predicted_data(cursor, user_id=user_id, column='cost', predicted_value=original_price,
                          data_to_solve_id=data_to_solve_id)

    connection.commit()

    return 1


@celery.task
def predict_area_for_user(user_id):
    connection, cursor = create_connection()
    data = get_data_from_db(user_id, cursor, 'area')
    if not data:
        return

    del data['area']

    area_prediction_model = load_models('area')
    data_to_solve_id = data.pop('id')
    data_to_predict = scaling_data_to_good_view(pd.DataFrame([data]))

    graph = tf.get_default_graph()
    with graph.as_default():
        predicted_area = area_prediction_model.predict(data_to_predict)

    original_area = return_original_area(predicted_area[0])
    K.clear_session()

    update_predicted_data(cursor, user_id=user_id, column='area', predicted_value=original_area,
                          data_to_solve_id=data_to_solve_id)
    connection.commit()

    return 1


# @celery.task
def predict_distance_to_center_for_user(user_id):
    connection, cursor = create_connection()
    data = get_data_from_db(user_id, cursor, 'distance_to_center')

    if not data:
        return
    del data['distance_to_center']

    distance_to_center_prediction_model = load_models('distance_to_center')
    data_to_solve_id = data.pop('id')
    data_to_predict = scaling_data_to_good_view(pd.DataFrame([data]))

    graph = tf.get_default_graph()
    with graph.as_default():
        predicted_distance_to_center = distance_to_center_prediction_model.predict(data_to_predict)

    original_distance = return_original_distance(predicted_distance_to_center[0])
    K.clear_session()

    update_predicted_data(cursor, user_id=user_id, column='distance_to_center', predicted_value=original_distance,
                          data_to_solve_id=data_to_solve_id)

    connection.commit()

    return 1


@celery.task
def predict_rooms_number_for_user(data: dict):
    user_id = data.pop('user_id')

    rooms_prediction_model = load_models('rooms')
    data_to_predict = scaling_data_to_good_view(pd.DataFrame([data]))

    graph = tf.get_default_graph()
    with graph.as_default():
        predicted_rooms = rooms_prediction_model.predict(data_to_predict)

    data['rooms'] = return_original_rooms_number(predicted_rooms)
    data['user_id'] = user_id

    connection, cursor = create_connection()
    # write_to_db(data, cursor)
    connection.commit()

    return 1

