import pandas as pd
from celery import Celery
from secret import REDIS_URL
from keras import backend as K
from data_processor import return_original_price, return_original_area, return_original_distance,\
    return_original_rooms_number, scaling_data_to_good_view
from helpers import load_models, create_connection, update_predicted_data, get_data_from_db, predict_data_with_model

#  celery -A predict_data.celery worker --loglevel=debug --concurrency=4          -- for running celery

# Create Celery object with Redis as broker
celery = Celery('', broker=REDIS_URL)

cached_models = {}


@celery.task
def predict_price_for_user(user_id):
    """
    :param user_id: user_id which will be used to get data from database
    :return: None
    """
    features = ['area', 'rooms', 'floor', 'balconies', 'distance_to_center', 'city']

    connection, cursor = create_connection()
    data = get_data_from_db(user_id, cursor, 'cost')

    # Check if new data to predict exist
    if not data:
        return

    for feature in features:
        if feature not in data or data.get(feature, None) is None:
            return

    del data['cost']                                                                    # Remove column which predict

    price_prediction_model = get_model_from_cache('price')

    data_to_solve_id = data.pop('id')
    data_to_predict = scaling_data_to_good_view(pd.DataFrame([data])[features])                   # Scale data

    # Rescale predicted value
    original_price = return_original_price(predict_data_with_model(price_prediction_model, data_to_predict))
    K.clear_session()

    update_predicted_data(cursor, user_id=user_id, column='cost', predicted_value=original_price,
                          data_to_solve_id=data_to_solve_id)

    connection.commit()


@celery.task
def predict_area_for_user(user_id):
    """
    :param user_id: user_id which will be used to get data from database
    :return: None
    """
    features = ['cost', 'rooms', 'floor', 'balconies', 'distance_to_center', 'city']

    connection, cursor = create_connection()
    data = get_data_from_db(user_id, cursor, 'area')

    # Check if new data to predict exist
    if not data:
        return

    for feature in features:
        if feature not in data or data.get(feature, None) is None:
            return

    del data['area']                                                            # Remove column which predict

    area_prediction_model = get_model_from_cache('area')                                 # Load model
    data_to_solve_id = data.pop('id')
    data_to_predict = scaling_data_to_good_view(pd.DataFrame([data])[features])           # Scale data

    # Rescale predicted value
    original_area = return_original_area(predict_data_with_model(area_prediction_model, data_to_predict))
    K.clear_session()

    # Update db with predicted value and commit changes
    update_predicted_data(cursor, user_id=user_id, column='area', predicted_value=original_area,
                          data_to_solve_id=data_to_solve_id)
    connection.commit()


@celery.task
def predict_distance_to_center_for_user(user_id):
    """
    :param user_id: user_id which will be used to get data from database
    :return: None
    """
    features = ['cost', 'area', 'rooms', 'floor', 'balconies', 'city']

    connection, cursor = create_connection()
    data = get_data_from_db(user_id, cursor, 'distance_to_center')

    # Check if new data to predict exist
    if not data:
        return

    for feature in features:
        if feature not in data or data.get(feature, None) is None:
            return

    del data['distance_to_center']                                                  # Remove column which predict

    distance_to_center_prediction_model = get_model_from_cache('distance_to_center')         # load model
    data_to_solve_id = data.pop('id')
    data_to_predict = scaling_data_to_good_view(pd.DataFrame([data])[features])               # Scale data

    # Rescale predicted value
    original_distance = return_original_distance(predict_data_with_model(distance_to_center_prediction_model,
                                                                         data_to_predict))
    K.clear_session()

    # Update db with predicted value and commit changes
    update_predicted_data(cursor, user_id=user_id, column='distance_to_center', predicted_value=original_distance,
                          data_to_solve_id=data_to_solve_id)

    connection.commit()


@celery.task
def predict_rooms_number_for_user(user_id):
    """
    :param user_id: user_id which will be used to get data from database
    :return: None
    """
    features = ['cost', 'area', 'conditions', 'distance_to_center', 'city']
    connection, cursor = create_connection()
    data = get_data_from_db(user_id, cursor, 'rooms')

    # Check if new data to predict exist
    if not data:
        return

    for feature in features:
        if feature not in data or data.get(feature, None) is None:
            return

    del data['rooms']                                                   # Remove column which predict

    data_to_solve_id = data.pop('id')

    data_to_predict = scaling_data_to_good_view(pd.DataFrame([data])[features])   # Scale data
    rooms_prediction_model = get_model_from_cache('rooms')                       # load ml model

    # Rescale predicted value
    original_rooms = return_original_rooms_number(predict_data_with_model(rooms_prediction_model, data_to_predict))
    K.clear_session()

    # Update db with predicted value and commit changes
    update_predicted_data(cursor, user_id=user_id, column='rooms', predicted_value=original_rooms,
                          data_to_solve_id=data_to_solve_id)

    connection.commit()


def get_model_from_cache(model_name):
    if model_name not in cached_models:
        price_prediction_model = load_models(model_name)  # Load model
        cached_models[model_name] = price_prediction_model
    else:
        price_prediction_model = cached_models[model_name]

    return price_prediction_model
