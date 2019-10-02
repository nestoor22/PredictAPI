
# List of column which was used for model training. Except 'id'
columns_from_db = ['id', 'cost', 'area', 'rooms', 'floor', 'floors', 'building_type', 'distance_to_center',
                   'living_area', 'kitchen_area', 'conditions', 'walls_material', 'balconies', 'ceiling_height']


def create_connection():
    """
    :return: database connection and cursor
    """
    # Create connection to database
    import mysql.connector
    connection = mysql.connector.connect(host='127.0.0.1', port=3306, user='<your_user>', password='<your_password>',
                                         use_pure=True)
    cursor = connection.cursor(dictionary=True, buffered=True)
    return connection, cursor


def load_models(model_to_predict=None):
    """
    :param model_to_predict: Name of model. Can be : ('price', 'area', 'rooms', 'distance_to_center')
    :return: loaded model
    """
    from keras import backend as K
    from keras.models import model_from_json
    from keras.utils import CustomObjectScope
    from keras.initializers import glorot_uniform

    # Clear sessions
    K.clear_session()

    # Check if parameter has given
    if model_to_predict is not None:
        with CustomObjectScope({'GlorotUniform': glorot_uniform()}):
            # Load model architecture and weights from files
            model = model_from_json(open('models/{}_prediction_model.json'.format(model_to_predict), 'r').read())
            model.load_weights('models/{}_prediction_weights.h5'.format(model_to_predict))
        return model
    else:
        return None


def update_predicted_data(db_cursor, **kwargs):
    """
    :param db_cursor: database cursor
    :param kwargs: named arguments
    :return: None
    """
    sql_query = """UPDATE dream_house.dream_house_datatopredict SET %s=%s 
                   WHERE user_id=%s AND id=%s""" % (kwargs['column'], kwargs['predicted_value'],
                                                    kwargs['user_id'], kwargs['data_to_solve_id'])

    db_cursor.execute(sql_query)


def get_data_from_db(user_id, db_cursor, column_to_predict):
    """
    :param user_id: user id
    :param db_cursor: database cursor
    :param column_to_predict: column which should be predicted
    :return: dict with all columns from list columns_from_db
    """
    columns_to_get = ','.join(columns_from_db)

    sql_query = """SELECT %s FROM dream_house.dream_house_datatopredict 
                   WHERE user_id=%s AND %s IS NULL""" % (columns_to_get, user_id, column_to_predict)

    db_cursor.execute(sql_query)
    return db_cursor.fetchone()


def predict_data_with_model(model, data_to_predict):
    """
    :param model: loaded model
    :param data_to_predict: pandas dataframe to solve
    :return: predicted value in list ( [value] )
    """

    import tensorflow as tf
    graph = tf.get_default_graph()
    with graph.as_default():
        predicted_value = model.predict(data_to_predict)

    return predicted_value[0]