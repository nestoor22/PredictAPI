def create_connection():
    # Create connection to database
    import mysql.connector
    connection = mysql.connector.connect(host='127.0.0.1', port=3306, user='<your_user>', password='<your password>',
                                         use_pure=True)
    cursor = connection.cursor()
    return connection, cursor


def load_models(model_predict=None):

    from keras import backend as K
    from keras.models import model_from_json
    from keras.utils import CustomObjectScope
    from keras.initializers import glorot_uniform

    # Clear sessions
    K.clear_session()

    # Check if parameter has given
    if model_predict is not None:
        with CustomObjectScope({'GlorotUniform': glorot_uniform()}):
            # Load model architecture and weights from files
            model = model_from_json(open('models/{}_prediction_model.json'.format(model_predict), 'r').read())
            model.load_weights('models/{}_prediction_weights.h5'.format(model_predict))
        return model
    else:
        return None


def write_to_db(data: dict):
    pass
