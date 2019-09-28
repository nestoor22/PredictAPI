def create_connection():
    # Create connection to database
    import mysql.connector
    connection = mysql.connector.connect(host='127.0.0.1', port=3306, user='root', password='asdfghjkl228',
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


def write_to_db(data: dict, db_cursor):
    place_holders = ", ".join("'"+str(value)+"'" for value in data.values())
    columns_query = ','.join(list(data.keys()))
    sql_query = """INSERT INTO dream_house.dream_house_datatopredict (%s) VALUES (%s)""" % (columns_query, place_holders)
    db_cursor.execute(sql_query)


if __name__ == '__main__':
    d = {'user_id': 1, 'rooms': 3, 'distance_to_center': 3.4, 'area': 126, 'floor': 2, 'building_type': 'New building',
         'living_area': 96, 'kitchen_area': 22, 'conditions': 'luxury', 'walls_material': 'brick',
         'balconies': 0, 'ceiling_height': 2.75, 'floors': 5}
    con, cur = create_connection()
    write_to_db(d, cur)
