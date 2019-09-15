import os
from celery import Celery
import pandas as pd
import tensorflow as tf
from keras.models import model_from_json
from keras.utils import CustomObjectScope
from keras.initializers import glorot_uniform
from data_processor import return_original_price, scaling_data_to_good_view


celery = Celery('', broker='redis://localhost:6379/0')


@celery.task
def predict_price_from_data(data):
    import mysql.connector
    connection = mysql.connector.connect(host='127.0.0.1', port=3306, user='root', password='asdfghjkl228',
                                         use_pure=True)
    cursor = connection.cursor()

    with CustomObjectScope({'GlorotUniform': glorot_uniform()}):
        price_prediction_model = model_from_json(open('models/price_prediction_model.json', 'r').read())
        price_prediction_model.load_weights('models/price_prediction_weights.h5')

    data_to_predict = scaling_data_to_good_view(pd.DataFrame([data]))

    graph = tf.get_default_graph()
    with graph.as_default():
        predicted_value = price_prediction_model.predict(data_to_predict)

    original_value = return_original_price(predicted_value)
    cursor.execute(f"""INSERT INTO dream_house.prices(price) VALUES ({original_value})""")
    connection.commit()
    connection.close()
    return 1
