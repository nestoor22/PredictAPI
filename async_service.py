import ast
import joblib
import pandas as pd
from aiohttp import web
from scale_data import scaling_data_to_good_view
from keras.models import model_from_json
from keras.utils import CustomObjectScope
from keras.initializers import glorot_uniform


FOLDER_WITH_MODELS = '../apartmentML/models/'
TRANSFORMERS_OBJECTS: dict = joblib.load(FOLDER_WITH_MODELS + 'transformers_info')

application_routes = web.RouteTableDef()

with CustomObjectScope({'GlorotUniform': glorot_uniform()}):

    price_prediction_model = model_from_json(open('models/price_prediction_model.json', 'r').read())
    price_prediction_model.load_weights('models/price_prediction_weights.h5')

    area_prediction_model = model_from_json(open('models/area_prediction_model.json', 'r').read())
    area_prediction_model.load_weights('models/area_prediction_weights.h5')

    distance_to_center_prediction_model = model_from_json(open('models/distance_to_center_prediction_model.json', 'r').read())
    distance_to_center_prediction_model.load_weights('models/distance_to_center_weights.h5')

    room_prediction_model = model_from_json(open('models/price_prediction_model.json', 'r').read())
    room_prediction_model.load_weights('models/room_prediction_weights.h5')


@application_routes.get('/predictPrice')
async def predict_price(request):
    data = await request.json()
    data = await scaling_data_to_good_view(pd.DataFrame([ast.literal_eval(data)]))
    return web.Response(text=str(price_prediction_model.predict(data)[0][0]))


@application_routes.get('/predictArea')
async def predict_price(request):
    data = await request.json()
    data = await scaling_data_to_good_view(pd.DataFrame([ast.literal_eval(data)]))
    return web.Response(text=str(price_prediction_model.predict(data)[0][0]))


@application_routes.get('/predictRooms')
async def predict_price(request):
    data = await request.json()
    data = await scaling_data_to_good_view(pd.DataFrame([ast.literal_eval(data)]))
    return web.Response(text=str(price_prediction_model.predict(data)[0][0]))


@application_routes.get('/predictDistance')
async def predict_price(request):
    data = await request.json()
    data = await scaling_data_to_good_view(pd.DataFrame([ast.literal_eval(data)]))
    return web.Response(text=str(price_prediction_model.predict(data)[0][0]))


if __name__ == '__main__':

    app = web.Application()
    app.add_routes(application_routes)
    web.run_app(app, host='localhost', port=3030)

