import ast
import pandas as pd
from aiohttp import web
from keras.models import model_from_json
from keras.utils import CustomObjectScope
from keras.initializers import glorot_uniform
from data_processor import scaling_data_to_good_view, rescale_price, rescale_area, rescale_distance, get_value_for_rooms


FOLDER_WITH_MODELS = '../apartmentML/models/'

application_routes = web.RouteTableDef()

with CustomObjectScope({'GlorotUniform': glorot_uniform()}):

    price_prediction_model = model_from_json(open('models/price_prediction_model.json', 'r').read())
    price_prediction_model.load_weights('models/price_prediction_weights.h5')

    area_prediction_model = model_from_json(open('models/area_prediction_model.json', 'r').read())
    area_prediction_model.load_weights('models/area_prediction_weights.h5')

    distance_to_center_prediction_model = model_from_json(open('models/distance_to_center_prediction_model.json', 'r').read())
    distance_to_center_prediction_model.load_weights('models/distance_to_center_weights.h5')

    room_prediction_model = model_from_json(open('models/rooms_prediction_model.json', 'r').read())
    room_prediction_model.load_weights('models/rooms_prediction_weights.h5')


@application_routes.get('/predictPrice')
async def predict_price(request):
    data = await request.json()
    data = await scaling_data_to_good_view(pd.DataFrame([ast.literal_eval(data)]))
    return web.Response(text=str(rescale_price(price_prediction_model.predict(data)[0][0])))


@application_routes.get('/predictArea')
async def predict_area(request):
    data = await request.json()
    data = await scaling_data_to_good_view(pd.DataFrame([ast.literal_eval(data)]))
    return web.Response(text=str(rescale_area(area_prediction_model.predict(data)[0][0])))


@application_routes.get('/predictRooms')
async def predict_rooms(request):
    data = await request.json()
    data = await scaling_data_to_good_view(pd.DataFrame([ast.literal_eval(data)]))
    result = get_value_for_rooms(room_prediction_model.predict(data))
    return web.Response(text=str(result))


@application_routes.get('/predictDistance')
async def predict_distance(request):
    data = await request.json()
    data = await scaling_data_to_good_view(pd.DataFrame([ast.literal_eval(data)]))
    return web.Response(text=str(rescale_distance(distance_to_center_prediction_model.predict(data)[0][0])))


app = web.Application()
app.add_routes(application_routes)
web.run_app(app, host='localhost', port=3030)

