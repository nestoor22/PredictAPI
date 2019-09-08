import os
import ast
import pandas as pd
from aiohttp import web

FOLDER_WITH_MODELS = '../apartmentML/models/'


application_routes = web.RouteTableDef()


@application_routes.get('/predictPrice')
async def predict_price(request):
    data = await request.json()
    data = ast.literal_eval(data)
    data_to_predict = pd.DataFrame([data])
    print(data_to_predict)

    return web.Response(text='Hello')


if __name__ == '__main__':

    app = web.Application()
    app.add_routes(application_routes)
    web.run_app(app, host='localhost', port=3030)

