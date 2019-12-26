from secret import REDIS_URL
from flask import Flask, request
from predict_data.celery import predict_price_for_user, predict_rooms_number_for_user,\
    predict_area_for_user, predict_distance_to_center_for_user

app = Flask(__name__)                                                       # Create app

# Add celery settings to application
app.config['CELERY_BROKER_URL'] = REDIS_URL


# Create flask routes. All functions get data with user_id as key and run celery task. If everything fine - return OK
@app.route('/predictPrice/', methods=['POST'])
def predict_price():
    user_id = request.form.get('user_id')
    predict_price_for_user.delay(user_id)
    return 'OK'


@app.route('/predictArea/', methods=['POST'])
def predict_area():
    user_id = request.form.get('user_id')
    predict_area_for_user.delay(user_id)
    return 'OK'


@app.route('/predictDistance/', methods=['POST'])
def predict_distance_to_center():
    user_id = request.form.get('user_id')
    predict_distance_to_center_for_user.delay(user_id)
    return 'OK'


@app.route('/predictRooms/', methods=['POST'])
def predict_rooms():
    user_id = request.form.get('user_id')
    predict_rooms_number_for_user.delay(user_id)
    return 'OK'


if __name__ == '__main__':
    app.run(host='localhost', debug=True)
