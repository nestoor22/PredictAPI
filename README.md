# PredictAPI
**How to use:**

1. Install requirements:  
`pip install requirements.txt`

2. Install Redis on 6380 port.

3. Run Redis

4. Run web-service in console: 
`python async_service.py`

**In folder model you have 4 models which can predict different properties
        (price, rooms, distance to city center, area)**
 
 ***For test you can use:***
 
 `import requests`
 
 `d = {'user_id': 1, 'rooms': 3, 'distance_to_center': 3.4, 'area': 126, 'floor': 2, 'building_type': 'New building',
     'living_area': 96, 'kitchen_area': 22, 'conditions': 'luxury', 'walls_material': 'brick',
     'balconies': 0, 'ceiling_height': 2.75, 'floors': 5}`
  
 `res = requests.post(' http://localhost:5000/predictPrice/', json=d).text`
 
 **In dictionary `d` is missed value which you want to know (price in example)**
 
 ***You can get possible values for keys (building_type, conditions, walls_material) by using this:***
 
 `import joblib`

`FOLDER_WITH_MODELS = '../apartmentML/models/'`

`# Dict with information how data was scaled. Created in apartmentML`

`TRANSFORMERS_OBJECTS: dict = joblib.load(FOLDER_WITH_MODELS + 'transformers_info')`

`print(TRANSFORMERS_OBJECTS[<key>]['transformer-objects']['LabelTransformer'].classes_)`
