import joblib
import numpy as np
import pandas as pd

FOLDER_WITH_MODELS = '../apartmentML/models/'
TRANSFORMERS_OBJECTS: dict = joblib.load(FOLDER_WITH_MODELS + 'transformers_info')

# Test thing
d = {'Area': 64, 'Rooms': 2, 'Floor': 4, 'BuildingType': 'New building', 'DistanceToCenter': 3,
     'LivingArea': 28.5, 'KitchenArea': 12, 'Condition': 'distinctive', 'WallsMaterial': 'brick',
     'Balconies': 1, 'CeilingHeight': 2.75, 'Floors': 10}


def search_different_types_column(data_frame):
    numeric_columns = []
    string_columns = []
    for column in data_frame:
        if data_frame[column].dtype == np.int64 or data_frame[column].dtype == np.float64:
            numeric_columns.append(column)

        elif data_frame[column].dtype == object:
            string_columns.append(column)

    return numeric_columns, string_columns


def scaling_data_to_good_view(data_frame):

    numeric_columns, string_columns = search_different_types_column(data_frame)

    for column in numeric_columns:
        data_frame[column] = TRANSFORMERS_OBJECTS[column]['transformer-object'].transform(data_frame[column].values.reshape(-1, 1))

    for column in string_columns:
        label_to_num_transformer = TRANSFORMERS_OBJECTS[column]['transformer-objects']['LabelTransformer']
        one_hot_transformer = TRANSFORMERS_OBJECTS[column]['transformer-objects']['OneHotTransformer']

        labels_number = label_to_num_transformer.transform(data_frame[column].values.reshape(-1, 1))
        labels_to_binary = one_hot_transformer.transform(labels_number.reshape(-1, 1)).toarray()

        one_hot_dataset = pd.DataFrame(labels_to_binary, columns=[column+'_'+str(int(i))
                                                                  for i in range(labels_to_binary.shape[1])])

        data_frame = pd.concat([data_frame.drop(columns=column), one_hot_dataset], axis=1)

    print(data_frame)


scaling_data_to_good_view(pd.DataFrame([d]))