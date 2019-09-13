import joblib
import numpy as np
import pandas as pd

# Constants
FOLDER_WITH_MODELS = '../apartmentML/models/'

# Dict with information how data was scaled. Created in apartmentML
TRANSFORMERS_OBJECTS: dict = joblib.load(FOLDER_WITH_MODELS + 'transformers_info')


def search_different_types_column(data_frame):
    """
    :param data_frame: pandas dataframe with all data
    :return: two list ( with column names which contain numeric data and with column names which contain string data )
    """

    numeric_columns = []
    string_columns = []

    # Detect data types
    for column in data_frame:
        if data_frame[column].dtype == np.int64 or data_frame[column].dtype == np.float64:
            numeric_columns.append(column)

        elif data_frame[column].dtype == object:
            string_columns.append(column)

    return numeric_columns, string_columns


async def scaling_data_to_good_view(data_frame):

    numeric_columns, string_columns = search_different_types_column(data_frame)

    # Scale numeric columns
    for column in numeric_columns:
        data_frame[column] = TRANSFORMERS_OBJECTS[column]['transformer-object'].transform(data_frame[column].
                                                                                          values.reshape(-1, 1))

    # Scale string columns
    for column in string_columns:
        label_to_num_transformer = TRANSFORMERS_OBJECTS[column]['transformer-objects']['LabelTransformer']
        one_hot_transformer = TRANSFORMERS_OBJECTS[column]['transformer-objects']['OneHotTransformer']

        labels_number = label_to_num_transformer.transform(data_frame[column].values.reshape(-1, 1))
        labels_to_binary = one_hot_transformer.transform(labels_number.reshape(-1, 1)).toarray()

        one_hot_dataset = pd.DataFrame(labels_to_binary, columns=[column+'_'+str(int(i))
                                                                  for i in range(labels_to_binary.shape[1])])

        data_frame = pd.concat([data_frame.drop(columns=column), one_hot_dataset], axis=1)

    # Return dataframe with scaled data
    return data_frame


# Return original price
def rescale_price(value):
    original_value = TRANSFORMERS_OBJECTS['Cost']['transformer-object'].inverse_transform(value.reshape(1, -1))
    return original_value[0][0]


# Return original area
def rescale_area(value):
    original_value = TRANSFORMERS_OBJECTS['Area']['transformer-object'].inverse_transform(value.reshape(1, -1))
    return original_value[0][0]


# Return original distance
def rescale_distance(value):
    original_value = TRANSFORMERS_OBJECTS['DistanceToCenter']['transformer-object'].inverse_transform(value.reshape(1, -1))
    return original_value[0][0]


# Return original count of rooms. Add 1 because in one hot enc class started from 0 but min class in pur case is 1
def get_value_for_rooms(values: np.array):
    value = np.argmax(values) + 1
    return value