import pandas as pd
import numpy as np
import json
import os


DATA_BASEPATH = os.path.join(os.path.dirname(__file__), '../datasets/')

def load_data() -> list:
    '''
    return: list
    This function loads the data from the data folder.
    '''
    with open(os.path.join(DATA_BASEPATH, 'questions_answer.json'), 'rt') as file_in:
        data: list = json.load(file_in)
    return data
