import pandas as pd
import numpy as np
import json

def clean_text(data):
    for i in range(len(data)):
        data[i]['answer'] = data[i]['answer'].replace('Answer\n', '')
    
    return data

