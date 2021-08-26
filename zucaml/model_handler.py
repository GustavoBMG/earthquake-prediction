import datetime
import os
from pickle import dump
import json

def save_model(model, label, results, train, notes):

    model_path = './models/' + datetime.datetime.now().strftime("%Y%m%dT%H%M%S") + '_' + label

    # https://thispointer.com/how-to-create-a-directory-in-python/
    try:
        os.makedirs(model_path)
    except FileExistsError:
        pass
    
    dump(model.model, open(model_path + '/model.pkl', 'wb')) 
    #TODO: save this model object
    #TODO: save training snapshot, one record
    
    with open(model_path + '/results.txt', 'w') as outfile:
        json.dump(results, outfile)
        
    with open(model_path + '/notes.txt', 'w') as outfile:
        json.dump(notes, outfile)

    return model_path