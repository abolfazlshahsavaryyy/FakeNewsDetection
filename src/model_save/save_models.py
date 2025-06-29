import joblib

def save_model(models:list):

    for model in models:
        joblib.dump(model,f'/home/abolfazl/Documents/python-code/FakeNewsDetection/data/model/{str(model)}.pkl')
        print(f"model {str(model)} has been saved")