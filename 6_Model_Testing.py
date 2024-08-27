import pickle
import requests
import pandas as pd

with open("data_split.pkl", "rb") as pkl_file:
    data_dict = pickle.load(pkl_file)

test_features = data_dict["test_features"]
test_labels = data_dict["test_labels"]

# Set URL for Flask app
url = 'http://localhost:12345/predict'

responses = {}

for index in range(10):

    payload = dict(enumerate(test_features[index]))
    payload = {str(k) : str(v) for k,v in payload.items()}

    payload = {'row': payload}

    # Send POST request with data and get response
    response = requests.post(url, json=payload)
    responses[index] = response.text
    #print(response.text)
    
print(responses)

print(pd.DataFrame({"Original" : test_labels[:10], "Predicted" : responses.values()}))