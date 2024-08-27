from flask import Flask, request
import torch
import numpy as np
app = Flask(__name__)

@app.route("/predict", methods = ["POST"])
def predict():
    
    model = torch.jit.load('model_3_layer_10_epoch.pt')
    model.eval()
    
    temp_feats = request.json["row"]
    feats = np.array([float(x) for x in temp_feats.values()])
    feats = torch.tensor(feats, dtype = torch.float32).reshape(1, -1)
    preds = model(feats)
    return "1" if preds.item() > 0.5 else "0"

if __name__ == "__main__":
    app.run(debug = True, host = "localhost", port = 12345)