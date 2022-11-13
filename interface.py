import config
import project_app.utils
from flask import Flask,render_template,request,jsonify

app = Flask(__name__)

@app.route("/")
def Home():
    print("Welcome to iris data")
    return "Welcome to iris flower prediction"

@app.route("/prediction")
def predict_flower():
    SepalLengthCm = 5.1
    SepalWidthCm = 3.5
    PetalLengthCm = 1.4
    PetalWidthCm = 0.2

    result = project_app.utils.IrisFlowerPrediction(SepalLengthCm,SepalWidthCm,
                      PetalLengthCm,PetalWidthCm)
    predicted_flower = result.Flower_prediction()
    return jsonify({'result':f"Predicted flower is {predicted_flower}"})

if __name__=="__main__":
    app.run()