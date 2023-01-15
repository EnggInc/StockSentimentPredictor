import numpy as np
from flask import Flask, request , jsonify , render_template
import pickle

#Create Flask append
app = Flask(__name__)

#load the pickel model
clf = pickle.load(open("Stock_Prediction.pkl","rb"))
load_vect = pickle.load(open("Vetorizer.pkl","rb"))

@app.route("/")
def Home():
    return render_template("index.html")
    
@app.route("/predict", methods = ["POST"])
def predict():
    ##features = request.form.values()
    
    ##cv = TfidfVectorizer()
    
    features=request.form['title']
    #feature = [[np.array(features)]]
    prediction = clf.predict(load_vect.transform([features]))
    #sdf = vec.transform([prediction]).reshape(1, -1)
    return render_template("index.html", prediction_text = "The Sentiment is {}".format(prediction))
    
    
if __name__ == "__main__":
   app.run(debug=True) 