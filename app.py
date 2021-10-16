import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('Model.pkl', 'rb'))


crops=['wheat','mungbean','tea','millet','rice','maize','lentil','jute','cofee','cotton','ground nut','peas','rubber','sugarcane','tobacco','kidney beans','moth beans','coconut','blackgram','adzuki beans','pigeon peas','chick peas','banana','grapes','apple','mango','muskmelon','orange','papaya','watermelon','pomegranate']
crops.sort()

def cropPredictor(p):
    count=0
    for i in range(0,31):
        if(p[0][i]==1):
            count=count+1
            c=crops[i]
            break

    if(count==0):
        return 'rice' 
    return c

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [x for x in request.form.values()]
    print(int_features)
    final_features = [np.array(int_features)]
   
    prediction = model.predict(final_features)
    
    

 

    output=cropPredictor(prediction)

    return render_template('index.html', prediction_text='Suitable crop for growing  $ {}'.format(output))
   



if __name__ == "__main__":
    app.run(debug=True)
