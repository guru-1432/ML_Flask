import os
import pickle
from flask import Flask
from ML_models import create_mdoel


app = Flask(__name__)

@app.route('/',methods=['GET'])
def home():
    return 'Serve runnning'

@app.route('/create_model',methods=['GET'])
def get():
    result = create_mdoel()
    return result
#@app.route('/',methods=['GET'])
#pass
if __name__ == '__main__':
    app.run(debug= True)

# IF the file already exist do not train the model else create a new file 
#1. Chekc for the file in the app file 
#2. create function which we can import in the app file and execute it
# 3. This shud happen when the server starts so when we recieve the request we are not touching this logic



    