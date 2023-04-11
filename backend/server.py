from flask_cors import CORS
from json import *
from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import pickle
import requests
import warnings
import joblib
import os
from dotenv import load_dotenv
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from werkzeug.utils import secure_filename
from data import disease_map, details_map
from flask_cors import CORS, cross_origin

load_dotenv()

API_KEY = os.getenv('API_KEY')
DRIVE_URL = os.getenv('DRIVE_URL')
ACCESS_TOKEN = os.getenv('ACCESS_TOKEN')

warnings.filterwarnings('ignore')

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'
folder_num = 0
folders_list = []

crop_recommendation_model_path = './RandomForest_Crop.pkl'
crop_recommendation_model = pickle.load(
    open(crop_recommendation_model_path, 'rb'))

fertilizer_recommendation_model_path = './RandomForest_Fertilizer.pkl'
fertilizer_recommendation_model = pickle.load(
    open(fertilizer_recommendation_model_path, 'rb'))

# Download Model File
if not os.path.exists('model.h5'):
    print("Downloading model...")
    r = requests.get(DRIVE_URL, stream=True)
    with open('./model.h5', 'wb') as f:
        for chunk in r.iter_content(chunk_size=1024):
            if chunk:
                f.write(chunk)
    print("Finished downloading model.")

# Load model from downloaded model file
model = load_model('model.h5')

# Create folder to save images temporarily
if not os.path.exists('./static/test'):
        os.makedirs('./static/test')

ALLOWED_EXTENSIONS = {'png', 'jpeg', 'jpg'}
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def predict(test_dir):
    test_img = [f for f in os.listdir(os.path.join(test_dir)) if not f.startswith(".")]
    test_df = pd.DataFrame({'Image': test_img})
    
    test_gen = ImageDataGenerator(rescale=1./255)

    test_generator = test_gen.flow_from_dataframe(
        test_df, 
        test_dir, 
        x_col = 'Image',
        y_col = None,
        class_mode = None,
        target_size = (256, 256),
        batch_size = 20,
        shuffle = False
    )
    predict = model.predict(test_generator, steps = np.ceil(test_generator.samples/20))
    test_df['Label'] = np.argmax(predict, axis = -1) # axis = -1 --> To compute the max element index within list of lists
    test_df['Label'] = test_df['Label'].replace(disease_map)

    prediction_dict = {}
    for value in test_df.to_dict('index').values():
        image_name = value['Image']
        image_prediction = value['Label']
        prediction_dict[image_name] = {}
        prediction_dict[image_name]['prediction'] = image_prediction
        prediction_dict[image_name]['description'] = details_map[image_prediction][0]
        prediction_dict[image_name]['symptoms'] = details_map[image_prediction][1]
        prediction_dict[image_name]['source'] = details_map[image_prediction][2]
    return prediction_dict

@app.route("/crop", methods=['POST'])
def members1():
    try:
        N = int(request.json['N'])
        P = int(request.json['P'])
        K = int(request.json['K'])
        ph = float(request.json['Ph'])
        state = request.json['state']
        district = request.json['district']
        start_month = int(request.json['start_month'])
        end_month = int(request.json['end_month'])
    except:
        return jsonify({"crop": 'Enter Valid Details', "data": request.json})

    x = requests.get('https://api.mapbox.com/geocoding/v5/mapbox.places/' + district + ' ' + state +
                     '.json?access_token='+ACCESS_TOKEN)

    coordinates = x.json()['features'][0]['center']

    y = requests.get('https://api.openweathermap.org/data/2.5/weather?lat=' + str(
        coordinates[1]) + '&lon=' + str(coordinates[0]) + '&appid=' + API_KEY)

    humidity = y.json()['main']['humidity']
    temprature = y.json()['main']['temp'] - 273.15

    df = pd.read_csv("./rainfall.csv")
    q = df.loc[(df['STATE_UT_NAME'] == state) & (df['DISTRICT'] == district)]

    total = 0
    l = 12

    if start_month <= end_month:
        l = (end_month-start_month)+1
        for i in range(start_month, end_month+1):
            try:
                total += int(q[i:i+1].value)
            except:
                total -= 1
    elif start_month > end_month:
        l = (end_month+12) - start_month + 1
        for i in range(start_month, 13):
            try:
                total += int(q[i:i+1].value)
            except:
                total -= 1
        for i in range(1, end_month+1):
            try:
                total += int(q[i:i+1].value)
            except:
                total -= 1

    avg_rainfall = total/l
    data = np.array([N, P, K, temprature, humidity,
                    ph, avg_rainfall]).reshape(1, -1)

    scaler = joblib.load("./all_scaler.gz")
    x_scaled = scaler.transform(data)
    probs = crop_recommendation_model.predict_proba(x_scaled)[0]
    top_3 = sorted(range(len(probs)), key=lambda i: probs[i])[-3:]
    print(top_3)

    final_prediction = []

    dict_crop = {'rice': 20,
                 'maize': 11,
                 'chickpea': 3,
                 'kidneybeans': 9,
                 'pigeonpeas': 18,
                 'mothbeans': 13,
                 'mungbean': 14,
                 'blackgram': 2,
                 'lentil': 10,
                 'pomegranate': 19,
                 'banana': 1,
                 'mango': 12,
                 'grapes': 7,
                 'watermelon': 21,
                 'muskmelon': 15,
                 'apple': 0,
                 'orange': 16,
                 'papaya': 17,
                 'coconut': 4,
                 'cotton': 6,
                 'jute': 8,
                 'coffee': 5}

    for top in top_3:
        for key, value in dict_crop.items():
            if value == top:
                final_prediction.append(key)

    return jsonify({"crop": final_prediction, "data": y.json()['main'], 'l': l})


@app.route("/fertilizer", methods=['POST'])
def members2():
    try:
        N = int(request.json['N'])
        P = int(request.json['P'])
        K = int(request.json['K'])
        state = request.json['state']
        district = request.json['district']
        moisture = float(request.json['moisture'])
        soil_type = int(request.json['soil_type'])
        crop_type = int(request.json['crop_type'])
    except:
        return jsonify({"crop": 'Enter Valid Data', "data": request.json})

    x = requests.get('https://api.mapbox.com/geocoding/v5/mapbox.places/' + district + ' ' + state +
                     '.json?access_token=' + ACCESS_TOKEN)

    coordinates = x.json()['features'][0]['center']

    y = requests.get('https://api.openweathermap.org/data/2.5/weather?lat=' + str(
        coordinates[1]) + '&lon=' + str(coordinates[0]) + '&appid=' + API_KEY)

    temperature = y.json()['main']['temp'] - 273.15

    data = np.array(
        [[temperature, moisture, soil_type, crop_type, N, K, P]]).reshape(1, -1)

    my_prediction = fertilizer_recommendation_model.predict(data)[0]
    print(my_prediction)

    dict_fert = {'10-26-26': 0,
                 '14-35-14': 1,
                 '17-17-17': 2,
                 '20-20': 3,
                 '28-28': 4,
                 'DAP': 5,
                 'Urea': 6}

    for key, value in dict_fert.items():
        if value == my_prediction:
            final_pred = key

    return jsonify({"crop": final_pred, "data": dict_fert})

@app.route('/predictdisease', methods=['POST'])
@cross_origin()
def api_predict():
    global folder_num
    global folders_list
    if folder_num >= 1000000:
            folder_num = 0
    # check if the post request has the file part
    print(request.files)
    print("hujbvhkvjhcjg")
    app.logger.info(request.files)
    if 'files' not in request.files:
        return {"Error": "No files part found."}
    # Create a new folder for every new file uploaded,
    # so that concurrency can be maintained
    files = request.files.getlist('files')
    app.config['UPLOAD_FOLDER'] = "./static/test"
    app.config['UPLOAD_FOLDER'] = app.config['UPLOAD_FOLDER'] + '/predict_' + str(folder_num).rjust(6, "0")
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
        folders_list.append(app.config['UPLOAD_FOLDER'])
        folder_num += 1
    for file in files:
        if file.filename == '':
            return {"Error": "No Files are Selected!"}
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        else:
            return {"Error": "Invalid file type! Only PNG, JPEG/JPG files are supported."}
    try:
        if len(os.listdir(app.config['UPLOAD_FOLDER'])) > 0:
            diseases = predict(app.config['UPLOAD_FOLDER'])
            return diseases
    except:
        return {"Error": "Something Went Wrong!"}

if __name__ == "__main__":
    app.run(debug=True)
