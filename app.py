# Importing essential libraries
from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
from flask_cors import CORS
import os
import pandas as pd  

# Membuka file pickle dan memuat data
with open('data_file.pkl', 'rb') as f:
    loaded_data = pickle.load(f)

# Membuka file pickle untuk akurasi
with open('akurasiakhir.pkl', 'rb') as f1:
    accuracy = pickle.load(f1)

# Membuka file pickle untuk classification report
with open('cfreport.pkl', 'rb') as f2:
    classification_report_str = pickle.load(f2)

# Fungsi untuk membaca pickle heart_featureselection.pkl
with open('heart_featureselection.pkl', 'rb') as f3:
    heart_featureselection = pickle.load(f3)


    
# Extract path gambar dan dataframe fitur dari data yang dimuat
confusion_matrix_path = loaded_data['confusion_matrix']
selected_features = loaded_data['selected_features']

# Inisialisasi classification_report dengan string kosong
classification_report = loaded_data.get('classification_report', '')

# Load the Random Forest Classifier model
filename = 'heart-disease-prediction-RFC-SVC-RFE.pkl'
model = pickle.load(open(filename, 'rb'))

# Inisialisasi objek Flask
app = Flask(__name__, template_folder='templates', static_folder='static')

# Mengaktifkan CORS
CORS(app)

@app.route('/')
def home():
    return render_template('index.html', css_file='css/app.css')

@app.route('/prediksi')
def prediksi():

    # Menyertakan data dari file penjelasan.csv
    csv_penjelasan_path = os.path.join('penjelasan.csv')
    # Membaca dataset heart.csv
    df_penjelasan = pd.read_csv(csv_penjelasan_path)

    # Mengganti nilai NaN dengan string kosong
    df_penjelasan = df_penjelasan.fillna('')

    return render_template('pages-prediksi.html', df_penjelasan=df_penjelasan, css_file='css/app.css')

@app.route('/confusion')
def confusion():
    return render_template('pages-conmat.html', css_file='css/app.css')



@app.route('/input')
def input():
    # Menentukan path lengkap file heart.csv
    csv_heart_path = os.path.join('heart.csv')
    # Menentukan path lengkap file pendahulu.csv
    csv_pendahulu_path = os.path.join('pendahulu.csv')

    # Membaca dataset heart.csv
    df_heart = pd.read_csv(csv_heart_path)

    # Membaca dataset pendahulu.csv
    df_pendahulu = pd.read_csv(csv_pendahulu_path, encoding='latin1')  # Menggunakan encoding latin1

    # Membuka file pickle untuk akurasi
    with open('akurasiakhir.pkl', 'rb') as f1:
        accuracy = pickle.load(f1)

    # Membuka file pickle untuk classification report
    with open('cfreport.pkl', 'rb') as f2:
        classification_report_str = pickle.load(f2)

    # Memuat data dari pickle heart_featureselection.pkl
    with open('heart_featureselection.pkl', 'rb') as f3:
        heart_featureselection = pickle.load(f3)

    return render_template('pages-input.html', akurasi=accuracy, classification_report=classification_report_str, df_heart=df_heart, df_pendahulu=df_pendahulu, heart_featureselection=heart_featureselection, css_file='css/app.css')


@app.route('/about')
def about():
    return render_template('pages-about.html', css_file='css/app.css')



@app.route('/gambarcm', methods=['POST'])
def gambarcm():
    if request.method == 'POST':
        try:
            # Menggunakan jsonify untuk mengirimkan path gambar dan data tabel ke HTML
            return jsonify({
                'confusion_matrix': confusion_matrix_path,
                'selected_features': selected_features.to_dict(orient='records')
            })

        except Exception as e:
            return {"error": str(e)}


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            # Extracting form values
            sex = int(request.form.get('sex'))
            cp = int(request.form.get('cp'))
            trestbps = float(request.form['trestbps'])
            thalach = float(request.form.get('thalach'))  # Tambahkan ini
            exang = int(request.form.get('exang'))
            oldpeak = float(request.form['oldpeak'])
            slope = int(request.form.get('slope'))
            ca = float(request.form['ca'])
            thal = int(request.form.get('thal'))

            # Creating a NumPy array for the input data
            data = np.array([[sex, cp, trestbps, thalach, exang, oldpeak, slope, ca, thal]])

            predicted_class = model.predict(data)
            probabilities = model.predict_proba(data)
            predicted_prob = probabilities[0][predicted_class[0]]

            # Return prediction as JSON
            return jsonify({'Prediksinya': predicted_class.tolist(), 'Akurasi': predicted_prob})

        except Exception as e:
            return {"error": str(e)}

if __name__ == '__main__':
    app.run(debug=True)

