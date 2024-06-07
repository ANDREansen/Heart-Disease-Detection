# importing required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.svm import SVC
from sklearn.feature_selection import RFE
from sklearn.model_selection import GridSearchCV
import pickle


# loading and reading the dataset
heart = pd.read_csv("heart.csv")
df = heart.copy()

# Pre-processing
X = df.drop(['num'], axis=1)
Y = df['num']

# Inisialisasi StandardScaler
scaler = StandardScaler()
# Melakukan standardization (fit dan transform) pada data X dan menyimpan hasilnya dalam X_scaled
X_scaled = scaler.fit_transform(X)

# Konversi X_scaled kembali ke DataFrame
X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

# Splitting the dataset
x_train, x_test, y_train, y_test = train_test_split(X_scaled_df, Y, test_size=0.3, stratify=Y, random_state=42)

# Buat model SVC dan definisikan parameter yang sesuai:
clf = SVC(kernel='linear', C=1.0, random_state=0)

# Ganti angka berikut dengan jumlah fitur yang ingin Anda pilih
n_features_to_select = 9
rfe = RFE(estimator=clf, n_features_to_select=n_features_to_select, step=1)
selector = rfe.fit(x_train, y_train)
selected_features = pd.DataFrame({'Feature': list(X.columns), 'Ranking': selector.ranking_})
selected_features.sort_values(by='Ranking', inplace=True)

# Memilih fitur yang distandardisasi
X_scaled_selected = X_scaled_df.loc[:, selector.support_]

# Menyimpan DataFrame yang telah distandardisasi dan diseleksi ke Excel
excel_path = 'static/standarisasi.xlsx'
X_scaled_selected.to_excel(excel_path, index=False)

# Menampilkan fitur yang diseleksi
print("Selected Features:")
selected_features_list = selected_features[selected_features['Ranking'] == 1]['Feature'].tolist()
print(selected_features_list)

# Saving selected features to a pickle file
selected_features_path = 'static/selected_features.pkl'  # Sesuaikan path sesuai kebutuhan
with open(selected_features_path, 'wb') as f:
    pickle.dump(selected_features_list, f)

# Simpan data ke dalam objek
data_to_pickle = {
    'confusion_matrix': 'static/img/photos/confusion_matrix.png',
    'selected_features': selected_features
}

# Model Building
X_scaled = df.drop(['num','age','fbs','restecg','chol'], axis=1)
Y = df['num']

# Print dataframe
print(X_scaled)

# Simpan dataset setelah drop kolom tertentu
heart_featureselection_path = 'static/heart_featureselection.pkl'  # Sesuaikan path sesuai kebutuhan
with open(heart_featureselection_path, 'wb') as f3:
    pickle.dump(X_scaled_selected, f3)

# Memperbarui data_to_pickle dengan path heart_featureselection.pkl
data_to_pickle['heartdrop'] = heart_featureselection_path


# Splitting the dataset
x_train, x_test, y_train, y_test = train_test_split(X_scaled_selected, Y, test_size=0.3, stratify=Y, random_state=13)


# Tentukan hyperparameter yang ingin disesuaikan
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [10, 15, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Inisialisasi model Random Forest
RF = RandomForestClassifier(random_state=42)

# Gunakan GridSearchCV untuk mencari kombinasi hyperparameter terbaik
grid_search = GridSearchCV(estimator=RF, param_grid=param_grid, cv=5, scoring='accuracy')
grid_search.fit(x_train, y_train)

# Ambil model terbaik dari hasil pencarian grid
best_RF = grid_search.best_estimator_

# Latih model terbaik pada data pelatihan
best_RF.fit(x_train, y_train)

# Prediksi kelas dengan model terbaik pada data pelatihan
y_train_pred = best_RF.predict(x_train)

# Prediksi kelas dengan model terbaik pada data uji
y_test_pred = best_RF.predict(x_test)

# Hitung akurasi
y_pred = best_RF.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print("Akurasi Model Terbaik:", accuracy)

####################################
# Hitung akurasi pada data pelatihan dan data uji
accuracy_train = accuracy_score(y_train, y_train_pred)

# Tampilkan akurasi
print("Akurasi pada Data Pelatihan:", accuracy_train)
######################################

# Membuat prediksi pada data test
RF_y_pred = best_RF.predict(x_test)

# Ubah tipe data prediksi menjadi string
RF_y_pred_str = RF_y_pred.astype(str)

# Membuat confusion matrix
cm = confusion_matrix(y_test, RF_y_pred)

# Menampilkan confusion matrix menggunakan ConfusionMatrixDisplay
cm_display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])

# Plot confusion matrix
cm_display.plot(cmap="Blues", values_format='d', colorbar=False)
plt.title("Random Forest Confusion Matrix")
plt.xlabel('Data yang di Prediksi')
plt.ylabel('Data Fakta')
plt.savefig('static/img/photos/confusion_matrix.png')  # Menyimpan gambar sebagai file PNG
plt.show()


# Melakukan kalkulasi dan menampilkan akurasion
# Konversi hasil prediksi ke dalam string
RF_y_test_str = y_test.astype(str)  # Ubah tipe data label uji menjadi string
classification_report_str = classification_report(RF_y_test_str, RF_y_pred_str)
print(classification_report_str)


# Simpan data ke dalam objek
data_to_pickle = {
    'confusion_matrix': 'static/img/photos/confusion_matrix.png',
    'selected_features': selected_features,
    'accuracy': accuracy,
    'classification_report': classification_report_str,
    'heartdrop': X  
}


# Simpan objek ke dalam file pickle
with open('data_file.pkl', 'wb') as f:
    pickle.dump(data_to_pickle, f)
    
# Simpan akurasi ke dalam file pickle
with open('akurasiakhir.pkl', 'wb') as f1:
    pickle.dump(accuracy, f1)

# Simpan classification report ke dalam file pickle
with open('cfreport.pkl', 'wb') as f2:
    pickle.dump(classification_report_str, f2)

# Simpan heartdrop ke dalam file pickle
with open('heart_featureselection.pkl', 'wb') as f3:
    pickle.dump(X, f3)




# Creating a pickle file for the classifier
filename = 'heart-disease-prediction-RFC-SVC-RFE.pkl'
pickle.dump(best_RF, open(filename, 'wb'))


