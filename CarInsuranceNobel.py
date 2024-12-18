# https://www.kaggle.com/datasets/sagnik1511/car-insurance-data
# https://github.com/xzhangfox/Prediction-of-Car-Insurance-Claims/blob/master/Code/FinalProjectCodes.py
# https://machinelearningmastery.com/predicting-car-insurance-payout/
# https://www.kaggle.com/code/davegn/car-insurance-claims-classification/notebook
# https://www.kaggle.com/code/vinhtq115/decision-tree-classifier/data

# gerekli kutuphaneler koda yukleniyor
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import warnings

warnings.filterwarnings('ignore')
# veriyi manipule etmek icin
import pandas as pd
#matematiksel islemelr icin
import numpy as np
# veriyi bolumeme, performans degerlendirmesi icin
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, ConfusionMatrixDisplay, auc, \
    precision_score, \
    recall_score, f1_score, classification_report, roc_curve, precision_recall_curve, \
     average_precision_score
from sklearn.preprocessing import MinMaxScaler, QuantileTransformer, PowerTransformer
# oversampling uygulamak icin
from imblearn.over_sampling import ADASYN
# ysa mimarisi kurmak icin
import tensorflow as tf
from keras import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from keras.optimizers import Adam

from statsmodels.stats.outliers_influence import variance_inflation_factor
#grafik cidirmek icin
from matplotlib import pyplot as plt
import seaborn as sns

import random

pd.set_option('display.width', 400)
pd.set_option('display.max_columns', 15)

# Veri seti calisma ortamina yukleniyor
df = pd.read_csv('Car_Insurance_Claim.csv')


##############################
# 1. Veriyi Anlama
##############################
# Veri setinde gerek duyulmayan ID niteligi siliniyor
df = df.drop(['ID'], axis=1)

# Veri setinin boyutlarina (satir ve sutun sayilarina) bakiliyor
df.shape

# Veri seti nitelik adlarina bakiliyor
print(df.columns)

# Niteliklerin veri tiplerine bakiliyor
df.info()

# Veri setinin ozetine bakiliyor
print(df.describe().apply(lambda s: s.apply('{0:.2f}'.format)).T)
# print(df.describe(include="all"))

# Veri setinin ilk birkac satirina goz atiyoruz
df.head()

# for kolon in df.columns:
#     print(df[kolon].value_counts())

#

##############################
# 2. Veri On-isleme adimi
##############################
# veri setinde kayip deger olup olmadigi kontrol ediliyor
df.isna().sum()

# kayip degerler, o niteligin rasgele bir degeriyle degistiriliyor
#df['ANNUAL_MILEAGE'].fillna(random.choice(df['ANNUAL_MILEAGE'][df['ANNUAL_MILEAGE'].notna()]), inplace=True)
#df['CREDIT_SCORE'].fillna(random.choice(df['CREDIT_SCORE'][df['CREDIT_SCORE'].notna()]), inplace=True)



# df = df.dropna()
# kayip degerler, o niteligin ortalamasi ile dolduruluyor
df["CREDIT_SCORE"] = df["CREDIT_SCORE"].fillna(df["CREDIT_SCORE"].mean())
df["ANNUAL_MILEAGE"] = df["ANNUAL_MILEAGE"].fillna(df["ANNUAL_MILEAGE"].mean())

# kayip degerlerin tamamlandigini kontrol ediyoruz
df.isna().sum()

# veri setinde tekrar eden satirlarin varligi kontrol ediliyor
df.duplicated().sum()
# veri setinde tekrar eden satirlar siliniyor
df = df.drop_duplicates()

# kategorik nitelikler dummy table haline getiriliyor
df = pd.get_dummies(df, drop_first=True)
df = pd.get_dummies(df, columns=['POSTAL_CODE'], drop_first=True)

# veri setindeki niteliklerin histogram grafigi
df.hist(figsize=(15, 15))
plt.show()

# korelasyon matrisi hesaplaniyor
kor = df.corr().round(2)

# korelasyon matrisi isi haritasi ile gorsellestiriliyor
plt.subplots(figsize=(12, 10))
mask = np.triu(np.ones_like(kor, dtype=bool))
cmap = sns.diverging_palette(230, 20)
sns.heatmap(kor, annot=True, vmin=-1, vmax=1, mask=mask, cmap=cmap)
plt.show()
# plt.savefig('heatmap.png')


# kutu grafikleri
numerik_nitelikler = ['CREDIT_SCORE', 'ANNUAL_MILEAGE', 'SPEEDING_VIOLATIONS', 'DUIS', 'PAST_ACCIDENTS']
# kutu grafigi yanyana cizdirmek icin 1 satir 5 kolonluk grafik olusturuluyor
fig, ax = plt.subplots(1, 5, figsize=(10, 6))
plt.subplots_adjust(wspace=1)
colors = ['pink', 'lightblue', 'lightgreen', 'brown', 'red']
# surekli niteliklerin kutu grafigi ciziliyor
for x in range(5):
    sns.boxplot(data=df[numerik_nitelikler[x]], ax=ax[x], color=colors[x])
    ax[x].set_xlabel(numerik_nitelikler[x])
plt.show()

for kolon in numerik_nitelikler:
    print(df[kolon].value_counts())

df2 = df


# kolon ='CREDIT_SCORE'   # 8 tane
# kolon ='ANNUAL_MILEAGE'   # 51 tane
# kolon ='SPEEDING_VIOLATIONS'   # 746 tane
# kolon ='PAST_ACCIDENTS'   # 422 tane

# aykiri deger tespit fonksiyonu
def outlier_sil(df, kolon):
    birinci_ceyrek = df[kolon].quantile(0.25)
    ucuncu_ceyrek = df[kolon].quantile(0.75)
    IQR = ucuncu_ceyrek - birinci_ceyrek
    dusuk = birinci_ceyrek - 1.5 * IQR
    yuksek = ucuncu_ceyrek + 1.5 * IQR
    yeni_df = df.loc[(df[kolon] > dusuk) & (df[kolon] < yuksek)]
    return yeni_df

# normal dagilim gosteren numerik niteliklerin aykiri degerleri siliniyor
outlier_kontrol_edilecek_nitelikler = ['CREDIT_SCORE', 'ANNUAL_MILEAGE']
for kolon in outlier_kontrol_edilecek_nitelikler:
    df = outlier_sil(df, kolon)

# carpik numerik niteliklerin histogram grafigi - tek grafikte
carpik_numerik_nitelikler = ['SPEEDING_VIOLATIONS', 'PAST_ACCIDENTS', 'DUIS']
fig, ax = plt.subplots(1, 3, sharex=False, figsize=(8, 8))
for x in range(3):
    sns.distplot(df[carpik_numerik_nitelikler[x]], ax=ax[x], hist=True)
plt.show()

# carpik numerik niteliklerin histogram grafigi - ayri ayri
for kolon in carpik_numerik_nitelikler:
    sns.distplot(df[kolon], hist=True)
    plt.show()

# log transformasyon
for kolon in carpik_numerik_nitelikler:
    df[kolon] = df[kolon].map(lambda i: np.log(i) if i > 0 else 0)


def compute_vif(X):
    # the calculation of variance inflation requires a constant
    X['intercept'] = 1

    # create dataframe to store vif values
    vif = pd.DataFrame()
    vif["Variable"] = X.columns
    vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    vif = vif[vif['Variable'] != 'intercept']
    return vif


def calc_vif(X):
    # Calculating VIF
    vif = pd.DataFrame()
    vif["variables"] = X.columns
    vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    return (vif)


X = df.drop(['OUTCOME'], axis=1)
compute_vif(X)
calc_vif(X)

vif_data = pd.DataFrame()
vif_data["feature"] = df.columns
# Function to calculate VIF
vif_data["VIF"] = [variance_inflation_factor(X.values, i)
                   for i in range(len(X.columns))]

# kategorik_nitelikler = df.columns[~df.columns.isin(numerik_nitelikler)]
# df[kategorik_nitelikler] = df[kategorik_nitelikler].astype('category')

# df.info()
#


#
# for kolon in numerik_nitelikler:
#    # plt.figure(figsize=(20, 10))
#     plt.title(kolon + ' histogram')
#     sns.histplot(data=df, x=kolon, kde=True)
#     plt.show()
#

# for kolon in numerik_nitelikler:
#     sns.distplot(df[kolon])
#     plt.show()

# skewed data
# https://www.kaggle.com/code/mitramir5/anomaly-detection-skewed-features-and-stories/notebook

# https://opendatascience.com/transforming-skewed-data-for-machine-learning/
# https://datamadness.github.io/Skewness_Auto_Transform
# https://pyimagesearch.com/2021/05/31/hyperparameter-tuning-for-deep-learning-with-scikit-learn-keras-and-tensorflow/

# https://www.kaggle.com/code/amritvirsinghx/eda-basics-handling-skewed-data/notebook
# https://www.pluralsight.com/guides/preparing-data-modeling-scikit-learn
# https://machinelearningmastery.com/quantile-transforms-for-machine-learning/

carpik_nitelikler = ['SPEEDING_VIOLATIONS', 'PAST_ACCIDENTS', 'DUIS']
df.skew().sort_values(ascending=False)

df[carpik_nitelikler].skew().sort_values(ascending=False)

df = None
df2 = df
df3 = df2
df4 = df


df = df3
from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import ColumnTransformer

for kolon in carpik_nitelikler:
    sns.distplot(df[kolon], hist=True)
    plt.show()

# log transformasyon
for kolon in carpik_nitelikler:
    df[kolon] = np.log(df[kolon])

# sqrt root transformasyon
for kolon in carpik_nitelikler:
    df[kolon] = np.sqrt(df[kolon])

# cube root transformsyon
for kolon in carpik_nitelikler:
    df[kolon] = np.cbrt(df[kolon])


# yeo-johnson transformsyon
yeo_johnson = PowerTransformer(method='yeo-johnson')
for kolon in carpik_nitelikler:
    df[kolon] =  yeo_johnson.fit_transform(df[kolon])

# Power transformsyon
power = PowerTransformer()
for kolon in carpik_nitelikler:
    df[kolon] =  power.fit_transform(df[kolon])

# quantile_normal transformsyon
quantile_normal = QuantileTransformer(n_quantiles=100, output_distribution='normal')
for kolon in carpik_nitelikler:
    df[kolon] =  quantile_normal.fit_transform(df[kolon])

# quantile_uniform transformsyon
quantile_uniform = QuantileTransformer(n_quantiles=100, output_distribution='uniform')
for kolon in carpik_nitelikler:
    df[kolon] =  quantile_uniform.fit_transform(df[kolon])

# yeo_johnson = PowerTransformer(method='yeo-johnson')
# df[carpik_nitelikler] = yeo_johnson.fit_transform(df[carpik_nitelikler])
#
# power = PowerTransformer()
# df[carpik_nitelikler] = power.fit_transform(df[carpik_nitelikler])
#
# quantile_uniform = QuantileTransformer(n_quantiles=100, output_distribution='uniform')
# df[carpik_nitelikler] = quantile_uniform.fit_transform(df[carpik_nitelikler])
#
# quantile_normal = QuantileTransformer(n_quantiles=100, output_distribution='normal')
# df[carpik_nitelikler] = quantile_normal.fit_transform(df[carpik_nitelikler])




# https://machinelearningmastery.com/power-transforms-with-scikit-learn/
# https://machinelearningmastery.com/quantile-transforms-for-machine-learning/


df3 = df
df2 = df
df = df2
df = df3

df2 = df


# bagimsiz nitelikleri X degiskenine, bagimli (hedef) niteligi Y degiskenine aliyoruz
X = df.drop(['OUTCOME'], axis=1)
y = df['OUTCOME']

# bunun grafigini cizdir. bu cizmiyor
# df = sns.load_dataset(df)
# sns.pairplot(df, hue="OUTCOME")
# plt.show()
# sns.countplot(y)

# plt.figure(figsize=(20, 10))
# plt.title('Credit Score histogram')
# sns.histplot(data=df, x='CREDIT_SCORE', kde=True)
# plt.show()

# pd.plotting.scatter_matrix(df, alpha=0.2)

y.unique()
# Hedef nitelikteki sinif degerlerinin sayilari kontrol ediliyor
print(y.value_counts())

# Sinif dagilimi pasta grafigi ile goruntuleniyor
df['OUTCOME'].groupby(df['OUTCOME']).count().plot.pie(figsize=(5, 5), autopct='%1.1f%%', startangle=30.)
plt.show()
#print("Sigorta talep edenlerin yuzdesi: %" + str(round(y.value_counts()[0] * 100 / y.shape[0])))
#print("Sigorta talep etmeyenlerin yuzdesi: %" + str(round(y.value_counts()[1] * 100 / y.shape[0])))

# sns.boxplot(x=df['ANNUAL_MILEAGE'])
# plt.show()
#
# for column in df.columns[1:-1]:
#     sns.scatterplot(data=df, x=x, y=column, hue="OUTCOME")
#     plt.show()

# pd.plotting.scatter_matrix(df, alpha=0.2)
# plt.show()


# Veri seti rastgele bicimde %70 egitim ve %30 test veri seti olarak bolunuyor
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
print(y_train.value_counts())
print(y_train.value_counts(normalize=True))

# Egitim veri seti smote sampling yapiliyor
# ADASYN SMOTE metodu ile egitim veri setinde sınıf dagilimlari dengeli hale getiriliyor
adasyn_over_sample = ADASYN(sampling_strategy='minority', random_state=42, n_neighbors=5)
X_train, y_train = adasyn_over_sample.fit_resample(X_train, y_train)
print(y_train.value_counts())
print(y_train.value_counts(normalize=True))

# egitim ve test veri setindeki numerik nitelikler 0-1 araliginde yeniden olcekleniyor
min_max_scaler = MinMaxScaler()
normalize_edilecek_nitelikler = ['ANNUAL_MILEAGE', 'SPEEDING_VIOLATIONS', 'PAST_ACCIDENTS', 'DUIS']
X_train[normalize_edilecek_nitelikler] = min_max_scaler.fit_transform(X_train[normalize_edilecek_nitelikler])
X_test[normalize_edilecek_nitelikler] = min_max_scaler.transform(X_test[normalize_edilecek_nitelikler])

# X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.35, random_state=1)  # 0.35 x 0.7 = 0.2

# summarize the class distribution of the training dataset
print(y_train.value_counts())
print(y_test.value_counts())
# print(y_val.value_counts())



X_train.info()

# float veri tiplerini int veri tipine ceviriyoruz
X_train = X_train.astype({'VEHICLE_OWNERSHIP': 'int', 'MARRIED': 'int', 'CHILDREN': 'int'})
X_test = X_test.astype({'VEHICLE_OWNERSHIP': 'int', 'MARRIED': 'int', 'CHILDREN': 'int'})
# X_val = X_val.astype({'VEHICLE_OWNERSHIP': 'int', 'MARRIED': 'int', 'CHILDREN': 'int'})

y_train = y_train.astype('int')
y_test = y_test.astype('int')
# y_val = y_val.astype('int')
# select all columns except 'rebounds' and 'assists'
# df.loc[:, ~df.columns.isin(['rebounds', 'assists'])]


# from sklearn import preprocessing
# le = preprocessing.LabelEncoder()
# le.fit(["paris", "paris", "tokyo", "amsterdam"])
# list(le.classes_)
# le.transform(["tokyo", "tokyo", "paris"])
# list(le.inverse_transform([2, 2, 1]))

# enc = OneHotEncoder(handle_unknown='ignore')
# X = [['Male', 1], ['Female', 3], ['Female', 2]]
# enc.fit(X)

# resampling yontemi sadece egitim veri setine uygulanir
# test veri setine resampling uygulanmaz


# transform the training dataset
# resampling library
# from imblearn.over_sampling import SMOTE
# oversample = SMOTE(random_state=42)
# X_train2, y_train2 = oversample.fit_resample(X_train, y_train)
# # summarize the new class distribution of the training dataset
# print(y_train2.value_counts())

# df.to_pickle("df.pkl")
# X.to_pickle("X.pkl")
# y.to_pickle("y.pkl")
# X_train.to_pickle("X_train.pkl")
# X_test.to_pickle("X_test.pkl")
# y_train.to_pickle("y_train.pkl")
# y_test.to_pickle("y_test.pkl")
#

df = pd.read_pickle("df.pkl")
X = pd.read_pickle("X.pkl")
y = pd.read_pickle("y.pkl")
X_train = pd.read_pickle("X_train.pkl")
X_test = pd.read_pickle("X_test.pkl")
y_train = pd.read_pickle("y_train.pkl")
y_test = pd.read_pickle("y_test.pkl")

from keras import (layers, optimizers, losses)
from keras import regularizers

# https://stats.stackexchange.com/a/136542
# Set random seed
tf.random.set_seed(42)
# define model
ysa_model = None
ysa_model = Sequential()
# model.add(Dense(100, activation='relu', kernel_initializer='he_normal', input_shape=(X.shape[1],)))
# model.add(Dense(20, activation ='relu',    kernel_regularizer = regularizers.l1(1e-2), input_shape=(X.shape[1],)))
ysa_model.add(Dense(200, activation='relu', input_shape=(X.shape[1],)))
ysa_model.add(BatchNormalization())
ysa_model.add(Dropout(0.3))
ysa_model.add(Dense(100, activation='relu'))
ysa_model.add(BatchNormalization())
ysa_model.add(Dropout(0.3))
ysa_model.add(Dense(50, activation='relu'))
ysa_model.add(BatchNormalization())
ysa_model.add(Dropout(0.3))
ysa_model.add(Dense(1, activation='sigmoid'))  # binary activation output
# model.add(Dense(2, activation='softmax')) # binary activation output


# https://www.kaggle.com/code/safayet1610039/heart-attack-prediction-using-neural-network
#
# Building Neural Network for classification
# Model Arcitechture
# In this task we used a neural network composing of these following layers, activation functions and optmizer and loss function:
# Input Layer : Takes input
# Dense Layer : Dense layer is the regular deeply connected neural network layer.
# BatchNormalization Layer : Batch normalization applies a transformation that maintains the mean output close to 0 and the output standard deviation close to 1.
# The Dropout layer randomly sets input units to 0 with a frequency of rate at each step during training time, which helps prevent overfitting. Inputs not set to 0 are scaled up by 1/(1 - rate) such that the sum over all inputs is unchanged
# Activation function: An activation function in a neural network defines how the weighted sum of the input is transformed into an output from a node or nodes in a layer of the network.
#
# Regularization : Regularization is a technique which makes slight modifications to the learning algorithm such that the model generalizes better. This in turn improves the model's performance on the unseen data as well.
#
# Activation Function :
# ReLU :The rectified linear activation function or ReLU for short is a piecewise linear function that will output the input directly if it is positive, otherwise, it will output zero,
# Sigmoid : Sigmoid activation function, sigmoid(x) = 1 / (1 + exp(-x)). Applies the sigmoid activation function. For small values (<-5), sigmoid returns a value close to zero, and for large values (>5) the result of the function gets close to 1.
# Regularization :
# L1 regulazier : In L1 norm we shrink the parameters to zero.
# Adam Optimizer :
# Adam is a replacement optimization algorithm for stochastic gradient descent for training deep learning models. Adam combines the best properties of the AdaGrad and RMSProp algorithms to provide an optimization algorithm that can handle sparse gradients on noisy problems.
# SGD Optimizer :
# Stochastic Gradient Descent (SGD) is a simple yet very efficient approach to fitting linear classifiers and regressors under convex loss functions such as (linear) Support Vector Machines and Logistic Regression.
# Binary Crossentropy Loss:
# Cross-entropy is the default loss function to use for binary classification problems. It is intended for use with binary classification where the target values are in the set {0, 1}.
#
# compile the model
# model.compile(optimizer='adam', loss='mse')

ysa_model.compile(
    # optimizer='sgd',
    optimizer='adam',
    # optimizer=tf.keras.optimizers.Adam(lr=0.02),
    # optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    # loss=tf.keras.losses.CategoricalCrossentropy(),
    loss=tf.keras.losses.BinaryCrossentropy(),
    # metrics=['binary_accuracy']
    metrics=['accuracy']
)

# Create a learning rate scheduler callback
# lr_scheduler = tf.keras.callbacks.LearningRateScheduler(lambda epoch: 1e-4 * 10**(epoch/20)) # traverse a set of learning rate values starting from 1e-4, increasing by 10**(epoch/20) every epoch


early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, min_delta=0.001,
                                                  restore_best_weights=True)
# fit the model
# ann_model = model.fit(X_train, y_train, epochs=100, batch_size=8, verbose=0, validation_data=(X_test,y_test))
# history  = model.fit(X_train, y_train, epochs=100, batch_size=8, verbose=1, callbacks=[lr_scheduler])
# history = model.fit(X_train, y_train, validation_split=0.2, epochs=200, batch_size=8, verbose=1)
# history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=200,
history = ysa_model.fit(np.asarray(X_train).astype('float32'), np.asarray(y_train).astype('float32'), validation_split=0.2, epochs=200,
                    # batch_size=8,
                    verbose=1,
                    callbacks=[early_stopping],
                    # use_multiprocessing = True
                    )

ysa_model.summary()
# model.save('model.ann')
# evaluate the model

# modelin egitim dogruluguna ve kayip degerine bakiliyor
test_loss, test_acc = ysa_model.evaluate(np.asarray(X_test).astype('float32'), y_test, verbose=0)
test_loss
test_acc

# Egitilmis Yapay sinir agi modeli test veri seti ile test ediliyor
predictions = ysa_model.predict(np.asarray(X_test).astype('float32'))

# tahmin sonuclarina goz gezdiriliyor
# print(predictions)

# tahmin sonuclari 0.5 degerinin uzerindeyse 1'e, altindaysa 0'a yuvarlaniyor
prediction_classes = [1 if prob > 0.5 else 0 for prob in np.ravel(predictions)]
# prediction_classes2 = predictions.round()
# prediction_classes3 = tf.round(predictions)

# ConfusionMatrix ile modelin performans degerleri elde ediliyor
conf_mat = confusion_matrix(y_test, prediction_classes)
# conf_mat2 = confusion_matrix(y_test, prediction_classes2)
# conf_mat3 = confusion_matrix(y_test, prediction_classes3)

print(conf_mat)

# Modelin performans raporu ciktilaniyor
print(classification_report(y_test, prediction_classes))

# Muhim performans metrikleri ciktilaniyor
print(f'Accuracy: {accuracy_score(y_test, prediction_classes):.2f}')
print(f'Precision: {precision_score(y_test, prediction_classes):.2f}')
print(f'Recall: {recall_score(y_test, prediction_classes):.2f}')
print(f'F1Score: {f1_score(y_test, prediction_classes):.2f}')



# plotting model accuracy and loss combined
pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.show()

# plotting model accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

# plotting model loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

labels = ['Kredi almayan', 'Kredi alan']
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(conf_mat, cmap=plt.cm.Blues)
fig.colorbar(cax)
ax.set_xticklabels([''] + labels)
ax.set_yticklabels([''] + labels)
plt.xlabel('Tahmini Değerler')
plt.ylabel('Asıl Değerler')
plt.show()

labels = ['Kasko kullanan', 'Kasko kullanmayan']
sns.heatmap(conf_mat, square=True, annot=True, fmt='d', cbar=False, cmap='Blues',
            xticklabels=labels, yticklabels=labels)

plt.xlabel('Tahmini Değerler')
plt.ylabel('Asıl Değerler')
plt.show()

print('TF YSA Sınıflandırma:')
print(classification_report(y_test, prediction_classes))

score_nn = round(accuracy_score(prediction_classes, y_test) * 100, 2)
print("The accuracy score achieved using Neural Network is: " + str(score_nn) + " %")


def plot_roc_curve(fpr, tpr):
    plt.plot(fpr, tpr, color='orange', label='ROC')
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.show()


# Computing manually fpr, tpr, thresholds and roc auc
fpr, tpr, thresholds = roc_curve(y_test, predictions)
roc_auc = auc(fpr, tpr)
print("ROC_AUC Score : ", roc_auc)
print("Function for ROC_AUC Score : ", roc_auc_score(y_test, predictions))  # Function present
optimal_idx = np.argmax(tpr - fpr)
optimal_threshold = thresholds[optimal_idx]
print("Threshold value is:", optimal_threshold)
plot_roc_curve(fpr, tpr)

# Checkout the history
pd.DataFrame(history.history).plot(figsize=(10, 7), xlabel="epochs");
plt.show()

# Plot the learning rate versus the loss

# plot learning curves
plt.title('Learning Curves')
plt.xlabel('Epoch')
plt.ylabel('Mean Squared Error')
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='val')
plt.legend()
plt.show()


def create_model(learning_rate=0.1, dropout_rate=0.5, activation='softsign', optimizer='Adam', init_mode='uniform'):
    model = Sequential()
    # model.add(Dense(100, activation='relu', kernel_initializer='he_normal', input_shape=(X.shape[1],)))
    # model.add(Dense(20, activation ='relu',    kernel_regularizer = regularizers.l1(1e-2), input_shape=(X.shape[1],)))
    model.add(Dense(20, activation=activation, kernel_initializer=init_mode, input_shape=(X.shape[1],)))
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))
    model.add(Dense(10, activation=activation, kernel_initializer=init_mode, ))
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))
    model.add(Dense(5, activation=activation, kernel_initializer=init_mode))
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))
    model.add(Dense(1, activation='sigmoid', kernel_initializer=init_mode))  # binary activation output

    adam = Adam(learning_rate=0.1)
    model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
    return model


# Create the model
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier


n_iter_search = 16  # Number of parameter settings that are sampled.
init_mode = ['uniform', 'lecun_uniform', 'normal', 'zero', 'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform']
init_mode = ['uniform']

epochs = np.array([50, 100, 200])
batches = np.array([8, 32])
learning_rate = [0.001, 0.01, 0.1]
dropout_rate = [0.1, 0.2, 0.3, 0.5]
optimizer = ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']
activation = ['softmax', 'softplus', 'softsign', 'relu', 'tanh', 'sigmoid', 'hard_sigmoid', 'linear']

param_grid = dict(nb_epoch=epochs,
                  batch_size=batches,
                  init_mode=init_mode,
                  learning_rate=learning_rate,
                  dropout_rate=dropout_rate,
                  activation=activation,
                  optimizer=optimizer
                  )

model = KerasClassifier(build_fn=create_model, verbose=0)
cv = [(slice(None), slice(None))]
grid = GridSearchCV(estimator=model, param_grid=param_grid, verbose=10, cv=cv, n_jobs=-1)
grid_result = grid.fit(X_train, y_train, validation_split=0.2)

import joblib

#
# #save your model or results
# joblib.dump(grid_result, 'grid_result.pkl')
joblib.dump(grid, 'grid2.pkl')
# joblib.load("grid_result.pkl")

# Summarize the results
print(" Results from Grid Search ")
print("\n The best estimator across ALL searched params:\n", grid_result.best_estimator_)
print("\n The best score across ALL searched params:\n", grid_result.best_score_)
print("\n The best parameters across ALL searched params:\n", grid_result.best_params_)

sorted(grid_result.cv_results_.keys())

# model_best = create_model(learning_rate=0.1, dropout_rate=0.5, activation='softsign', optimizer='Adam', init_mode='uniform')
# early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, min_delta=0.001,
#                                                   restore_best_weights=True)
#
# history = model_best.fit(X_train, y_train,
#                          validation_split=0.2,
#                          epochs= 100
#                     )
# test_loss, test_acc = model_best.evaluate(X_test, y_test, verbose=0)
# test_loss
# test_acc
# # predict test set
predictions = grid_result.predict(X_test)
# print(predictions)

# print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


prediction_classes = [1 if prob > 0.5 else 0 for prob in np.ravel(predictions)]
prediction_classes2 = predictions.round()
prediction_classes3 = tf.round(predictions)

conf_mat = confusion_matrix(y_test, prediction_classes)
conf_mat2 = confusion_matrix(y_test, prediction_classes2)
conf_mat3 = confusion_matrix(y_test, prediction_classes3)

print(conf_mat)

print(f'Accuracy: {accuracy_score(y_test, prediction_classes):.2f}')
print(f'Precision: {precision_score(y_test, prediction_classes):.2f}')
print(f'Recall: {recall_score(y_test, prediction_classes):.2f}')
print(f'F1Score: {f1_score(y_test, prediction_classes):.2f}')

print('Best : {}, using {}'.format(grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print('{},{} with: {}'.format(mean, stdev, param))

