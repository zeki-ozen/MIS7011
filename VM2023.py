
# https://www.kaggle.com/datasets/sagnik1511/car-insurance-data
# https://github.com/xzhangfox/Prediction-of-Car-Insurance-Claims/blob/master/Code/FinalProjectCodes.py
# https://machinelearningmastery.com/predicting-car-insurance-payout/
# https://www.kaggle.com/code/davegn/car-insurance-claims-classification/notebook
# https://www.kaggle.com/code/vinhtq115/decision-tree-classifier/data


# gerekli kutuphaneler yukleniyor
import pandas as pd
from keras import Sequential
from keras.layers import Dense
from matplotlib import pyplot
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

df = pd.read_csv('Car_Insurance_Claim.csv')

#veri seti nitelikleri hakkinda bilgi aliyoruz
df.info()

# veri setine goz atiyoruz
df.head()


# show summary statistics
print(df.describe())



# # plot histograms
from matplotlib import pyplot
df.hist()
pyplot.show()

#veri setindeki kayip deger varligini kontrol ediyoruz
df.isna().sum()

#veri setinde tekrar eden satirlarin varligi kontrol ediliyor
df.duplicated().sum()
df=df.drop_duplicates()

# GEREKSİZ SUTUNLAR SİLİNİYOR
df = df.drop(['ID', 'POSTAL_CODE'], axis=1)

#replace null values with averages from columns
df["CREDIT_SCORE"] = df["CREDIT_SCORE"].fillna(df["CREDIT_SCORE"].mean())
df["ANNUAL_MILEAGE"] = df["ANNUAL_MILEAGE"].fillna(df["ANNUAL_MILEAGE"].mean())
df.describe()

# float veri tiplerini int veri tipine ceviriyoruz
df = df.astype({'OUTCOME':'int', 'VEHICLE_OWNERSHIP' :'int', 'MARRIED':'int', 'CHILDREN':'int', 'ANNUAL_MILEAGE':'int', 'OUTCOME':'int'})


#kategorik nitelikleri dummy table haline getiriyoruz
df = pd.get_dummies(df, drop_first=True)


#bagimsiz nitelikleri ve bagimli niteligi iki ayri degiskene aliyoruz
X = df.drop(['OUTCOME'], axis=1)
y = df['OUTCOME']



# Egitim ve Test verlerini %70-%30 oraninda ayarliyoruz.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)





#libtune to tune model, get different metric score
from collections import Counter
# summarize the class distribution of the training dataset
counter = Counter(y_train)
print(counter)

# veri setlerimizi normalize ediyoruz
# ilk sutunu normalize etme, haric birak
scaler = MinMaxScaler()

X_train.loc[:, X_train.columns!='CREDIT_SCORE'] = scaler.fit_transform(X_train.loc[:, X_train.columns!='CREDIT_SCORE'])
X_test.loc[:, X_test.columns!='CREDIT_SCORE'] = scaler.fit_transform(X_test.loc[:, X_test.columns!='CREDIT_SCORE'])

# float veri tiplerini int veri tipine ceviriyoruz
X_train = X_train.astype({ 'VEHICLE_OWNERSHIP' :'int', 'MARRIED':'int', 'CHILDREN':'int', 'ANNUAL_MILEAGE':'int'})
X_test = X_test.astype({ 'VEHICLE_OWNERSHIP' :'int', 'MARRIED':'int', 'CHILDREN':'int', 'ANNUAL_MILEAGE':'int'})



enc = OneHotEncoder(handle_unknown='ignore')
X = [['Male', 1], ['Female', 3], ['Female', 2]]
enc.fit(X)


# transform the training dataset
#resampling library
from imblearn.over_sampling import SMOTE
oversample = SMOTE(random_state=33)
X_train, y_train = oversample.fit_resample(X_train, y_train)
# summarize the new class distribution of the training dataset
counter = Counter(y_train)
print(counter)


# determine the number of input features
n_features = X_train.shape[1]
# define model
model = Sequential()
model.add(Dense(10, activation='relu', kernel_initializer='he_normal', input_shape=(n_features,)))
model.add(Dense(1))
# compile the model
model.compile(optimizer='adam', loss='mse')
# fit the model
history = model.fit(X_train, y_train, epochs=100, batch_size=8, verbose=1, validation_data=(X_test,y_test))
# predict test set
predictions = model.predict(X_test)
print(predictions)
# evaluate predictions
score = mean_absolute_error(y_test, predictions)
print('MAE: %.3f' % score)

training_score = model.score(X_train, y_train)
acc = accuracy_score(y_test, predictions)
con = confusion_matrix(y_test, predictions)
report = classification_report(y_test, predictions)
print(f'Training Score: {training_score}')
print(f'Accuracy Score: {acc}')
print(f'Confusion Matrix: {con}')
print(f'Classification Report: {report}')


# plot learning curves
pyplot.title('Learning Curves')
pyplot.xlabel('Epoch')
pyplot.ylabel('Mean Squared Error')
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='val')
pyplot.legend()
pyplot.show()