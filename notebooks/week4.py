# Wednesday, January 30, 2019

## Load in the data set (Internet Access needed)

url = "http://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.data"
names = ["times_pregnant", "glucose_tolerance_test", "blood_pressure", "skin_thickness", "insulin", 
         "bmi", "pedigree_function", "age", "has_diabetes"]
diabetes_df = pd.read_csv(url, names=names)

# Building a single hidden neural network
normalizer = StandardScaler()
X_train_norm = normalizer.fit_transform(X_train)
X_test_norm = normalizer.transform(X_test)

model_1 = Sequential([
    Dense(12, input_shape=(8,), activation="relu"),
    Dense(1, activation="sigmoid")
])

## Thursday, January 31, 2019
from keras.models import Sequential, K
from keras.layers import Dense, Input, Flatten, Dropout, BatchNormalization
from keras.optimizers import Adam, SGD, RMSprop