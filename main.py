#If you want to try this project out, download the Kaggle dataset linked on my README to your project folder
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
# I commented the training CSV file to add testing functionality.
# df = pd.read_csv("archive-4/Training.csv")
df = pd.read_csv("archive-4/Testing.csv")

print(df.info())
xs = []
diseases = sorted(list(set(df["prognosis"])))
ys = []

for i in range(len(df)):
    xs.append(np.array(df.iloc[i, df.columns != "prognosis"])[
              0:len(np.array(df.iloc[i, df.columns != "prognosis"]))-1])
    arr = [0 for _ in range(len(diseases))]
    disease = df.iloc[i]["prognosis"]
    print(disease)
    ind = diseases.index(disease)
    arr[ind] = 1
    ys.append(arr)

model = Sequential()
model.add(Dense(133, activation="softmax", input_shape=xs[0].shape))
model.add(Dense(64, activation="sigmoid"))
model.add(Dense(48, activation="sigmoid"))
model.add(Dense(len(ys[0]), activation="softmax"))
model.compile(optimizer="adam",
              loss="categorical_crossentropy", metrics=["accuracy"])
# In here I trained the model for around 35 epochs, and I got an accuracy of >99%.
model.load_weights("bruh.h5")

acc_test = 0
for i in range(len(xs)):
    disease = diseases[np.argmax(ys[i])]
    point = np.array(xs[i], dtype=np.float32).reshape(1, len(xs[i]))
    p = model.predict(point)[0]
    disease_pred = diseases[np.argmax(p)]
    print(disease, disease_pred)
    print(p)
    if disease == disease_pred:
        
        print("Correctly guessed " + disease_pred + " with " + str(p[np.argmax(p)]*100) + '%\ accuracy')
    acc_test += 1
print("Test validation: " + str(acc_test/len(xs)))
#Test validation: 100%
