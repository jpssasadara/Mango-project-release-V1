import numpy as np

data=np.load('data.npy')
target=np.load('target.npy')

#loading the save numpy arrays in the previous code

from sklearn.model_selection import train_test_split

train_data,test_data,train_target,test_target=train_test_split(data,target,test_size=0.1)

from sklearn.neighbors import KNeighborsClassifier

algorithm = KNeighborsClassifier()
algorithm.fit(train_data,train_target)
predicted_target = algorithm.predict(test_data)


print("Actual Target : ",test_target)
print("Predicted Target :",predicted_target)


from sklearn.metrics import accuracy_score

acc = accuracy_score(test_target,predicted_target)
print("Accuracy : ",acc)

# $$$$$$  Saving algo  --> pip install joblib  $$$$$$$$$$$$$$$$$$
import joblib
joblib.dump(algorithm,'KNN_model.sav')

