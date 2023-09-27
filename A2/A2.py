import os
import pickle
import ReadCSV
import Readimg

from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


#prepare img for training
input_dir_img = './Datasets/celeba/img'
data = Readimg.img_reader(input_dir_img)

#prepare labels for training
input_dir_label = './Datasets/celeba/labels.csv'
label_name = 'smiling'
labels = ReadCSV.csv(input_dir_label, label_name)

#prepare img for testing
input_dir_img_test = './Datasets/celeba_test/img'
data_test = Readimg.img_reader(input_dir_img_test)

#prepare labels for testing
input_dir_label_test = './Datasets/celeba_test/labels.csv'
label_name_test = 'smiling'
labels_test = ReadCSV.csv(input_dir_label_test, label_name_test)

#use SVM as the method for Gender detection
svm = SVC()

#try different parameter combination to find the best model - rbf kernel
parameters = [{'gamma': [0.01, 0.001, 0.0001], 'C': [1, 10, 100, 1000]}]

grid_search = GridSearchCV(svm, parameters,return_train_score=True)
grid_search.fit(data, labels)

#check testing score for each parameter combination
train_scores = grid_search.cv_results_
print (train_scores)

# test performance
best_smilingdetection = grid_search.best_estimator_
labels_prediction = best_smilingdetection.predict(data_test)

score = accuracy_score(labels_prediction, labels_test)

print('{}% of samples were correctly classified'.format(str(score * 100)))

with open('smilingdetection_model.pkl', 'wb') as file:
    pickle.dump(best_smilingdetection, file)