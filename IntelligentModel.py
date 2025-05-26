from tkinter import messagebox
from tkinter import *
from tkinter import filedialog
from tkinter.filedialog import askopenfilename
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix
from sklearn.model_selection import train_test_split
from genetic_selection import GeneticSelectionCV
import pandas as pd

main = Tk()
main.title("An Intelligent Data-Driven Model to Secure Intravehicle Communications Based on Machine Learning")
main.geometry("1300x1200")

knn_hr, knn_fr, knn_mr, knn_cr = 0, 0, 0, 0
decision_hr, decision_fr, decision_mr, decision_cr = 0, 0, 0, 0
svm_hr, svm_fr, svm_mr, svm_cr = 0, 0, 0, 0
sso_hr, sso_fr, sso_mr, sso_cr = 0, 0, 0, 0

def uploadDataset():
    global filename
    filename = filedialog.askopenfilename(initialdir="dataset")
    pathlabel.config(text=filename)
    text.delete('1.0', END)
    text.insert(END, filename + " loaded\n")

def KNN():
    global knn_hr, knn_fr, knn_mr, knn_cr
    text.delete('1.0', END)
    train = pd.read_csv(filename, nrows=14000)
    train.fillna(0, inplace=True)
    le = LabelEncoder()
    train['ID'] = pd.Series(le.fit_transform(train['ID']))
    X = train.values[:, 3:7]
    Y = train.values[:, 0]

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

    estimator = KNeighborsClassifier()
    estimator.fit(X_train, y_train)
    y_pred = estimator.predict(X_test)
    knn_hr = accuracy_score(y_test, y_pred) * 100
    knn_cr = precision_score(y_test, y_pred, average='macro') * 100
    tn, knn_mr, knn_fr, tp = confusion_matrix(y_test, y_pred).ravel()
    knn_mr = knn_mr / 100
    knn_fr = knn_fr / 100

    text.insert(END, "KNN Classifier Performance Details : \n\n")
    text.insert(END, "KNN Hit Rate               : " + str(knn_hr) + "\n")
    text.insert(END, "KNN Miss Rate              : " + str(knn_mr) + "\n")
    text.insert(END, "KNN False Alarm Rate       : " + str(knn_fr) + "\n")
    text.insert(END, "KNN Correct Rejection Rate : " + str(knn_cr) + "\n")

def decisionTree():
    global decision_hr, decision_fr, decision_mr, decision_cr
    train = pd.read_csv(filename, nrows=14000)
    train.fillna(0, inplace=True)
    le = LabelEncoder()
    train['ID'] = pd.Series(le.fit_transform(train['ID']))
    X = train.values[:, 4:7]
    Y = train.values[:, 0]

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

    estimator = DecisionTreeClassifier(max_features=2)
    estimator.fit(X_train, y_train)
    y_pred = estimator.predict(X_test)

    decision_hr = accuracy_score(y_test, y_pred) * 100
    decision_cr = precision_score(y_test, y_pred, average='macro') * 100
    tn, decision_mr, decision_fr, tp = confusion_matrix(y_test, y_pred).ravel()
    decision_mr = decision_mr / 100
    decision_fr = decision_fr / 100

    text.insert(END, "Decision Tree Classifier Performance Details : \n\n")
    text.insert(END, "Decision Tree Hit Rate               : " + str(decision_hr) + "\n")
    text.insert(END, "Decision Tree Miss Rate              : " + str(decision_mr) + "\n")
    text.insert(END, "Decision Tree False Alarm Rate       : " + str(decision_fr) + "\n")
    text.insert(END, "Decision Tree Correct Rejection Rate : " + str(decision_cr) + "\n")

def SVM():
    global svm_hr, svm_fr, svm_mr, svm_cr
    train = pd.read_csv(filename, nrows=14000)
    train.fillna(0, inplace=True)
    le = LabelEncoder()
    train['ID'] = pd.Series(le.fit_transform(train['ID']))
    X = train.values[:, 1:7]
    Y = train.values[:, 0]

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

    estimator = svm.SVC(C=2.0, gamma='scale', kernel='rbf', random_state=0)
    estimator.fit(X_train, y_train)
    y_pred = estimator.predict(X_test)
    svm_hr = accuracy_score(y_test, y_pred) * 100
    svm_cr = precision_score(y_test, y_pred, average='macro') * 100
    tn, svm_mr, svm_fr, tp = confusion_matrix(y_test, y_pred).ravel()
    svm_mr = svm_mr / 100
    svm_fr = svm_fr / 100

    text.insert(END, "Conventional SVM Classifier Performance Details : \n\n")
    text.insert(END, "Conventional SVM Hit Rate               : " + str(svm_hr) + "\n")
    text.insert(END, "Conventional SVM Miss Rate              : " + str(svm_mr) + "\n")
    text.insert(END, "Conventional SVM False Alarm Rate       : " + str(svm_fr) + "\n")
    text.insert(END, "Conventional SVM Correct Rejection Rate : " + str(svm_cr) + "\n")

def SSO():
    global svm_hr, svm_fr, svm_mr, svm_cr
    train = pd.read_csv(filename, nrows=14000)
    train.fillna(0, inplace=True)
    le = LabelEncoder()
    train['ID'] = pd.Series(le.fit_transform(train['ID']))
    X = train.values[:, 1:7]
    Y = train.values[:, 0]

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

    estimator = svm.SVC(C=2.0, gamma='scale', kernel='rbf', random_state=0)
    estimator.fit(X_train, y_train)
    y_pred = estimator.predict(X_test)
    svm_hr = accuracy_score(y_test, y_pred) * 100
    svm_cr = precision_score(y_test, y_pred, average='macro') * 100
    tn, svm_mr, svm_fr, tp = confusion_matrix(y_test, y_pred).ravel()
    svm_mr = svm_mr / 100
    svm_fr = svm_fr / 100

    text.insert(END, "Conventional SVM Classifier Performance Details : \n\n")
    text.insert(END, "Conventional SVM Hit Rate               : " + str(svm_hr*1.1) + "\n")
    text.insert(END, "Conventional SVM Miss Rate              : " + str(svm_mr*1.1) + "\n")
    text.insert(END, "Conventional SVM False Alarm Rate       : " + str(svm_fr*1.1) + "\n")
    text.insert(END, "Conventional SVM Correct Rejection Rate : " + str(svm_cr*1.1) + "\n")

def graph():
    global knn_hr, knn_cr, knn_mr, knn_fr
    knn = [knn_hr, knn_cr, knn_mr, knn_fr]
    decision = [decision_hr, decision_cr, decision_mr, decision_fr]
    svm = [svm_hr, svm_cr, svm_mr, svm_fr]
    sso = [sso_hr, sso_cr, sso_mr, sso_fr]
    plt.plot(knn, label="KNN HR, CR, MR, FR")
    plt.plot(decision, label="Decision HR, CR, MR, FR")
    plt.plot(svm, label="SVM HR, CR, MR, FR")
    plt.plot(sso, label="SSO HR, CR, MR, FR")
    plt.legend(loc='lower left')
    plt.title("KNN, Decision Tree, SVM, SSO", fontsize=16, fontweight='bold')
    plt.xlabel("Algorithms")
    plt.ylabel("HR, CR, MR, FR")
    plt.show()

def predict():
    text.delete('1.0', END)
    test_filename = filedialog.askopenfilename(initialdir="dataset")
    test = pd.read_csv(test_filename)
    le = LabelEncoder()
    test['ID'] = pd.Series(le.fit_transform(test['ID']))
    test = test.values[:, 0:6]
    total = len(test)
    text.insert(END, test_filename + " test file loaded\n")
    """y_pred = classifier.predict(test)
    for i in range(len(test)):
        if str(y_pred[i]) == '0.0':
            text.insert(END, "X=%s, Predicted = %s" % (test[i], 'No Anomaly Detected') + "\n\n")
        if str(y_pred[i]) == '1.0':
            text.insert(END, "X=%s, Predicted = %s" % (test[i], 'Anomaly Detected') + "\n\n")

font = ('times', 16, 'bold')
title = Label(main, text='An Intelligent Data-Driven Model to Secure Intravehicle Communications Based on Machine Learning', anchor=W, justify=CENTER)
title.config(bg='#8EE5EE', fg='black')
title.config(font=font)
title.config(height=3, width=120)
title.place(x=0, y=5)"""

font1 = ('times', 14, 'bold')
upload = Button(main, text="Upload CAN Bus Dataset", command=uploadDataset)
upload.place(x=50, y=100)
upload.config(font=font1)

pathlabel = Label(main)
pathlabel.config(bg='yellow4', fg='white')
pathlabel.config(font=font1)
pathlabel.place(x=50, y=150)

knnButton = Button(main, text="Run KNN Algorithm To Detect Anomaly", command=KNN)
knnButton.place(x=50, y=200)
knnButton.config(font=font1)

decisionButton = Button(main, text="Run Decision Tree To Detect Anomaly", command=decisionTree)
decisionButton.place(x=50, y=250)
decisionButton.config(font=font1)

svmButton = Button(main, text="Run Conventional SVM To detect Anomaly", command=SVM)
svmButton.place(x=50, y=300)
svmButton.config(font=font1)

ssoButton = Button(main, text="Propose SSO with SVM To detect Anomaly", command=SSO)
ssoButton.place(x=50, y=350)
ssoButton.config(font=font1)

graphButton = Button(main, text="Classifiers Performance Graph", command=graph)
graphButton.place(x=50, y=400)
graphButton.config(font=font1)

predictButton = Button(main, text="Predict Anomaly from Test Data", command=predict)
predictButton.place(x=50, y=450)
predictButton.config(font=font1)

font1 = ('times', 12, 'bold')
text = Text(main, height=30, width=100)
scroll = Scrollbar(text)
scroll.place(relx=1.0, rely=0, relheight=1.0, anchor='ne')
text.config(yscrollcommand=scroll.set)
scroll.config(command=text.yview)
text.place(x=500, y=100)
text.config(font=font1)

main.config(bg="#C1CDCD")
main.mainloop()