#importarea librariilor folosite
from sklearn import svm
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.metrics import mean_squared_error


#incarcarea datelor intr-un fisier general
# data = np.loadtxt("ticdata2000.txt")
#
#
# #incarcarea datelor in variabile si prelucrarea acestora
#
# people = data[:,:1]   #contine primul feature al fisierului(tipul de persoana)
#
# # print(people)
#
# insurances = data[:,44:-1] # contine tipurile de asigurari ale fiecarei instante(fiecarei persoane)
#
# X = np.append(people,insurances,axis = 1) #crearea variabilei
#
# y = data[:,-1] # crearea variabilei de target ( ultima coloana din fisier)
#
# data_eval = np.loadtxt("ticeval2000.txt") # incarcarea datelor pentru evaluare
#
# X_people = data_eval[:,:1]
#
# X_insurances = data_eval[:,44:]
#
# X_predictions = np.append(X_people,X_insurances, axis= 1) # reprezinta de fapt setul pentru testare
#
# y_eval = np.loadtxt("tictgts2000.txt") # variabila pentru predictii
#
# X_train,X_test,y_train,y_test = train_test_split(X,y,train_size=0.75,random_state= 42) # impartirea variabilelor in train si test
#
# cost =[0.03125, 0.125 , 0.5, 2 , 8 , 32 , 128 ]
#
# rezultate = []
#
# for i in cost:
#     model = svm.SVR(kernel='linear', C= i, gamma=1, epsilon=0.1)  # crearea modelului
#
#     model.fit(X_train, y_train)
#
#     predictions = model.predict(X_predictions)
#
#     MSE = mean_squared_error(y_eval, predictions)  # eroarea medie patratica ca masura a modelului
#
#     rezultate.append(MSE)
#
#     roc = roc_auc_score(y_eval, predictions)  # procentajul de distinctie/separare intre predictii
#
#     print("Pentru cost =",i ,"MSE este :", MSE, "si ROC este :",roc * 100)
#
# minim = min(rezultate)
#
# print("Cea mai mica eroare este :", minim)

import numpy as np
from sklearn import datasets
from sklearn.metrics import accuracy_score  # acuratete
from sklearn.metrics import mean_squared_error   # MSE(pentru regresie)
from sklearn import ensemble # pt random forest/bagging
from sklearn import neural_network # pt MLP
from sklearn import svm  #pt vectori suport


date,etichete=datasets.load_iris(return_X_y=True)
indices=np.random.permutation(150)
date=date[indices,:]
etichete=etichete[indices]

#matricea de date are 150 linii si 4 coloane
#matricea de etichete are 150 de valori care pot fi 0,1,2

X_train,X_test,y_train,y_test=train_test_split(date,etichete,test_size=0.4)

X_test= date[:60][:]
X_train=date[60:][:]
y_test=etichete[:60][:]
y_train=etichete[60:][:]

#pentru random forest
b = [25,50,85]
d =[10,50,80]

MSE = np.square(np.subtract(y_test,predictii)).mean()   # functie pentru MSE(NEIMPORTATA)

def acuratete(y_test, predictii):
    preziceri_corecte = 0
    for adevar , prezise in zip(y_test, predictii):
        if adevar == prezise:
            preziceri_corecte +=1

    acc = preziceri_corecte/ len(y_test)
    return acc
#pentru MLP
l = [0.0001, 0.1,1]
s =[10,50,200]
#pentru SVM
c = [1,0.1,0.01,10]

rezultate = []  # matrice de rezultate
for i in c:
    for j in s:

        # model = svm.SVR(kernel = 'rbf', C= i, gamma=1, epsilon=0.1)

        # model = neural_network.MLPClassifier(learning_rate_init=i, hidden_layer_sizes=j )


        # model = ensemble.BaggingClassifier(n_estimators= 10 , max_samples = i/100,
        #                                    # max_features= j/100)

        model.fit(X_train, y_train)

        predictii = model.predict(X_test)

        accurate = acuratete(y_test, predictii)

        print("pentru ....... =", accurate)
        rezultate.append(accurate)


max = 0
for i in rezultate:

    if  max < i:
        max = i


print(max)


































