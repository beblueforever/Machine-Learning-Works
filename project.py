#!/usr/bin/env python
# coding: utf-8

# ## Kredi Kartı Dolandırıcılık Tespiti Projesi

# In[17]:


# Gerekli modülleri içe aktarın

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import itertools

#for ann 
import warnings
from sklearn.neural_network import MLPClassifier
from sklearn.exceptions import ConvergenceWarning

# for pca
from scipy import stats
from sklearn.decomposition import PCA


#Load libraries
import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV, KFold

import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")
# unused but required import for doing 3d projections with matplotlib < 3.2
import mpl_toolkits.mplot3d  # noqa: F401
 
#######
from matplotlib.colors import ListedColormap

from sklearn.linear_model import LogisticRegression 
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score, KFold
import matplotlib.pyplot as plt
#from imblearn import under_sampling, over_sampling
#from imblearn.over_sampling import SMOTE

# manual nested cross-validation for random forest on a classification dataset
from numpy import mean
from numpy import std
from sklearn.datasets import make_classification
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix, recall_score, f1_score


# In[2]:


# csv dosyasını yükleyin

dataframe = pd.read_csv("creditcard.csv")
dataframe.head()


# ### Keşifsel Veri Analizi Gerçekleştirin

# In[3]:


dataframe.info()


# In[4]:


# Boş değerler olup olmadığını kontrol edin

dataframe.isnull().values.any()


# In[5]:


dataframe["Amount"].describe()


# In[6]:


non_fraud = len(dataframe[dataframe.Class == 0])
fraud = len(dataframe[dataframe.Class == 1])
fraud_percent = (fraud / (fraud + non_fraud)) * 100

print("Gerçek işlem sayısı: ", non_fraud)
print("Dolandırıcılık işlemlerinin sayısı: ", fraud)
print("Dolandırıcılık işlemlerinin yüzdesi: {:.4f}".format(fraud_percent))


# In[7]:


# Veri kümemizde "Etiketler" sütununu görselleştirin

labels = ["Genuine", "Fraud"]
count_classes = dataframe.value_counts(dataframe['Class'], sort= True)
count_classes.plot(kind = "bar", rot = 0)
plt.title("Visualization of Labels")
plt.ylabel("Count")
plt.xticks(range(2), labels)
plt.show()


# In[8]:


# Ölçeklendirmeyi Gerçekleştirin

scaler = StandardScaler()
dataframe["NormalizedAmount"] = scaler.fit_transform(dataframe["Amount"].values.reshape(-1, 1))
dataframe.drop(["Amount", "Time"], inplace= True, axis= 1) 

Y = dataframe["Class"]
X = dataframe.drop(["Class"], axis= 1)


# In[9]:


Y.head()


# In[10]:


# verileri böldük ve train ile testin şekillerini yazdırdık.

(train_X, test_X, train_Y, test_Y) = train_test_split(X, Y, test_size= 0.33, random_state= 42)

print("Train_X'in şekli: ", train_X.shape)
print("test_X şekli: ", test_X.shape)


# Let's train different models on our dataset and observe which algorithm works better for our problem.
# 
# Let's apply Random Forests and Decision Trees algorithms to our dataset.


# In[11]:

#ann classifier 
    
clf = MLPClassifier()

clf.fit(train_X, train_Y)

#clf.predict_proba(test_X,test_Y)
#clf.predict(test_X)




predictions_cfl = clf.predict(test_X)
clf_score = clf.score(test_X, test_Y)* 100  #score yazdırıyoruz. 
print(clf.score(test_X, test_Y)) 

"""
# In[12]:

    # Nested cross validation
 
    #Set a seed to ensure reproducibility
seed = 42

#Instantiate the Random Forest classifier
rf = RandomForestClassifier(random_state=seed)

#Number of rounds
rounds = 20
#Define the hyperparameter grid
rf_param_grid = {'max_depth': [10, 50],
                'n_estimators': [100, 200, 400]}

#Create arrays to store the scores
outer_scores = np.zeros(rounds)
nested_scores = np.zeros(rounds)
# Loop for each round
for i in range(rounds):

   #Define both cross-validation objects (inner & outer)
   inner_cv = StratifiedKFold(n_splits=3000, shuffle=True, random_state=i)
   outer_cv = StratifiedKFold(n_splits=3000, shuffle=True, random_state=i)

   # Non-nested parameter search and scoring
   clf = GridSearchCV(estimator=rf, param_grid=rf_param_grid, cv=inner_cv)
   clf.fit(train_X, train_Y)
   outer_scores[i] = clf.best_score_

   # Nested CV with parameter optimization
   nested_score = cross_val_score(clf, X=test_X, y=test_Y, cv=outer_cv)
   nested_scores[i] = nested_score.mean()
   #Take the difference from the non-nested and nested scores
score_difference = outer_scores - nested_scores

print("Avg. difference of {:6f} with std. dev. of {:6f}."
      .format(score_difference.mean(), score_difference.std()))
# Plot scores on each round for nested and non-nested cross-validation
plt.style.use('seaborn')
plt.tight_layout()
plt.figure(figsize=(10,5))
outer_scores_line, = plt.plot(outer_scores, color='orange')
nested_line, = plt.plot(nested_scores, color='steelblue')
plt.ylabel("Score", fontsize="14")
plt.legend([outer_scores_line, nested_line],
          ["Non-Nested CV", "Nested CV"],
          bbox_to_anchor=(0, .4, .5, 0))
plt.title("Non-Nested vs Nested Cross-Validation on the Wine Dataset",
         x=.5, y=1.1, fontsize="15")

# Plot bar chart of the difference.
plt.figure(figsize=(10,5))
plt.tight_layout()
difference_plot = plt.bar(range(rounds), score_difference)
plt.xlabel("Individual Trial #")
plt.legend([difference_plot],
          ["Non-Nested CV - Nested CV Score"],
          bbox_to_anchor=(0, 1, .8, 0))
plt.ylabel("score difference", fontsize="14")
plt.show()
"""
    
# In[11]:

# Decision Tree Classifier
decision_tree = DecisionTreeClassifier()
decision_tree.fit(train_X, train_Y)

predictions_dt = decision_tree.predict(test_X)
decision_tree_score = decision_tree.score(test_X, test_Y) * 100


# In[11]:

# PCA
sc = StandardScaler()
 
train_X = sc.fit_transform(train_X)
test_X = sc.transform(test_X)


pca = PCA(n_components = 2)
 
train_X = pca.fit_transform(train_X)
test_X = pca.transform(test_X)
explained_variance = pca.explained_variance_ratio_

classifier = LogisticRegression(random_state = 0)
classifier.fit(train_X, train_Y)
 
y_pred = classifier.predict(test_X)

 
cm = confusion_matrix(test_Y, y_pred)

X_set, y_set = train_X, train_Y
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1,
                     stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1,
                     stop = X_set[:, 1].max() + 1, step = 0.01))
 
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(),
             X2.ravel()]).T).reshape(X1.shape), alpha = 0.75,
             cmap = ListedColormap(('yellow', 'white', 'aquamarine')))
 
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
 
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green', 'blue'))(i), label = j)
 
plt.title(' PCA With Logistic Regression')
plt.xlabel('versiyon') # for Xlabel
plt.ylabel('harcamalar') # for Ylabel
plt.legend() # to show legend
 
# show scatter plot
plt.show()




    
# In[12]:


# Random Forest
random_forest = RandomForestClassifier(n_estimators= 100)
random_forest.fit(train_X, train_Y)

predictions_rf = random_forest.predict(test_X)
random_forest_score = random_forest.score(test_X, test_Y) * 100


# In[13]:


# Print scores of our classifiers

print("Random Forest Score: ", random_forest_score)
print("Decision Tree Score: ", decision_tree_score)
print("ANN Score: ", clf_score)


# In[14]:


# Aşağıdaki işlev, karışıklık matrisini çizmek için doğrudan scikit-learn web sitesinden alınmıştır.

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion Matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=0)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()




# In[22]:


# Plot confusion matrix for Ann

confusion_matrix_ann = confusion_matrix(test_Y, predictions_cfl.round())
print("Confusion Matrix -  Neural network")
print(confusion_matrix_ann)


# In[18]:


plot_confusion_matrix(confusion_matrix_ann, classes=[0, 1], title= "Confusion Matrix - Neural network")

    
# In[22]:


# Plot confusion matrix for Decision Trees

confusion_matrix_dt = confusion_matrix(test_Y, predictions_dt.round())
print("Confusion Matrix - Decision Tree")
print(confusion_matrix_dt)


# In[18]:


plot_confusion_matrix(confusion_matrix_dt, classes=[0, 1], title= "Confusion Matrix - Decision Tree")


# In[20]:


# Plot confusion matrix for Random Forests

confusion_matrix_rf = confusion_matrix(test_Y, predictions_rf.round())
print("Confusion Matrix - Random Forest")
print(confusion_matrix_rf)


# In[23]:


plot_confusion_matrix(confusion_matrix_rf, classes=[0, 1], title= "Confusion Matrix - Random Forest")


# In[24]:


# The below function prints the following necesary metrics

def metrics(actuals, predictions):
    print("Accuracy: {:.5f}".format(accuracy_score(actuals, predictions)))
    print("Precision: {:.5f}".format(precision_score(actuals, predictions)))
    print("Recall: {:.5f}".format(recall_score(actuals, predictions)))
    print("F1-score: {:.5f}".format(f1_score(actuals, predictions)))
    



# In[25]:

print()
print("Evaluation of Neural network Model")
print()
metrics(test_Y, predictions_cfl.round())


# In[26]:
    
    
# In[25]:

print()
print("Evaluation of Decision Tree Model")
print()
metrics(test_Y, predictions_dt.round())


# In[26]:

print()
print("Evaluation of Random Forest Model")
print()
metrics(test_Y, predictions_rf.round())


# Açıkçası, Rastgele Orman modeli Karar Ağaçlarından daha iyi çalışıyor

# Ancak, veri setimizin ciddi bir **sınıf dengesizliği** sorunu olduğunu açıkça gözlemlersek.
# Gerçek (dolandırıcılık olmayan) işlemler %99'un üzerinde olup, dolandırıcılık işlemleri %0.17'yi oluşturmaktadır.
#
# Bu tür bir dağılımla, modelimizi dengesizlik sorunlarına aldırmadan eğitirsek, orijinal işlemlere verilen önemin daha yüksek olduğu etiketi (bunlar hakkında daha fazla veri olduğu için) tahmin eder ve dolayısıyla daha fazla doğruluk elde eder.

# Sınıf dengesizliği sorunu çeşitli tekniklerle çözülebilir. **Aşırı örnekleme** bunlardan biridir.
#
# Dengesiz veri kümelerini ele almaya yönelik bir yaklaşım, azınlık sınıfını aşırı örneklemektir. En basit yaklaşım, azınlık sınıfındaki örneklerin çoğaltılmasını içerir, ancak bu örnekler modele herhangi bir yeni bilgi eklemez.
#
# Bunun yerine mevcut örneklerden yeni örnekler sentezlenebilir. Bu, azınlık sınıfı için bir tür veri büyütmedir ve **Sentetik Azınlık Aşırı Örnekleme Tekniği** veya kısaca **SMOTE** olarak adlandırılır.
# In[27]:


# Performing oversampling on RF and DT

from imblearn.over_sampling import SMOTE

X_resampled, Y_resampled = SMOTE().fit_resample(X, Y)
print("Resampled shape of X: ", X_resampled.shape)
print("Resampled shape of Y: ", Y_resampled.shape)

value_counts = Counter(Y_resampled)
print(value_counts)

(train_X, test_X, train_Y, test_Y) = train_test_split(X_resampled, Y_resampled, test_size= 0.3, random_state= 42)


# In[35]:


# Build the Random Forest classifier on the new dataset

rf_resampled = RandomForestClassifier(n_estimators = 100)
rf_resampled.fit(train_X, train_Y)

predictions_resampled = rf_resampled.predict(test_X)
random_forest_score_resampled = rf_resampled.score(test_X, test_Y) * 100


# In[36]:


# Visualize the confusion matrix

cm_resampled = confusion_matrix(test_Y, predictions_resampled.round())
print("Confusion Matrix - Random Forest")
print(cm_resampled)


# In[37]:


plot_confusion_matrix(cm_resampled, classes=[0, 1], title= "Confusion Matrix - Random Forest After Oversampling")


# In[38]:


print("Evaluation of Random Forest Model")
print()
print("Hello")
metrics(test_Y, predictions_resampled.round())

"""

# In[35]:


# Build the Decision Tree classifier on the new dataset

#dt_resampled = DecisionTreeClassifier(b_estimators = 100)
dt_resampled = DecisionTreeClassifier()
dt_resampled.fit(train_X, train_Y)

predictions_resampled = dt_resampled.predict(test_X)
decision_tree_score_resampled = dt_resampled.score(test_X, test_Y) * 100



# In[36]:


# Visualize the confusion matrix

cm_resampled = confusion_matrix(test_Y, predictions_resampled.round())
print("Confusion Matrix -Decision Tree")
print(cm_resampled)


# In[37]:


plot_confusion_matrix(cm_resampled, classes=[0, 1], title= "Confusion Matrix -Decision Tree After Oversampling")


# In[38]:


print("Evaluation of Decision Tree Model")
print()
metrics(test_Y, predictions_resampled.round())



"""

#Sınıf dengesizliği sorununu ele aldıktan sonra, SMOTE'lu Rastgele orman sınıflandırıcımızın SMOTE'lu Rastgele orman sınıflandırıcısından çok daha iyi performans gösterdiği artık açıktır.
# In[39 ]:

# create dataset
# X, y = make_classification(n_samples=1000, n_features=20, random_state=1, n_informative=10, n_redundant=10)
# configure the cross-validation procedure
# cv_outer = KFold(n_splits=10, shuffle=True, random_state=1)
# enumerate splits
# outer_results = list()
# for train_ix, test_ix in cv_outer.split(X):
	# split data
# 	X_train, X_test = X[train_ix, :], X[test_ix, :]
# 	y_train, y_test = y[train_ix], y[test_ix]
# 	# configure the cross-validation procedure
# 	cv_inner = KFold(n_splits=3, shuffle=True, random_state=1)
# 	# define the model
# 	model = RandomForestClassifier(random_state=1)
	# define search space
# 	space = dict()
# 	space['n_estimators'] = [10, 100, 500]
# 	space['max_features'] = [2, 4, 6]
	# define search
# 	search = GridSearchCV(model, space, scoring='accuracy', cv=cv_inner, refit=True)
	# execute search
# 	result = search.fit(X_train, y_train)
# 	# get the best performing model fit on the whole training set
# 	best_model = result.best_estimator_
# 	# evaluate model on the hold out dataset
# 	yhat = best_model.predict(X_test)
# 	# evaluate the model
# 	acc = accuracy_score(y_test, yhat)
# 	# store the result
# 	outer_results.append(acc)
	# report progress
# 	print('>acc=%.3f, est=%.3f, cfg=%s' % (acc, result.best_score_, result.best_params_))
# summarize the estimated performance of the model
# print('Accuracy: %.3f (%.3f)' % (mean(outer_results), std(outer_results)))






