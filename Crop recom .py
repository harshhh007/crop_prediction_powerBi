#!/usr/bin/env python
# coding: utf-8

# In[37]:


import pandas as pd
import numpy as np
import random
from IPython.core.display import update_display
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import seaborn as sns
import warnings
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.svm import SVC
from tkinter import *
from tkinter import ttk
from tkinter import messagebox  
from PIL import ImageTk, Image
warnings.filterwarnings('ignore')


# In[2]:


data=pd.read_csv('crop_recommendation.csv')


# In[3]:


data.head(5) 


# In[4]:


data.tail(5)


# In[5]:


data.shape


# In[6]:


data.columns


# In[7]:


data.duplicated().sum()


# In[8]:


data.isnull().sum()


# In[9]:


data.info


# In[10]:


data.describe()


# In[11]:


data.nunique()


# In[12]:


data['label'].unique()


# In[13]:


data['label'].value_counts()


# In[14]:


crop_summary=pd.pivot_table(data,index=['label'],aggfunc='mean')


# In[15]:


crop_summary


# In[16]:


data.columns


# # BOXPLOT

# In[18]:


#checking and treating outliers in each column

fig =px.box(data,y='N',points='all',title="Boxplot of N")
fig.show()


# In[19]:


fig= px.box(data, y="P",points="all",title="Boxplot of P")
# sns.boxplot(data["P"])
# plt.xticks(rotation=90)
fig.show()


# In[20]:


fig= px.box(data, y="K",points="all",title="Boxplot of K")
fig.show()


# In[21]:


fig= px.box(data, y="temperature",points="all",title="Boxplot of temperature")
fig.show()


# In[22]:


fig= px.box(data, y="humidity",points="all",title="Boxplot of humidity")
fig.show()

#boxplot of humidity means that the humidity is between 0.5 and 0.7 for most of the time 
#and there are some outliers which are above 0.7 and below 0.5 which are very few in number


# In[23]:


data.columns


# In[24]:


fig= px.box(data, y="ph",points="all",title="Boxplot of ph")
fig.show()

#boxplot of ph means that the ph of the water is between 6.5 and 8.5 and the median is 7.5 and 
# the outliers are 0 and 14 which are not possible values for ph of water  


# In[25]:


fig= px.box(data, y="rainfall",points="all",title="Boxplot of rainfall")
fig.show()


# In[26]:


#Detection & removal of outliers

df_boston = data
df_boston.columns = df_boston.columns
df_boston.head()

#Detection of Outliers
#IQR = Q3 - Q1
Q1=np.percentile(df_boston['rainfall'],25,interpolation='midpoint') # type: ignore
Q3=np.percentile(df_boston['rainfall'],75,interpolation='midpoint') # type: ignore
IQR=Q3-Q1

print("Old Shape: ", df_boston.shape)

# Upper bound
upper = np.where(df_boston['rainfall'] >= (Q3+1.5*IQR))

# Lower bound
lower = np.where(df_boston['rainfall'] <= (Q1-1.5*IQR))

#Removing the Outlier
df_boston.drop(upper[0], inplace=True)
df_boston.drop(lower[0], inplace=True)


print("New Shape: ", df_boston.shape)


# In[27]:


data=df_boston


# In[28]:


plt.figure(figsize=(15,6))
sns.barplot(y='N',x='label',data=data,palette='hls')
plt.xticks(rotation=90)
plt.show()

#This bar plot shows that the nitrogen content is highest in cotton and lowest in lentil


# In[29]:


plt.figure(figsize=(15,6))
sns.barplot(y='P',x='label',data=data,palette='hls')
plt.xticks(rotation=90)
plt.show()

#This bar plot shows that the phosphorous content is highest in apple and lowest in watermelon.


# In[30]:


plt.figure(figsize=(15,6))
sns.barplot(y='K',x='label',data=data,palette='hls')
plt.xticks(rotation=90)
plt.show()

#This bar plot shows that the potassium content is highest in grapes and lowest in orange.


# In[31]:


crop_summary_new=data.copy()

#we used a variable crop_summary_new to store the data of crop_summary and then we used the variable crop_summary_new to plot the graph
#because if we use crop_summary to plot the graph then the graph will be plotted in the order of the index of crop_summary which is label
#and the order of the index of crop_summary is alphabetical order and we want the graph to be plotted in the order of the yield of the crops
#so we used crop_summary_new to plot the graph


# In[33]:


fig1=px.bar(crop_summary_new,x='label',y='N')
fig1.show()

#this shows that the nitrogen content is highest in cotton and lowest in coconut


# In[34]:


fig1=px.bar(crop_summary_new,x='label',y='K')
fig1.show()

#this shows that the crop which requires more nitrogen also requires more potassium 


# In[35]:


fig1=px.bar(crop_summary_new,x='label',y='P')
fig1.show()

#this shows that the crops which require more nitrogen also require more phosphorous and potassium.


# # CORRELATION

# In[86]:


# Visualization and data exploration
fig, ax = plt.subplots(1, 1, figsize=(6, 6))

# Exclude non-numeric column before creating correlation matrix
numeric_data = data.drop('label', axis=1)
sns.heatmap(numeric_data.corr(), annot=True, cmap='Wistia')

ax.set(xlabel='Features')
ax.set(ylabel='Features')
plt.title('Correlation between features', fontsize=15, c='blue')
plt.show()


# In[55]:


X=data.drop('label',axis=1)
y=data['label']


# In[56]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.30,shuffle=True,random_state=0)


# In[61]:


Classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
Classifier.fit(X_train, y_train)

y_pred_decisiontree=Classifier.predict(X_test)

from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred_decisiontree)
print('decision tree model accuracy score: {0:0.4f}'.format(accuracy_score(y_test,y_pred_decisiontree)))

#classification report
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred_decisiontree))

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred_decisiontree)

plt.figure(figsize=(10,10))
sns.heatmap(cm, annot=True, fmt=".0f", linewidths=.5, square = True, cmap = 'plasma');
plt.ylabel('Actual label');
plt.xlabel('Predicted label');
all_sample_title='Confusion Matrix -score'+str(accuracy_score(y_test, y_pred_decisiontree))
plt.title(all_sample_title, size = 15);
plt.show()

#decision tree is used to predict the yield of the crops based on the input given by the user like the nitrogen content,phosphorous content,potassium content,temperature,humidity,rainfall and ph of the water and the output is the name of the crop which has the highest yield based on the input given by the user.


# In[59]:


classifier_lr = LogisticRegression(random_state = 0)
classifier_lr.fit(X_train, y_train)

y_pred_lr=classifier_lr.predict(X_test)

from sklearn.metrics import accuracy_score
accuracy= accuracy_score(y_test, y_pred_lr)
print('Logistic Regression Model accuracy score: {0:0.4f}'.format(accuracy_score(y_test, y_pred_lr)))

from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred_lr))

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred_lr)

plt.figure(figsize=(10,10))
sns.heatmap(cm, annot=True, fmt=".0f", linewidths=.5, square = True, cmap = 'civi');
plt.ylabel('Actual label');
plt.xlabel('Predicted label');
all_sample_title='Confusion Matrix -score'+str(accuracy_score(y_test, y_pred_lr))
plt.title(all_sample_title, size = 15);
plt.show()


# In[62]:


#random forest model
classifier_rf = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier_rf.fit(X_train, y_train)

y_pred_rf=classifier_rf.predict(X_test)

from sklearn.metrics import accuracy_score
accuracy= accuracy_score(y_test, y_pred_rf)
print('Random Forest Model accuracy score: {0:0.4f}'.format(accuracy_score(y_test, y_pred_rf)))

from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred_rf))

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred_rf)

plt.figure(figsize=(10,10))
sns.heatmap(cm, annot=True, fmt=".0f", linewidths=.5, square = True, cmap = 'magma');
plt.ylabel('Actual label');
plt.xlabel('Predicted label');
all_sample_title='Confusion Matrix -score'+str(accuracy_score(y_test, y_pred_rf))
plt.title(all_sample_title, size = 15);
plt.show()


# In[63]:


#svm model
classifier_svm = SVC(kernel = 'linear', random_state = 0)
classifier_svm.fit(X_train, y_train)

y_pred_svm=classifier_svm.predict(X_test)

from sklearn.metrics import accuracy_score
accuracy= accuracy_score(y_test, y_pred_svm)
print('SVM Model accuracy score: {0:0.4f}'.format(accuracy_score(y_test, y_pred_svm)))

from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred_svm))

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred_svm)

plt.figure(figsize=(10,10))
sns.heatmap(cm, annot=True, fmt=".0f", linewidths=.5, square = True, cmap = 'mako');
plt.ylabel('Actual label');
plt.xlabel('Predicted label');
all_sample_title='Confusion Matrix -score'+str(accuracy_score(y_test, y_pred_svm))
plt.title(all_sample_title, size = 15);
plt.show()


# In[64]:


#Designing a hybrid model using LR and decision tree classifier
# Create sub models
estimators = []
model1 = LogisticRegression()
estimators.append(('logistic', model1))
model2 = DecisionTreeClassifier()
estimators.append(('cart', model2))

# Create the ensemble model
ensemble = VotingClassifier(estimators)
ensemble.fit(X_train, y_train)
y_pred_hybrid = ensemble.predict(X_test)
print("Accuracy score of ensemble model is:",accuracy_score(y_test, y_pred_hybrid))

#classification report
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred_hybrid))

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred_hybrid)

plt.figure(figsize=(10,10))
sns.heatmap(cm, annot=True, fmt=".0f", linewidths=.5, square = True, cmap = 'plasma');
plt.ylabel('Actual label');
plt.xlabel('Predicted label');
all_sample_title='Confusion Matrix -score'+str(accuracy_score(y_test, y_pred_hybrid))
plt.title(all_sample_title, size = 15);
plt.show()


# ## All models are used for the prediction of the values of the dataset , such as 

# In[65]:


#now design bar plot for accuracy score of models used above
models = ['LR', 'Decision Tree', 'Ensemble','RF','SVM']
accuracy = [accuracy_score(y_test, y_pred_lr),accuracy_score(y_test, y_pred_decisiontree),accuracy_score(y_test, y_pred_hybrid),accuracy_score(y_test, y_pred_rf),accuracy_score(y_test, y_pred_svm)]

#make different color for each model
colors = ['red', 'green', 'blue', 'yellow','orange']
plt.bar(models,accuracy,color=colors)
plt.xlabel('Models',color='pink',fontsize=15,fontweight='bold',horizontalalignment='center',fontname='Times New Roman')
plt.ylabel('Accuracy',color='pink',fontsize=15,fontweight='bold',horizontalalignment='center',fontname='Times New Roman')
plt.title('Accuracy of models')
plt.show()


# In[66]:


X_test[0:1]


# In[67]:


result=Classifier.predict(X_test[0:1])


# In[68]:


result


# In[69]:


y_test[0:1]


# In[84]:


import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px

# Initialize the Dash app
app = dash.Dash(__name__)

# Layout of the dashboard
app.layout = html.Div([
    html.H1("Crop Recommendation Dashboard"),

    # Bar plot of accuracy scores
    dcc.Graph(
        id='accuracy-bar-plot',
        figure={
            'data': [
                {'x': ['LR', 'Decision Tree', 'Ensemble', 'RF', 'SVM'],
                 'y': [accuracy_score(y_test, y_pred_lr),
                       accuracy_score(y_test, y_pred_decisiontree),
                       accuracy_score(y_test, y_pred_hybrid),
                       accuracy_score(y_test, y_pred_rf),
                       accuracy_score(y_test, y_pred_svm)],
                 'type': 'bar',
                 'name': 'Accuracy',
                 'marker': {'color': ['red', 'green', 'blue', 'yellow', 'orange']}
                 }
            ],
            'layout': {
                'title': 'Accuracy of Models',
                'xaxis': {'title': 'Models'},
                'yaxis': {'title': 'Accuracy'},
            }
        }
    ),

    # Scatter plot for N content by crop
    dcc.Graph(
        id='scatter-plot-N',
        figure=px.scatter(crop_summary_new, x='label', y='N', title='Nitrogen Content by Crop')
    ),

    # Box plot for temperature
    dcc.Graph(
        id='box-plot-temperature',
        figure=px.box(data, y='temperature', points='all', title='Boxplot of Temperature')
    ),

    # Add more visualizations as needed

])

if __name__ == '__main__':
    app.run_server(debug=True, port=8053)


# In[ ]:





# In[ ]:




