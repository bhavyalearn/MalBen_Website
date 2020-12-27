# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns  # visualization libraries


## updated with Github path

data = pd.read_csv('https://raw.githubusercontent.com/bhavyalearn/MalBen_Website/main/Mal_ben_website_data.csv')

data.head()

data.info()


# before doing anything, rename all the columsn to smaller case
# Also 'url' column is of no use. Remove this column

data = data.rename(str.lower,axis='columns')

del data['url']

data.columns


#Find out null values in dataset
data.isna().sum()

## Dns_query_times has only one null value. Lets fix that first.
## easy way for this is to simply remove it
## but lets fix it, not remove it
plt.hist(data['dns_query_times'])


## fill this single data point with fillna method
data['dns_query_times'].fillna(value = data['dns_query_times'].mean(),inplace = True)
data.isna().sum()




##content_length also has nan values
#lets see how the values are distributed first
plt.hist(data['content_length'])



#lets see the type of url where content length is NaN
data[data['content_length'].isna()]['type'].value_counts()


type_zero_mean = data[ data['type'] == 0]['content_length'].mean()
type_one_mean = data[ data['type'] == 1]['content_length'].mean()


print('Benign mean', type_zero_mean)
print('Malicious mean', type_one_mean)


for i in range(len(data)):
    if (data.loc[i,'type'] == 0) & (pd.isna(data.loc[i,'content_length'])):
        data.loc[i,'content_length'] = type_zero_mean
    elif(data.loc[i,'type'] == 1) & (pd.isna(data.loc[i,'content_length'])):
        data.loc[i,'content_length'] = type_one_mean

print(data.isna().sum())

# drop the null value in 'server'
data = data.drop(data.index.values[1306])
data = data.reset_index()

data.isna().sum()


data.head()

## Add servernames in place of 'server' as there are too many server names in 'server' column

serverNamesList = ['apache','nginx','microsoft','openresty']
data['server_name'] =  ""

for i in range(len(data)):
    servername = str(data.loc[i,'server']).lower()
    for s in serverNamesList:  # go thru each server name from the list above
        if s in servername: # if servername is found in the list vs data
          data.loc[i,'server_name'] = s 
          #print("server list name", s, "data",data.loc[i,'server'])
          #print(data.loc[i,'server_name'])
          break # exit the inner loop if the name is found
        else:
          data.loc[i,'server_name'] = 'Other'

data.drop(['server'],axis=1, inplace = True)


plt.figure(figsize=(12,8))
sns.boxplot(x='server_name',y='content_length', hue= 'type', data = data)
plt.show()

# there are too many outliers. those will be removed before setting up the model part
## 19-12-2020

data['charset'].value_counts()





#charset column has multiple valuecounts due to different letter cases

#Need to rename values to a single case
#charsetList = ['none','windows-1251','windows-1252']

for i in range(len(data)):
    data.loc[i,'charset'] = str(data.loc[i,'charset']).lower()
    if (data.loc[i,'charset'] == 'none') | (data.loc[i,'charset'] == 'windows-1251') | (data.loc[i,'charset'] == 'windows-1252'):
        data.loc[i,'charset'] = 'Others'
    elif data.loc[i,'charset'] == 'iso-8859':
        data.loc[i,'charset'] = 'iso-8859-1'
    

data['charset'].value_counts()


data['whois_country'].value_counts()

## cleaniing for multiple GB entries

data['whois_country']= data['whois_country'].replace('United Kingdom','GB')
data['whois_country']= data['whois_country'].replace('[u\'GB\'; u\'UK\']','GB')
data['whois_country']= data['whois_country'].replace('UK','GB')
data['whois_country'].value_counts()



#replacing countries with less than 6 counts with others
# also covered 'none' country type 

country = data['whois_country'].value_counts()
smallCountries = list(country[country < 6].index) # make a list of countries with less than 6 counts
#print(smallCountries)

for i in range(len(data)):
  for c in smallCountries:
    if (c in data.loc[i,'whois_country']) or (data.loc[i,'whois_country'] == 'None'):
      data.loc[i,'whois_country'] = 'Others'
data['whois_country'].value_counts()


# count of rows where country is not mentioend but state is mentioned.
# ignoring this for now as the count is only 50
data[(data['whois_country']== 'Others') & (data['whois_statepro'] != 'None')].count()

#drop whois_statepro column for now

data=data.drop(['whois_statepro'],axis=1)


from datetime import datetime as dt


data[data['whois_regdate']=='None'].count()

len(np.where(data['whois_regdate']=='None')[0])

#droppping the dates columns for now

data = data.drop(data[['whois_regdate','whois_updated_date']],axis=1)
data.head()




plt.figure(figsize=(8,10))
sns.boxplot(x='server_name',y='number_special_characters',data = data,hue = 'type')
plt.show()


plt.figure(figsize=(12,10))
sns.boxplot(x='type',y='tcp_conversation_exchange',data = data)
plt.show()

plt.figure(figsize=(12,10))
sns.boxplot(x='type',y='dist_remote_tcp_port',data = data)
plt.show()


# prepare the data for modelling
# first step is to covert all relevant features into 1 and 0s

data = pd.get_dummies(data,prefix_sep='_',drop_first=True)

data.head()
data.drop('index',inplace = True,axis=1)


from sklearn.model_selection import train_test_split


x = data.drop('type',axis=1)
y = data['type']

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.20)
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)


print(x_train.shape[1]) ## 33 is the output


# first deep learning model
## remember, i have not removed the outliers at this time



from tensorflow.keras import models, layers



input_shape = x_train.shape[1]

dl_model = models.Sequential()

dl_model.add(layers.Flatten(input_shape = (input_shape,)))

for neuron_count in range(input_shape-1,3,-2):
    dl_model.add(layers.Dense(int(neuron_count), activation = 'relu'))

dl_model.add(layers.Dense(1,activation='sigmoid'))

dl_model.compile(loss='BinaryCrossentropy',optimizer='adam',metrics=['accuracy'])

dl_model.summary()

batch_size= 64

dl_trained_model = dl_model.fit(x_train,y_train,epochs=250,batch_size=batch_size)



# what is the maximum accuracy achieved? 
print(max(dl_trained_model.history['accuracy']))


#check if the model is predicting class 1 - malacious websites

dl_y_pred = dl_model.predict_classes(x_test)
print(np.max(dl_y_pred)) # maximum numbers from predicted y list
## Oh yes it is! :) 


#moment of truth

from sklearn.metrics import accuracy_score,f1_score,recall_score,precision_score

print("F1 Score: ", f1_score(y_test,dl_y_pred))
print("Accuracy: ", accuracy_score(y_test, dl_y_pred))
print('Recall:', recall_score(y_test, dl_y_pred))
print('Precision:', precision_score(y_test, dl_y_pred))


## interpretation of the above metrics to go here
## since I am trying to distinguish between malacious and benign websites, a higher precision is required.
## Recall helps when the cost of false negatives is high. if the model predicted a bad website as good one, its a potential security threat. 
## better get that double checked
## but what if model predictes a good website as bad one? it will block that website. probabely some complaint from user, some client dissatisfaction.
## Better than a disstatisfied client due to a virus attack

## precision is very low here at < 80 %
## need to work on improving this


















