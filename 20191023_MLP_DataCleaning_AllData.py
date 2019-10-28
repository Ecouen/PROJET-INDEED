#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re


# In[2]:


df = pd.read_csv("/Users/yinghua/Downloads/indeed_unclean.csv")


# # Preprocessing

# In[3]:


###Remove duplicated by ID, by description

df['ID'] = df['id'].apply(lambda x: x[-16:]) #Ceate a new column ID
df.drop_duplicates(subset ='ID', keep = 'first', inplace=True) #Delete duplicated by ID
df.drop_duplicates(subset = 'description', keep = 'first', inplace=True) #Delete duplicated by description 
df.drop(df.columns[0], axis=1, inplace=True) #Delete first column "unnamed"


# In[4]:


#Convert df to lower case. Easy to search and replace

df = df.applymap(lambda x : x.lower() if type(x) == str else x)


# In[5]:


df.shape


# In[6]:


df.isnull().sum()


# In[7]:


df.info()


# # Split the df into 2 parts: one with salary (s), one without salary(ns) 

# In[8]:


df.head()


# In[9]:


s = df[df['salaire'].isnull() == False]
ns = df[df['salaire'].isnull() == True]


# In[10]:


print(s.shape, ns.shape)


# In[11]:


s.isnull().sum()


# In[12]:


###Delete rows with NaN
s.dropna(inplace=True)

###Reset index after delete rows 
s.reset_index(drop=True, inplace=True)


# In[13]:


s.shape


# In[14]:


s.isnull().sum()


# # Create new column - Contract
# 
# ##pb: argument of type 'float' is not iterable 
# ##why: NaN in the col of type 'float'
# ##solution: convert float to string. (Cannot apply string method to float type).
# #col = df['contrat']
# #t = str
# #for idx, i in enumerate(col):
# #if(type(i) != t):
# #print(idx, type(i))

# In[15]:


####Column of contract  
s['Contract'] = ''
##According to the reuslt of df['contrat'].value_counts(), there are 7 types of contract in the dataset
##CDI, CDD, contrat pro, intérim, freelance/indépendant, apprentissage, stage

for i in range(s.shape[0]):
    if 'cdi' in s['meta1'][i] or 'cdi' in s['meta2'][i] or 'cdi' in s['contrat'][i] or 'cdi' in s['description']:
        s['Contract'][i] = 'cdi'
    elif 'cdd' in s['meta1'][i] or 'cdd' in s['meta2'][i] or 'cdd' in s['contrat'][i] or 'cdd' in s['description']:
        s['Contract'][i] = 'cdd'
    elif 'contrat pro' in s['meta1'][i] or 'contrat pro' in s['meta2'][i] or 'contrat pro' in s['contrat'][i]    or 'contrat pro' in s['description']:
        s['Contract'][i] = 'contrat pro'
    elif 'intérim' in s['meta1'][i] or 'intérim' in s['meta2'][i] or 'intérim' in s['contrat'][i]     or 'intérim' in s['description']:
        s['Contract'][i] = 'intérim'
    elif 'freelance / indépendant' in s['meta1'][i] or 'freelance / indépendant' in s['meta2'][i] or     'freelance / indépendant' in s['contrat'][i] or 'freelance / indépendant' in s['description']:
        s['Contract'][i] = 'freelance / indépendant'
    elif 'apprentissage' in s['meta1'][i] or 'apprentissage' in s['meta2'][i] or 'apprentissage' in s['contrat'][i]    or 'apprentissage' in s['description']:
        s['Contract'][i] = 'apprentissage'
    elif 'stage' in s['meta1'][i] or 'stage' in s['meta2'][i] or 'stage' in s['contrat'][i]     or 'stage' in s['description']:
        s['Contract'][i] = 'stage'
    else:
        pass


# In[16]:


s['Contract'].value_counts()  #Values before replacing and dropping


# In[17]:


##Delete rows of Contract == stage / apprentissage 
s = s[(s['Contract']!='stage') & (s['Contract']!='apprentissage')]

##Replace missing values by cdi 
s['Contract'] = s['Contract'].replace('', 'cdi')


# In[18]:


s['Contract'].value_counts()


# In[19]:


1921/s.shape[0]


# # Create new column - Location

# In[20]:


###Extract data from 'localisation'
s['Location'] = s['localisation'].str.split(" ").str[-1].str.replace("(", "").str.replace(")", "")

# Use df['department'].value_counts() to check exceptions and solve these problems 
s['Location'] = s['Location'].replace("hauts-de-seine", "92")
s['Location'] = s['Location'].replace(["île-de-france", "france", "paris"], "75")
s['Location'] = s['Location'].replace("rhône", "69")
s['Location'] = s['Location'].replace("occitanie", "31") #occitanie is replaced by Toulouse Métropole
s['Location'] = s['Location'].replace("nouvelle-aquitaine","33") #nouvelle-aquitaine is replaced by Bordeaux Métropole
s['Location'] = s['Location'].replace("loire", "42")
s['Location'] = s['Location'].replace("haute-garonne", "31")
s['Location'] = s['Location'].replace("seine-saint-denis", "93")
s['Location'] = s['Location'].replace("essonne", "91")
s['Location'] = s['Location'].replace("charente-maritime", "17")
s['Location'] = s['Location'].replace("val-d'oise", "95")
s['Location'] = s['Location'].replace("vendée", "85")
s['Location'] = s['Location'].replace("loire-atlantique", "44")
s['Location'] = s['Location'].replace("val-de-marne", "94")
s['Location'] = s['Location'].replace("seine-et-marne", "77")
s['Location'] = s['Location'].replace("gers", "32")
s['Location'] = s['Location'].replace("auvergne-rhône-alpes", "69") #auvergne-rhone-alpes is replaced by Grand Lyon 
s['Location'] = s['Location'].replace("yvelines", "78")
s['Location'] = s['Location'].replace("gironde", "33")
s['Location'] = s['Location'].replace("ain", "01")
s['Location'] = s['Location'].replace("oise", "60")
s['Location'] = s['Location'].replace("maine-et-loire", "49")
s['Location'] = s['Location'].replace("eure-et-loir", "28")
s['Location'] = s['Location'].apply(lambda x: '75' if x in ('77','78','91','92','93','94','95') else x) #Grand Paris

###Delete rows that donot match the 5 locations 
#cities = ['75','69','33','44','31']
#location out of the range : temp = s[~s['Location'].isin(cities)], more than 200 rows

cities = ['75','69','33','44','31']
s = s[s['Location'].isin(cities)]  


# In[21]:


s['Location'].value_counts()


# In[22]:


s.shape


# In[23]:


###Reset index after dropping rows
s.reset_index(drop=True, inplace=True)


# # Create 3 new columns regarding salary: salary type, salary with range, salary(mean)

# In[24]:


####Salary column 
##Inconsistence bewteen annual/monthly/weekly/daily salary. 
##If we convert daily salary into annual salary, the salary would be really high 
##The max salary in the dataset is thus a job paid by day
##This increases the bias of the prediction
##Its better seperate annual salary with the others 

###Solutuion: create 3 columns
#Salary_type (y for year, m for month, w for week, d for day) 
#Salary_clean (keep the range)
#Salary (replace the range by mean)

s['Salary_type'] = ''
s['Salary_clean'] = ''
s['Salary'] = ''

s['salaire'] = s['salaire'].apply(lambda x: x.replace(" ", "")) #Delete all spaces 
s['salaire'] = s['salaire'].apply(lambda x: x.replace(",", ".")) #cannot calculate the mean with , in the number

for i in range(s.shape[0]):
    if 'paran' in s['salaire'][i]:
        s['Salary_type'][i] = 'y'
        s['Salary_clean'][i] = s['salaire'][i][:-5]
        s['Salary_clean'][i] = s['Salary_clean'][i].replace('€', '') #Delete 'paran' et €, eg.50000 or 50000-60000
        if "-" in s['Salary_clean'][i]:
            s['Salary'][i] = s['Salary_clean'][i].split("-")
            s['Salary'][i] = (float(s['Salary'][i][0]) + float(s['Salary'][i][1]))/2
        else:
            s['Salary'][i] = float(s['Salary_clean'][i])  #str to int
            
    elif 'parmois' in s['salaire'][i]:  #Monthly salary has not been converted to annual salary 
        s['Salary_type'][i] = 'm'
        s['Salary_clean'][i] = s['salaire'][i][:-7].replace('€', '') 
        if "-" in s['Salary_clean'][i]:
            s['Salary'][i] = s['Salary_clean'][i].split("-")
            s['Salary'][i] = (float(s['Salary'][i][0]) + float(s['Salary'][i][1]))/2
        else:
            s['Salary'][i] = float(s['Salary_clean'][i])   

    elif 'parsemaine' in s['salaire'][i]:  
        s['Salary_type'][i] = 'w'
        s['Salary_clean'][i] = s['salaire'][i][:-10].replace('€', '') 
        if "-" in s['Salary_clean'][i]:
            s['Salary'][i] = s['Salary_clean'][i].split("-")
            s['Salary'][i] = (float(s['Salary'][i][0]) + float(s['Salary'][i][1]))/2
        else:
            s['Salary'][i] = float(s['Salary_clean'][i])  

    elif 'parjour' in s['salaire'][i]:  
        s['Salary_type'][i] = 'd'
        s['Salary_clean'][i] = s['salaire'][i][:-7].replace('€', '') 
        if "-" in s['Salary_clean'][i]:
            s['Salary'][i] = s['Salary_clean'][i].split("-")
            s['Salary'][i] = (float(s['Salary'][i][0]) + float(s['Salary'][i][1]))/2
        else:
            s['Salary'][i] = float(s['Salary_clean'][i]) 

    elif 'parheure' in s['salaire'][i]:  
        s['Salary_type'][i] = 'h'
        s['Salary_clean'][i] = s['salaire'][i][:-8].replace('€', '') 
        if "-" in s['Salary_clean'][i]:
            s['Salary'][i] = s['Salary_clean'][i].split("-")
            s['Salary'][i] = (float(s['Salary'][i][0]) + float(s['Salary'][i][1]))/2
        else:
            s['Salary'][i] = float(s['Salary_clean'][i]) 
    else:
        pass
        


# In[25]:


s['Salary_type'].value_counts()


# # Add one column of job type classification: data or dev
# ##Data related announce labeled as 1, Dev as 0 

# In[26]:


data_job_list = ["data scientist", "data consultant", "performance analyst", "data engineer","data miner",                  "intelligence artificielle", "data manager", "data analyst"] #According to scraping key word 

s['Job_class'] = s['keyword_metier'].apply(lambda x : 1 if x in data_job_list else 0)


# In[27]:


s['Job_class'].value_counts()


# # Create a new column Job_title

# In[28]:


###Delete h/f, f/h, (h/f), (f/h), -, /, 
##prepreration for analyse the title 

s['Job_title'] = s['titre_poste'].str.replace("\(h/f\)", "") ##why a list doesnot work? 
s['Job_title'] = s['Job_title'].str.replace("\(f/h\)", "")
s['Job_title'] = s['Job_title'].str.replace("h/f", "")
s['Job_title'] = s['Job_title'].str.replace("f/h", "")
s['Job_title'] = s['Job_title'].str.replace("-", "")
s['Job_title'] = s['Job_title'].str.replace("\/", "")
s['Job_title'] = s['Job_title'].str.replace("(", "")
s['Job_title'] = s['Job_title'].str.replace(")", "")


# # Tu peux ignorer cette partie de skills columns
# # Continuer avec la partie d'expérience 
# We dont use these columns
# 
# 
# # Create 10 new columns of skills: data qualities, n° of data qualities, data skills, n° of data skills, dev qualities, n° of dev qualities, dev skills, n° of dev skills, no technical skills, n° of no technical skills

# In[29]:


#### STEP1
##Create new columns Description and Description_filtered
## These columns contain tokens of decription

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer

##Tokenization 
s['Description'] = s['description'].apply(lambda x : re.sub('\W+', ' ', x)) #Delete all punctuations 
s['Description'] = s['Description'].apply(lambda x : re.sub('[0-9]+', ' ', x)) #Delete all numbers 
s['Description'] = s['Description'].apply(lambda x : word_tokenize(x))  

##Delete stopwords 
nltk.download('stopwords')                   #Download stopwords
stop_words = set(stopwords.words('french'))  #Define stopwords

##Create a new column 'Description_filtered' : stopwords filtered 
s['Description_filtered'] = s['Description'].apply(lambda x : [word for word in x if word not in stop_words]) 


# In[30]:


##Delete stopwords : update stop words list
##As French stopwrods are applied, the English stop words are not filtered 
##This list can be extended according to the needs of analytics 

s['Description_filtered'] = s['Description_filtered'].apply(lambda x : [word for word in x                                     if word not in ("a", "an", "and", "is", "the", "e", "to", 'it', 'g', 'f', 'h',                                                     'plus', 'sein', 'tels', 'profile', 'poste', 'etc', 'dont',                                                     'tant')])


# In[31]:


####STEP2
##Create data skill list, dev skill list and no technical skill list 

## qualities : skills in general such as ml, dl
## skills : specific skills/tools, such as python, power bi 


data_qualities_check_list = ['statistique', 'statistical', 'analyse', 'analysis', 'mathématiques', 'mathematics',                   'machine learning', 'ml', 'deep learning', 'dl','visualisation', 'big data', 'data'                  'bases de données', 'database', 'business intelligence', 'bi', 'intelligence artificielle', 'ia']
data_skills_check_list = ['python', 'r', 'sql', 'mongo', 'nlp', 'natural language processing', 'sas', 'spark',                           'hadoop', 'hive', 'scala', 'torch', 'tensorflow', 'power bi', 'tableau']             

dev_qualities_check_list = ['fullstack', 'full-stack', 'frontend','front-end', 'backend', 'back-end',                             'graphics', 'design', 'mobile', 'game', 'test', 'cloud', 'securité']
dev_skills_check_list = ['javascript','js', 'html','css','java', 'c', 'c#', 'c++', 'swift', 'ruby', 'php',                             'docker', '.net', 'aws', 'azure', 'api', 'wordpress'] 

no_technical_skills_check_list = ['gestion', 'management', 'communication', 'anglais', 'agile']


# In[32]:


####STEP 3
##Create columns related to skills

##Data
s['Data_qualities'] = s['Description'].apply(lambda x:                                                       [word for word in x if word in data_qualities_check_list])
s['Num_data_qualities'] = s['Data_qualities'].apply(lambda x : len(x))

s['Data_skills'] = s['Description'].apply(lambda x:                                                       [word for word in x if word in data_skills_check_list])
s['Num_data_skills'] = s['Data_skills'].apply(lambda x : len(x))

##Dev
s['Dev_qualities'] = s['Description_filtered'].apply(lambda x:                                                       [word for word in x if word in dev_qualities_check_list])
s['Num_dev_qualities'] = s['Dev_qualities'].apply(lambda x : len(x))
s['Dev_skills'] = s['Description'].apply(lambda x:                                                       [word for word in x if word in dev_skills_check_list])
s['Num_dev_skills'] = s['Dev_skills'].apply(lambda x : len(x))

##No technical 
s['No_technical_skills'] = s['Description'].apply(lambda x:                                                       [word for word in x if word in no_technical_skills_check_list])
s['Num_no_technical_skills'] = s['No_technical_skills'].apply(lambda x : len(x))


# In[33]:


###Rows with 0 skills (technical and no technical)
##26 rows 

s[(s['Num_data_qualities'] == 0) & (s['Num_data_skills'] == 0) &   (s['Num_dev_qualities'] == 0) & (s['Num_dev_skills'] == 0) & (s['Num_no_technical_skills'] == 0)]


# In[34]:


###Rows with 0 technical qualities and 0 technical skills 
s[(s['Num_data_qualities'] == 0) & (s['Num_data_skills'] == 0) &   (s['Num_dev_qualities'] == 0) & (s['Num_dev_skills'] == 0)]

##111 rows in total


# In[35]:


###PB: in the skill related columns, 2 types of duplicated exist. 
##same skills repeated
##'machine learning' == 'ml'

##Solution: 1)replace 2)set-list

###Review skill/quality check_list and replace keywords
#Data quality column:
s['Data_qualities'] = s['Data_qualities'].apply(lambda x: ['statistique' if word == 'statistical' else word for word in x]) 
s['Data_qualities'] = s['Data_qualities'].apply(lambda x: ['analyse' if word == 'analysis' else word for word in x])                                               
s['Data_qualities'] = s['Data_qualities'].apply(lambda x: ['mathématiques' if word == 'mathematics' else word for word in x])  
s['Data_qualities'] = s['Data_qualities'].apply(lambda x: ['ml' if word == 'machine learning' else word for word in x])  
s['Data_qualities'] = s['Data_qualities'].apply(lambda x: ['dl' if word == 'deep learning' else word for word in x])  
s['Data_qualities'] = s['Data_qualities'].apply(lambda x: ['bases de données' if word == 'database' else word for word in x])  
s['Data_qualities'] = s['Data_qualities'].apply(lambda x: ['bi' if word == 'business intelligence' else word for word in x])  
s['Data_qualities'] = s['Data_qualities'].apply(lambda x: ['ia' if word == 'intelligence artificielle' else word for word in x])  

###Data skill column 
s['Data_skills'] = s['Data_skills'].apply(lambda x: ['nlp' if word == 'natural language processing' else word for word in x])  

###Dev quality column
s['Dev_qualities'] = s['Dev_qualities'].apply(lambda x: ['fullstack' if word == 'full-stack' else word for word in x])
s['Dev_qualities'] = s['Dev_qualities'].apply(lambda x: ['frontend' if word == 'front-end' else word for word in x])  
s['Dev_qualities'] = s['Dev_qualities'].apply(lambda x: ['backend' if word == 'back-end' else word for word in x])  

###Dev skill column
s['Dev_skills'] = s['Dev_skills'].apply(lambda x: ['js' if word == 'javascript' else word for word in x])  


# In[36]:


###Delete duplicated skills and qualities 
s['Data_qualities'] = s['Data_qualities'].apply(lambda x: list(set(x)))
s['Data_skills'] = s['Data_skills'].apply(lambda x: list(set(x)))
s['Dev_qualities'] = s['Dev_qualities'].apply(lambda x: list(set(x)))
s['Dev_skills'] = s['Dev_skills'].apply(lambda x: list(set(x)))
s['No_technical_skills'] = s['No_technical_skills'].apply(lambda x: list(set(x)))

###Recalculate the number of qualities and skills 
s['Num_data_qualities'] = s['Data_qualities'].apply(lambda x : len(x))
s['Num_data_skills'] = s['Data_skills'].apply(lambda x : len(x))
s['Num_dev_qualities'] = s['Dev_qualities'].apply(lambda x : len(x))
s['Num_dev_skills'] = s['Dev_skills'].apply(lambda x : len(x))
s['Num_no_technical_skills'] = s['No_technical_skills'].apply(lambda x : len(x))


# # Create a new column regarding experience
# ###Experience data comes from two sources: description (how many years' experience) and job title (senior, junior)
# ###In general, experience is noted by 3 categories : 1, 2, 3 
# ###Category 1: 0-3 years(description) + junior(title)
# ###Cetegory 2: 3-5 years(description) + senior(title)
# ###Category 3: +5 years(description) + lead(title) + manager(title)

# In[37]:


###Regex : find experience related information

def find_experience_text(x):
    
    match1 = re.findall(r'expérience.*\d{1,2}\sans?', x) 
    match2 = re.findall(r'\d{1,2}\sans?\sd\'expérience', x)
    
    len_match = len(match1) + len(match2)
    
    if match1 == [] and match2 == []:
        return '0'
    
    return match1 + match2

s['Experience'] = s['description'].apply(lambda x: find_experience_text(x))


# In[38]:


s['Experience'].value_counts()


# In[39]:


###Extract the nunmbers from the text 

def find_numbers_in_experience_text(x):
    
    if x != '0': 
        nums = re.findall(r'\d{1,2}', ' '.join(x))     #find numbers
        nums = [num for num in nums if int(num) < 15]  #delete numbers > 15
        if len(nums) != 0:                             
            nums = nums[0]                             #Take the first number in the list
        else: 
            nums = 0                                   #if empty list --> nums = 0
    else:
        nums = 0 
    return nums


# In[40]:


s['Experience'] = s['Experience'].apply(lambda x: find_numbers_in_experience_text(x))


# In[41]:


s['Experience'].value_counts()


# In[43]:


s['Experience'] = s['Experience'].astype(str)


# In[48]:


s['Experience'][1]


# In[49]:


###For rows without experience information in the description column
##Try job title: take "senior", "junior", "lead", "manager"

for i in range(s.shape[0]):
    if s['Experience'][i] == '0':
        if 'senior' in s['titre_poste'][i]:
            s['Experience'][i] = 'senior'
        elif 'junior' in s['titre_poste'][i]:
            s['Experience'][i] = 'junior'
        elif 'lead' in s['titre_poste'][i]:
            s['Experience'][i] = 'lead'
        elif 'manager' in s['titre_poste'][i]:
            s['Experience'][i] = 'manager'
        else: 
            pass  


# In[50]:


s['Experience'].value_counts()


# In[51]:


###Transform experience to 3 categoreis : 

##Category 1: <3 years(description) + junior(title)
##Cetegory 2: 3-5 years(description) + senior(title)
##Category 3: >5 years(description) + lead(title) + manager(title)

s['Experience'] = s['Experience'].replace(['0', '1', '2', 'junior'], 1)
s['Experience'] = s['Experience'].replace(['3', '4', '5', 'senior'], 2)
s['Experience'] = s['Experience'].replace(['6', '7', '8', '9', '10', '11', '12', '13', '14', '15',                                            'manager', 'lead'], 3)

##If experience is not required in the description, we fill it with 1 which asks for the minimum experience   
s['Experience'] = s['Experience'].apply(lambda x: 1 if x == '0' else x)


# In[52]:


s['Experience'].value_counts()


# # FIN

# In[53]:


s.columns


# In[54]:


s.shape


# # Save Database Version1.0

# In[56]:


import pickle

pickle_out_v1 = open("indeed_db_v1.pickle", "wb")
pickle.dump(s, pickle_out_v1)
pickle_out_v1.close()


# In[57]:


s.to_csv(r'/Users/yinghua/Documents/indeed_db_v1.csv')


# In[ ]:





# # Save Database Version2.0
# ##Drop redundant columns(index cols, cols created by scraping)

# In[59]:


s_v2 = s.drop(['Unnamed: 0.1', 'id', 'meta1', 'meta2',
       'titre_poste','localisation', 'contrat', 'salaire',
       'date_publication', 'keyword_localisation'], axis = 1)


# In[60]:


s_v2.to_csv(r'/Users/yinghua/Documents/indeed_db_v2.csv')


# In[61]:


pickle_out_v2 = open("indeed_db_v2.pickle", "wb")
pickle.dump(s_v2, pickle_out_v2)
pickle_out_v2.close()


# ## Classify job announce : data or dev (NLP)
# ## Test models based on job title or job descripiton 

# In[62]:


###Select job announce with annual salaries: 
###monthly, weekly and daily salaries should be analysed seperately, or find a coefficient to convert them into annual salaries. 
oas = s_v2[s_v2['Salary_type'] == 'y']
oas.shape


# In[63]:


#How many rows of type data
oas['Job_class'].value_counts()


# In[64]:


from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score


# # CountVectorizer - description (0.9223)

# In[65]:


###Split dataset into train and test set

description_train,description_test,y_train,y_test = train_test_split(oas['description'], oas['Job_class'],                                                                      test_size = 0.2, random_state = 0)


# In[66]:


cv_d = CountVectorizer(stop_words = stop_words)
cv_d.fit(description_train)
X_train_d = cv_d.transform(description_train)
X_test_d = cv_d.transform(description_test)


# In[67]:


X_train_d.shape


# In[68]:


param_grid = {'C': [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100] }
logreg_d = GridSearchCV(LogisticRegression(penalty='l2'), param_grid, cv=5)
logreg_d.fit(X_train_d, y_train)
logreg_d.best_estimator_


# In[69]:


clf_cv_d = logreg_d.best_estimator_
y_pred_cv_d = clf_cv_d.predict(X_test_d)


# In[70]:


print("clf_cv_d score: ", clf_cv_d.score(X_test_d,y_test))
print("Confusion matrix: ", confusion_matrix(y_test, y_pred_cv_d))


# In[71]:


###Top 20 words related to data 
coeffs_cv_d = clf_cv_d.coef_[0]
indexes_cv_d = np.argsort(coeffs_cv_d)
top_20 = [cv_d.get_feature_names()[indexes_cv_d[i]] for i in range(-20,0)]
top_20


# # CountVectorizer - job title (0.8716)

# In[72]:


title_train,title_test,y_train,y_test = train_test_split(oas['Job_title'], oas['Job_class'],                                                                      test_size = 0.2, random_state=0)


# In[73]:


cv_t = CountVectorizer(stop_words=stop_words)
cv_t.fit(title_train)
X_train_t = cv_t.transform(title_train)
X_test_t = cv_t.transform(title_test)


# In[74]:


X_train_t.shape


# In[75]:


param_grid = {'C': [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100] }
logreg_t = GridSearchCV(LogisticRegression(penalty='l2'), param_grid, cv=5)
logreg_t.fit(X_train_t, y_train)
logreg_t.best_estimator_


# In[76]:


clf_cv_t = logreg_t.best_estimator_
y_pred_cv_t = clf_cv_t.predict(X_test_t)


# In[77]:


print("clf_cv_t score: ", clf_cv_t.score(X_test_t,y_test))
print("Confusion matrix: ", confusion_matrix(y_test, y_pred_cv_t))


# In[78]:


##Top 20 words related to data

coeffs_cv_t = clf_cv_t.coef_[0]
indexes_cv_t = np.argsort(coeffs_cv_t)
top_20 = [cv_t.get_feature_names()[indexes_cv_t[i]] for i in range(-20,0)]
top_20


# # tfidf - description (0.9223)

# In[79]:


v_tfidf = TfidfVectorizer()
v_tfidf.fit(description_train)
X_train_tfidf = v_tfidf.transform(description_train)
X_test_tfidf = v_tfidf.transform(description_test)


# In[80]:


print(v_tfidf.vocabulary_)


# In[81]:


X_train_tfidf.shape


# In[82]:


param_grid = {'C': [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100] }
logreg_tfidf = GridSearchCV(LogisticRegression(penalty='l2'), param_grid, cv=5)
logreg_tfidf.fit(X_train_tfidf, y_train)
logreg_tfidf.best_estimator_


# In[83]:


clf_tfidf = logreg_tfidf.best_estimator_
y_pred_tfidf = clf_tfidf.predict(X_test_tfidf)


# In[84]:


print("clf_tfidf score: ", clf_tfidf.score(X_test_tfidf,y_test))
print("Confusion matrix: ", confusion_matrix(y_test, y_pred_tfidf))


# # tfidf - title

# In[ ]:





# # Test on a complete skill list

# In[85]:


all_skills_list = ['statistique', 'statistical', 'analyse', 'analysis', 'analytics', 'analytiques', 'outils analytiques',                        'mathématiques', 'mathematics', 'machine learning', 'ml', 'deep learning', 'dl',                       'algorithme', 'algorithm', 'bases de données', 'excel'                       'database', 'business intelligence', 'bi', 'intelligence artificielle', 'ia', 'python',                        'numpy', 'pandas', 'matplotlib', 'seaborn', 'plotly', 'visualisation','neuron', 'ann',                        'cnn', 'lstm',                        'keras', 'gan', 'bayesian', 'bayésienne', 'reinforcement learning',                        'r', 'shine', 'sql', 'mysql', 'mongo', 'nosql', 'nlp', 'natural language processing',                        'sas', 'spark', 'big data', 'hadoop', 'hive', 'mapreduce', 'scala', 'pytorch',                        'tensorflow', 'sklearn', 'power bi', 'tableau',                        'fullstack', 'full-stack', 'frontend','front-end', 'backend', 'back-end', 'graphics',                        'design', 'mobile', 'game', 'test', 'cloud', 'cybersecurité', 'javascript','js',                        'html','css','java', 'c', 'c#', 'c++', 'swift', 'ruby', 'php', 'docker', '.net', 'aws',                        'azure', 'api', 'wordpress', 'angularjs', 'bootstrap','django', '.net', 'jquery', 'symphony', 
                       'frameworks', 'perl', 'xml', 'xhtml', 'optimisation web', 'drupal', 'poo', 'oop', 'cms', 'ulkit'\
                       'pure.css', 'git', 'mobile', 'mocha']


# In[86]:


s_v2['All_skills'] = s_v2['Description'].apply(lambda x:                                                       [word for word in x if word in all_skills_list])


# In[87]:


s_v2['All_skills'] = s_v2['All_skills'].apply(lambda x: list(set(x)))


# In[88]:


s_v2['Num_all_skills'] = s_v2['All_skills'].apply(lambda x: len(x))


# In[91]:


s_v2.head()


# In[94]:


s_v2.columns


# In[101]:


#Create a list which combines all description together 

all_description = []
for i in range(s_v2.shape[0]):
    for word in s_v2['Description_filtered'][i]: 
        all_description.append(word)


# In[102]:


len(all_description)


# In[99]:


from collections import Counter


# In[103]:


bag_of_words = Counter(all_description)
print(bag_of_words)


# In[104]:


bag_of_words = dict(bag_of_words)
bag_of_words_df = pd.DataFrame.from_dict(bag_of_words, orient = 'index', columns = ['times'])
most_mentioned = bag_of_words_df[bag_of_words_df['times'] > 300]
most_mentioned.sort_values(by=['times'], ascending = False)


# In[111]:


all_skills_dict = {}
for i in all_skills_list:
    if i not in bag_of_words:
        pass
    else: 
        all_skills_dict[i] = bag_of_words[i]
        print(i, bag_of_words[i])

sorted(all_skills_dict.items(), key=lambda x: x[1], reverse=True)


# In[130]:


###Frequence > 50 
select_columns = ['java', 'analyse', 'cloud', 'python', 'javascript', 'sql', 'js', 'php', 'html', 'r',                   'mobile', 'git', 'css', 'api', 'docker', 'aws', 'design', 'backend', 'azure', 'bi',                   'mysql', 'test', 'jquery', 'frontend', 'fullstack', 'frameworks', 'spark', 'scala', 'ruby',                   'analytics', 'hadoop', 'bootstrap', 'ia', 'django', 'nosql', 'angularjs', 'visualisation',                   'cms', 'xml', 'swift']


# In[202]:


copy = s_v2.copy()


# In[203]:


copy.columns


# In[204]:


copy['Salary_group'] = pd.cut(copy['Salary'], bins = 3, labels = [1, 2, 3])


# In[205]:


copy['Salary_group'].value_counts()


# In[206]:


copy.head()


# In[207]:


copy = copy[['Salary_group', 'Salary_type', 'description', 'Contract', 'Location', 'Job_class', 'Experience']]


# In[208]:


copy.shape


# In[209]:


copy = copy[copy['Salary_type'] == 'y']
copy.reset_index(drop=True, inplace=True)


# In[210]:


copy.shape


# In[211]:


for col in select_columns:
    copy[col] = 0


# In[212]:


for i in range(copy.shape[0]):    
    for col in select_columns:
        if col in copy['description'][i]:
            copy[col][i] = 1


# In[147]:


copy = pd.get_dummies(copy, prefix = ['Contract', 'Location'], columns = ['Contract', 'Location'])


# In[148]:


copy = copy.drop(['Contract_intérim', 'Location_44'], axis = 1)


# In[149]:


copy.shape


# In[150]:


copy.head()


# In[181]:


copy = copy.drop(['description', 'Description_filtered', 'Salary_type'], axis = 1)


# In[183]:


x.shape


# In[184]:


y.shape


# In[170]:


copy['Contract_cdd'] = copy['Contract_cdd'].astype('int64')
copy['Contract_cdi'] = copy['Contract_cdi'].astype('int64')
copy['Contract_contrat pro'] = copy['Contract_contrat pro'].astype('int64')
copy['Contract_freelance / indépendant'] = copy['Contract_freelance / indépendant'].astype('int64')
copy['Location_31'] = copy['Location_31'].astype('int64')
copy['Location_33'] = copy['Location_33'].astype('int64')
copy['Location_69'] = copy['Location_69'].astype('int64')
copy['Location_75'] = copy['Location_75'].astype('int64')


# In[164]:


from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler


# In[185]:


copy.head()


# In[195]:


x = copy.iloc[1:-1].values
y = copy.iloc[0].values


# In[196]:


x


# In[197]:


y


# In[190]:


x.shape


# In[191]:


y.shape


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


data_skills = ['data', 'machine Learning', "ML", 'deep learning', 'dl', 'AI', 'IA',                'artificial intelligence', 'intelligence artificielle', 'R','python', 'SAS',               'Natural Language Processing', 'nlp', 'Power BI', 'BI', 'business intelligence',                "mongo", 'SQL', 'Linux', "flask", 'docker', 'analytics', 'azure', 'regression', 'model',               'modèles', 'classification', 'TensorFlow', 'spark', 'hadoop', 'statistique', 'MongoDB'               'mathématiques', 'sklearn', 'Big Data', 'NoSQL', 'outils analytiques', 'MapReduce', 'analyse de données',                'MySQL', 'algorithme']
data_skills = [skill.lower() for skill in data_skills]
print(len(data_skills))

data_skills_dict = {}
for i in data_skills:
    if i not in bag_of_words:
        pass
    else: 
        data_skills_dict[i] = bag_of_words[i]
        print(i, bag_of_words[i])

sorted(data_skills_dict.items(), key=lambda x: x[1], reverse=True)


# In[107]:


data_skills_dict = {}
for i in data_skills:
    if i not in bag_of_words:
        pass
    else: 
        data_skills_dict[i] = bag_of_words[i]
        print(i, bag_of_words[i])


# In[ ]:


sorted(data_skills_dict.items(), key=lambda x: x[1], reverse=True)


# In[ ]:


dev_skills = ["mongo", 'SQL', 'Linux', "flask", 'docker', 'django', 'HTML', 'CSS',               'JavaScript', 'PHP', 'PHP5', 'CSS3', 'HTML5', 'AngularJS', 'Bootstrap',               'Java', 'C', 'C#', 'C++', '.NET', 'JQuery', 'Symphony','Frameworks', 'Perl', 'C', 'C++', 'NoSQL',               'Angular JS', 'MySQL', 'django','Ruby on Rails',               'XML', 'XHTML', 'Optimisation Web', 'Swift', 'MySQL', 'WordPress', 'Drupal',              'POO', 'OOP', 'CMS','Foundation', 'UlKit', 'Pure.css',              'Sass', 'Less', 'Git', 'versioning', 'Mobile Friendly', 'mobile first', 'Mocha', 'Chai',              'Sinon', 'Jasmine', 'JS', 'SQL server', 'MongoDB', 'CMS']
dev_skills = [skill.lower() for skill in dev_skills]
print(len(dev_skills))

dev_skills_dict = {}
for i in dev_skills:
    if i not in bag_of_words:
        pass
    else: 
        dev_skills_dict[i] = bag_of_words[i]
        
sorted(dev_skills_dict.items(), key=lambda x: x[1], reverse=True)


# In[ ]:


dev_skills_dict = {}
for i in dev_skills:
    if i not in bag_of_words:
        pass
    else: 
        dev_skills_dict[i] = bag_of_words[i]
        print(i, bag_of_words[i])


# In[ ]:


sorted(dev_skills_dict.items(), key=lambda x: x[1], reverse=True)


# In[ ]:


general_skills=['organisé', 'rigoureux','anglais',                'MapReduce', 'analyse de données', 'travailler en équipe',                 'curieux', 'créatif' , 'rigoureux', 'gestion de projet', 'gerer un projet',                'créativité', 'communication', 'polyvalent', 'communiquer']
general_skills = [skill.lower() for skill in general_skills]
print(len(general_skills))

general_skills_dict = {}
for i in general_skills:
    if i not in bag_of_words:
        pass
    else: 
        general_skills_dict[i] = bag_of_words[i]

sorted(general_skills_dict.items(), key=lambda x: x[1], reverse=True)


# In[ ]:


general_skills_dict = {}
for i in general_skills:
    if i not in bag_of_words:
        pass
    else: 
        general_skills_dict[i] = bag_of_words[i]
        print(i, bag_of_words[i])


# In[ ]:


sorted(general_skills_dict.items(), key=lambda x: x[1], reverse=True)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




