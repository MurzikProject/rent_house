# -*- coding: utf-8 -*-
"""
Created on Tue Dec 18 14:44:28 2018

@author: dn210481pav1
"""

# import librarias
import time
import numpy as np
import pandas as pd
# Matplotlib visualization
import matplotlib.pyplot as plt 
plt.style.use('ggplot')
%matplotlib inline
# Set default font size
plt.rc("font", size=14)
# Seaborn for visualization
from IPython.core.pylabtools import figsize
import seaborn as sns 
sns.set(font_scale = 2)
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn import linear_model
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn import ensemble


# FUNCTIONS
# 1. Распределение значений целевой функции и построение гистограммы
def part_rent_clients(dataset,score):
    count_score = dataset.groupby(score).size()
    part_score = count_score/len(dataset)
    print('Количество объектов класса "оформил ипотеку" составляет '+ str(count_score[1]))
    print('Количество объектов класса "не оформил ипотеку" составляет '+ str(count_score[0]))
    print('Доля класса "оформил ипотеку": '+ str((part_score[1])*100)+ ' %')
    print('Доля класса "не оформил ипотеку": '+ str((part_score[0])*100)+ ' %')
    
    sns.countplot(x=score,data=dataset,palette='hls')
    plt.show
    

# 2. Функция, демонстрирующая методологию подбора клиентов и их распределение
def clients_property(dataset,score,property):
    count_property_with_ipoteka = dataset[dataset[score]==1].groupby(property).size()
    part_property_with_ipoteka = round((count_property_with_ipoteka/len(dataset[dataset[score]==1]))*100)
    
    count_property_without_ipoteka = dataset[dataset[score]==0].groupby(property).size()
    part_property_without_ipoteka = round((count_property_without_ipoteka/len(dataset[dataset[score]==0]))*100)

    df_c_1 = pd.DataFrame(count_property_with_ipoteka)
    df_c_1 = df_c_1.rename(columns={0: 'N_with_ipo'})
    
    df_p_1 = pd.DataFrame(part_property_with_ipoteka)
    df_p_1 = df_p_1.rename(columns={0: '%_with_ipo'})
    
    df_c_0 = pd.DataFrame(count_property_without_ipoteka)
    df_c_0 = df_c_0.rename(columns={0: 'N_without_ipo'})
    
    df_p_0 = pd.DataFrame(part_property_without_ipoteka)
    df_p_0 = df_p_0.rename(columns={0: '%_without_ipo'})
    
    ready_df = pd.concat([df_c_1, df_p_1, df_c_0, df_p_0], axis = 1)
    
    print(ready_df)
    
# 3. Анализ заполнения признаков данными
def missing_values_table(dataset):
        # Общее число отсутствующих данных        
        mis_val = dataset.isnull().sum()
        # процент отсутствующих данных
        mis_val_percent = 100 * dataset.isnull().sum() / len(dataset)
        # Создание таблицы с результатами двух предыдущих операторов
        mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
        # Переименование полей
        mis_val_table_ren_columns = mis_val_table.rename(
        columns = {0 : 'Missing Values', 1 : '% of Total Values'})
        # Сортировка таблицы по % отсутвующих значений по убыванию
        mis_val_table_ren_columns = mis_val_table_ren_columns[
            mis_val_table_ren_columns.iloc[:,1] != 0].sort_values(
        '% of Total Values', ascending=False).round(1)
        # Возврат датасета с необходимой информацией
        return mis_val_table_ren_columns
    
# 4. Статистика по типам данных признаков
def dataset_params(dataset,score):
    dataset = dataset.drop(columns = [score])
    dtypes_list = pd.unique(dataset.dtypes)
    count_features = 0
    i=0
    for i in range(len(dtypes_list)):
        dt = str(dtypes_list[i])
        dt_list = list(dataset.select_dtypes(include=[dt]).columns)
        count_features += len(dt_list)
        print('- '+str(dt)+': '+str(len(dt_list)))
        i += 1
    print('The total number of predictors is '+str(count_features))

# 5. Избавление от мультиколлинеарности
def remove_collinear_features(dataset, threshold):
    '''
    Objective:
        Remove collinear features in a dataframe with a correlation coefficient
        greater than the threshold. Removing collinear features can help a model
        to generalize and improves the interpretability of the model.
        
    Inputs: 
        threshold: any features with correlations greater than this value are removed
    
    Output: 
        dataframe that contains only the non-highly-collinear features
    '''
    
    # Dont want to remove correlations between REGULAR_CUSTOMER
    y = dataset['IPOTEKA']
    x = dataset.drop(columns = ['IPOTEKA'])
    
    # Calculate the correlation matrix
    corr_matrix = dataset.corr()
    iters = range(len(corr_matrix.columns) - 1)
    drop_cols = []

    # Iterate through the correlation matrix and compare correlations
    for i in iters:
        for j in range(i):
            item = corr_matrix.iloc[j:(j+1), (i+1):(i+2)]
            col = item.columns
            row = item.index
            val = abs(item.values)
            
            # If correlation exceeds the threshold
            if val >= threshold:
                # Print the correlated features and the correlation value
                # print(col.values[0], "|", row.values[0], "|", round(val[0][0], 2))
                drop_cols.append(col.values[0])

    # Drop one of each pair of correlated columns
    drops = set(drop_cols)
    dataset = dataset.drop(columns = drops)

    dataset['IPOTEKA'] = y
               
    return dataset

# 6. Подсчет разницы между мат ожиданиями
def correl_calc(dataset):
    mean = np.array(dataset.dropna()).mean()
    std = np.array(dataset.dropna()).std()
    dataset.fillna(mean, inplace = True)
    dataset = dataset.apply(lambda x: (x - mean) / std)
    e_x1 = dataset[numeric_features.IPOTEKA == 1].mean()
    e_x2 = dataset[numeric_features.IPOTEKA == 0].mean()
    return (e_x1 - e_x2)

# 7. Построение распределения вещественных признаков в разрезе классов
def numerical_features_distrib(dataset,features_list):
    columns = features_list.Feature.iloc[:20]
    fig, axs = plt.subplots(20, figsize = (15,50))
    sns.set(color_codes=True)
    for ax, column in zip(axs, columns):
        sns.kdeplot(dataset[column][dataset['IPOTEKA'] == 1],
                    ax = ax,
                    color = "blue",
                    label = str(column)+" for IPOTEKA = 1")
        sns.kdeplot(dataset[column][dataset['IPOTEKA'] == 0],
                    ax = ax,
                    color = "orange",
                    label = str(column)+" for IPOTEKA = 0")

# 8. Построение гистограммы категориальных признаков в разрезе классов 
def categorical_features_hist(dataset,feature):
    return sns.catplot(x=feature,
               hue=feature,
               col='IPOTEKA',
               data=dataset,
               kind='count',
               height=4,
               aspect=.7)

# 9. Функция StraitKFold позволит построить и оценить линейную модель и градиентный бустинг.
def StraitKFold(classifier, X, y):
    skf = StratifiedShuffleSplit(n_splits = 3)
    skf.get_n_splits(X, y)
    y_scores = pd.DataFrame()
    y_tests = pd.DataFrame()
    y_pred = pd.DataFrame() 
    f1 = np.array([])
    accuracy = np.array([])
    n = 0
    for train_index, test_index in skf.split(X, y):
        classifier.fit(X.iloc[train_index, :], y.iloc[train_index, 0])
        y_scores['fold_'+str(n)] = classifier.decision_function(X.iloc[test_index, :])
        y_pred['fold_'+str(n)] = classifier.predict(X.iloc[test_index, :])
        y_tests['fold_'+str(n)] = y.iloc[test_index, 0].values
        f1 = np.append(f1, metrics.f1_score(y.iloc[test_index, 0], y_pred.iloc[:,n]))
        accuracy = np.append(f1, metrics.accuracy_score(y.iloc[test_index, 0], y_pred.iloc[:,n]))
        n += 1
    f1_score = np.mean(f1)
    accuracy = np.mean(accuracy)
    print('mean accuracy score: '+str(accuracy))
    print('mean f1 score: '+str(f1_score))
    return y_scores, y_tests, accuracy, f1_score

# 10. Функция построения AUC_ROC  
def ROC(y_scores, y_test):
    plt.figure(figsize = (7,7))
    Mean_ROC = np.array([])
    n = 0
    for i, j in zip(y_scores, y_tests):
        fpr, tpr, thresholds = metrics.roc_curve(y_tests[i], y_scores[j])
        roc_auc = metrics.auc(fpr, tpr)
        lw = 2
        Mean_ROC = np.append(Mean_ROC, roc_auc)
        plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC fold '+str(n)+' (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        n += 1
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    plt.title('Mean ROC area={0:0.3f}'.format(Mean_ROC.mean()))
    plt.show()
    
# 11. Функция построения графика сравнения производительности по указанной метрике
def model_comparison_chart(dataset,value):
    plt.style.use('fivethirtyeight')
    figsize(8, 6)
    
    dataset.sort_values(value, ascending = False).plot(x = 'model',
                                                       y = value,
                                                       kind = 'barh',
                                                       color = 'green',
                                                       edgecolor = 'black')
     
    plt.ylabel('Models')
    plt.yticks(size = 14)
    plt.xlabel(value)
    plt.xticks(size = 14)
    plt.xlim([0.0, 1.05])
    plt.title('Model comparison (cross-validation) on '+value, size = 20)

#==============================================================================
# 1. DATA CLEANING AND FORMATTING
#==============================================================================

# Считываем данные в датафрейм
#rent_clients = pd.read_csv('D:/Models/development/9_rent_house/ipoteka_clid_20190412_rem.csv', low_memory=False, encoding = "ISO-8859-1")
rent_clients = pd.read_csv('/home/anton/Projects/python/development/9_rent_house/ipoteka_clid_20190412_rem.csv', encoding = "ISO-8859-1")

rent_clients.shape

# Посмотрим на распределение целевой функции (оформит/не оформит и ипотеку)
part_rent_clients(rent_clients,'IPOTEKA')

# посмотрим на распределение клиентов по типу владения недвижимости в обоих группах
clients_property(rent_clients,'IPOTEKA','CL_ESTATE_TYPE')

# Удалим ненужные поля ай-ди клиентов
rent_clients = rent_clients.drop(['REP_CLID','CLID_CRM','CLID_TRAN'], axis=1)

# Заменим все значиения "Not Available" на np.nan
rent_clients = rent_clients.replace({'Not Available': np.nan})

# Удалим из датасета те поля, в которых заполнение менее 50%
missing_features = missing_values_table(rent_clients.drop(columns = ['IPOTEKA']))
missing_columns = list(missing_features[missing_features['% of Total Values'] > 50.0].index)
rent_clients = rent_clients.drop(list(missing_columns), axis = 1)

rent_clients.shape

# Посмотрим на распределение признаков по типам данных
dataset_params(rent_clients,'IPOTEKA')

# разделим признаки на количественные и вещественные
numeric_features = rent_clients.select_dtypes(include = [np.number])
numeric_features.shape

categorical_features = rent_clients.select_dtypes(include=[np.object])
categorical_features.shape

# Все количественные признаки проверим на мультиколлинеарность
numeric_features = remove_collinear_features(numeric_features, 0.6)
numeric_features.shape
dataset_params(numeric_features,'IPOTEKA')

# Обработаем категориальные признаки с помощью LabelEncoder
labelencoder = LabelEncoder()
z = len(categorical_features.columns)
for x in range(0,z):
    categorical_features.iloc[:,x] = labelencoder.fit_transform(categorical_features.iloc[:,x].values.astype(str))

categorical_features.shape

# соединяем категориальные и количественные признаки
features = pd.concat([numeric_features, categorical_features], axis = 1)
features.shape

# =============================================================================
# 2. FEATURE ENGINEERING AND SELECTION
# =============================================================================

# =============================================================================
# Посмотрим на влияние вещественных признаков на целевую переменную
# =============================================================================
number_corr_columns = list(numeric_features.drop(columns = ['IPOTEKA']).columns)
number_corr_values = []
number_nan_values = []

for (i, column) in enumerate(number_corr_columns):
    value = correl_calc(numeric_features[column])
    if np.isnan(value):
        number_nan_values.append(column)
    else:
        number_corr_values.append((column,np.abs(value)))

# для удобства из списка (corr_values) создадим dataframe 'corr_data':
number_corr_data = pd.DataFrame(number_corr_values, columns = ['Feature' , 'corr_value'])

# отсортируем и выведем топ-20 вещественных признаков:
sort_number_corr_data = number_corr_data.sort_values(by=['corr_value'], ascending=False)
top20_number_sort_corr_data = sort_number_corr_data[:20]
top20_number_sort_corr_data

# для вышеприведенных 20 вещественных признаков построим распределение в разрезе классов
numerical_features_distrib(features,top20_number_sort_corr_data)

# =============================================================================
# Посмотрим на влияние категориальных признаков на целевую переменную
# =============================================================================
object_corr_columns = list(categorical_features.columns)
object_corr_values = []
object_nan_values = []

for (i, column) in enumerate(object_corr_columns):
    value = correl_calc(categorical_features[column])
    if np.isnan(value):
        object_nan_values.append(column)
    else:
        object_corr_values.append((column,np.abs(value)))

# для удобства из списка (corr_values) создадим dataframe 'corr_data':
object_corr_data = pd.DataFrame(object_corr_values, columns = ['Feature' , 'corr_value'])

# отсортируем и выведем топ-20 категориальных признаков:
sort_object_corr_data = object_corr_data.sort_values(by=['corr_value'], ascending=False)
top20_object_sort_corr_data = sort_object_corr_data[:20]
top20_object_sort_corr_data

# построим непосредственно гистограммы категориальных признаков
categorical_features_columns = list(top20_object_sort_corr_data['Feature'].values)

for x in categorical_features_columns:
    categorical_features_hist(features,x)

categorical_features_hist(features,'FL_P24')

# =============================================================================
# Конструируем новый датасет со всеми объектами и 40 отобранными предикторами
# =============================================================================
features_final = features['IPOTEKA']
    
for i in top20_number_sort_corr_data['Feature']:
    features_final = pd.concat([features_final, features[i]],axis=1, sort=False)
    
for i in top20_object_sort_corr_data['Feature']:
    features_final = pd.concat([features_final, features[i]],axis=1, sort=False)

features_final.shape

features_final.head(12)
# =============================================================================
# 3. CROSS_VALIDATION
# =============================================================================
features = features_final.drop(columns='IPOTEKA')
targets = pd.DataFrame(features_final['IPOTEKA'])

# Разделим исходный датасет на треннировочный и тестовый
X_train, X_test, y_train, y_test = train_test_split(features,
                                                    targets,
                                                    test_size=0.30,
                                                    random_state=42)

train_features = X_train
test_features = X_test

# Обучим Imputer на треннирововчной базе и трансформируем ее и тестовую базы
imp_mean = SimpleImputer(missing_values=np.nan, strategy='median')
X_train = imp_mean.fit_transform(train_features)
X_test = imp_mean.transform(test_features)

# Проверим количесвто пустых значений в тестовом и обучающем датасете
print('Missing values in training features: ', np.sum(np.isnan(X_train)))
print('Missing values in testing features:  ', np.sum(np.isnan(X_test)))

X_train = pd.DataFrame(X_train)
X_test = pd.DataFrame(X_test)

X_train.columns = train_features.columns
X_test.columns = train_features.columns

X_train.shape
y_train.shape
# =============================================================================
# 4. COMPARE SEVERAL MACHINE LEARNING MODELS ON A PERFORMANCE METRIC
# =============================================================================
# Dataframe to hold the results
model_comparison = pd.DataFrame({'model': ['Logistic Regression',
                                           'Ridge Classifier',
                                           'Gradient Boosting'],
                                           'accuracy': [0.0,0.0,0.0],
                                           'f1_score': [0.0,0.0,0.0]})

# Logistic Regression
logReg = linear_model.LogisticRegression()
y_scores, y_tests, model_comparison['accuracy'][0], model_comparison['f1_score'][0] = StraitKFold(logReg, X_train, y_train)
ROC(y_scores, y_tests)

# Ridge classifier
ridge = linear_model.RidgeClassifier(random_state=2)
y_scores, y_tests, model_comparison['accuracy'][1], model_comparison['f1_score'][1] = StraitKFold(ridge, X_train, y_train)
ROC(y_scores, y_tests)

# градиентный бустинг:
gradBoost = ensemble.GradientBoostingClassifier()
y_scores, y_tests, model_comparison['accuracy'][2], model_comparison['f1_score'][2] = StraitKFold(gradBoost, X_train, y_train)
ROC(y_scores, y_tests)

# ПОсмотрим на матрицу с показателями производительности
model_comparison.head()

# Сравним производительность всех моделей по метрике accuracy
model_comparison_chart(model_comparison,'accuracy')

# Сравним производительность всех моделей по метрике f1_score
model_comparison_chart(model_comparison,'f1_score')


# =============================================================================
# Первая модель - логистическая регрессия
# =============================================================================
logreg = LogisticRegression()
logreg.fit(X_train, y_train)

# Спрогнозируем результат тестового набора и проверим  долю правильных ответов
y_pred = logreg.predict(X_test)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test, y_test)))

# Посмотрим распределение правильных прогнозов и неправильных!
confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)

# Расчитаем точность, полноту и f-меру и число вхождений каждого класса в тест
print(classification_report(y_test, y_pred))

# Построим ROC-кривую
logit_roc_auc = roc_auc_score(y_test, logreg.predict(X_test))
fpr, tpr, thresholds = roc_curve(y_test, logreg.predict_proba(X_test)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC')
plt.legend(loc="lower right")
plt.savefig('Log_ROC')
plt.show()

# =============================================================================
# Проверим работоспособность модели на клиентах, оформивших ипотеку
# после построения модели.
# =============================================================================
# Считываем данные проверки модели в датафрейм
check_rent_clients = pd.read_csv('/home/anton/Projects/python/development/9_rent_house/check_property/real_check_property_model_clid_20190613.csv', encoding = "ISO-8859-1")
#check_rent_clients = pd.read_csv('D:/Models/development/9_rent_house/check_property/real_check_property_model_clid_20190613.csv', encoding = "ISO-8859-1")

check_rent_clients.head()
check_rent_clients.shape

# Удалим ненужные поля ай-ди клиентов
check_rent_clients = check_rent_clients.drop(['CLID_CRM','CLID_TRAN'], axis=1)

# Заменим все значиения "Not Available" на np.nan
check_rent_clients = check_rent_clients.replace({'Not Available': np.nan})

# Предобработаем все категориальные предикторы
new_categorical_features = check_rent_clients.select_dtypes(include=[np.object])
new_categorical_features.shape

# Обработаем категориальные признаки с помощью LabelEncoder
z = len(new_categorical_features.columns)
for x in range(0,z):
    new_categorical_features.iloc[:,x] = labelencoder.fit_transform(new_categorical_features.iloc[:,x].values.astype(str))

# Выделим все вещественные предикторы 
new_numeric_features = check_rent_clients.select_dtypes(include = [np.number])

# соединяем категориальные и количественные признаки
new_features = pd.concat([new_numeric_features, new_categorical_features], axis = 1)
new_features.shape
new_features.head()

# Заполним отсутствующие данные медианными значениями
new_train_features = new_features
new_features = imp_mean.fit_transform(new_train_features)
new_features = pd.DataFrame(new_features)
new_features.columns = new_train_features.columns

# Получаем итоговый датафрейм для анализа
client_100 = pd.DataFrame()
client_100 = new_features
client_100.shape
client_100.head()

# Создаем итоговую таблицу с ай-ди клиентов расчитанными значениями 
y = client_100['REP_CLID']
x = client_100.drop(columns = ['REP_CLID'])
exit_data = pd.DataFrame(columns = ['CLID','PRED'])

# Заполним итоговую таблицу готовыми значениями
for i in range(len(y)):
    z = logreg.predict(x[i:i+1])
    exit_data.loc[i,'CLID'] = (y[i])
    exit_data.loc[i,'PRED'] = z[0].astype(int)

# Посмотрим на распределение расчитанных значений и отклонение от рельных значений
exit_data.shape
exit_data.head()
part_rent_clients(exit_data,exit_data['PRED'])





# Считываем данные проверки модели в датафрейм
#model_rent_clients = pd.read_csv('/home/varvara/anton/projects/9_rent_house/check_property/check_property_model_clid_20190524.csv', encoding = "ISO-8859-1")
#model_rent_clients = pd.read_csv('D:/Models/9_rent_house/check_property/check_property_model_clid_20190527.csv', low_memory=False, encoding = "ISO-8859-1")
model_rent_clients = pd.read_csv('/home/anton/Projects/python/development/9_rent_house/check_property/check_property_model_clid_20190527.csv', encoding = "ISO-8859-1")

model_rent_clients.head()
model_rent_clients.shape

# Удалим ненужные поля ай-ди клиентов
model_rent_clients = model_rent_clients.drop(['CLID_CRM','CLID_TRAN'], axis=1)

# Заменим все значиения "Not Available" на np.nan
model_rent_clients = model_rent_clients.replace({'Not Available': np.nan})

# Предобработаем все категориальные предикторы
new_categorical_features = model_rent_clients.select_dtypes(include=[np.object])
new_categorical_features.shape

# Обработаем категориальные признаки с помощью LabelEncoder
z = len(new_categorical_features.columns)
for x in range(0,z):
    new_categorical_features.iloc[:,x] = labelencoder.fit_transform(new_categorical_features.iloc[:,x].values.astype(str))

# Выделим все вещественные предикторы 
new_numeric_features = model_rent_clients.select_dtypes(include = [np.number])

# соединяем категориальные и количественные признаки
new_features = pd.concat([new_numeric_features, new_categorical_features], axis = 1)
new_features.shape
new_features.head()

# Заполним отсутствующие данные медианными значениями
new_train_features = new_features
new_features = imp_mean.fit_transform(new_train_features)
new_features = pd.DataFrame(new_features)
new_features.columns = new_train_features.columns

client_100 = pd.DataFrame()
client_100 = new_features[:201]
client_100.shape
client_100.head()

y = client_100['REP_CLID']
x = client_100.drop(columns = ['REP_CLID'])
exit_data = pd.DataFrame(columns = ['CLID','PRED','PROB'])

for i in range(len(y)):
    z = logreg.predict(x[i:i+1])
    proba = logreg.predict_proba(x[i:i+1])
    exit_data.loc[i,'CLID'] = (y[i])
    exit_data.loc[i,'PRED'] = z[0].astype(int)
    exit_data.loc[i,'PROB'] = proba

exit_data.shape
exit_data.head()
part_rent_clients(exit_data,exit_data['PRED'])

print(exit_data)
print(exit_data[exit_data['PRED']==0])

exit_data.to_csv('/home/anton/Projects/python/development/9_rent_house/check_property/ready300000.csv', sep='\t', encoding='utf-8')
#exit_data.to_csv('D:/Models/development/9_rent_house/check_property/ready300000.csv', sep='\t', encoding='utf-8')

