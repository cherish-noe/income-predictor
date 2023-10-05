# Import required libraries
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv('adult.csv')
print(df.head())


# Filling missing values
col_names = df.columns
for c in col_names:
    df = df.replace("?", np.NaN)
df = df.apply(lambda x: x.fillna(x.value_counts().index[0]))

# Discretisation
df.replace(['Divorced', 'Married-AF-spouse', 
            'Married-civ-spouse', 'Married-spouse-absent', 
            'Never-married', 'Separated', 'Widowed'],
            ['divorced', 'married', 'married', 'married',
            'not married', 'not married', 'not married'], inplace=True)

# Label Encoder
category_col = ['workclass', 'race', 'education', 'marital-status', 'occupation',
                'relationship', 'gender', 'native-country', 'income']
labelEconder = preprocessing.LabelEncoder()

# Create a map of all the numerical values of each categorical labels
mapping_dict = {}
for col in category_col:
    df[col] = labelEconder.fit_transform(df[col])
    le_name_mapping = dict(zip(labelEconder.classes_, labelEconder.transform(labelEconder.classes_)))
    mapping_dict[col] = le_name_mapping
print(mapping_dict)

# Drop redundant columns
df = df.drop(['fnlwgt', 'educational-num'], axis=1)

X = df.values[:, 0:12]
y = df.values[:, 12]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100)
dt_clf_gini = DecisionTreeClassifier(criterion="gini", random_state=100, max_depth=5, min_samples_leaf=5)
dt_clf_gini.fit(X_train, y_train)
y_pred_gini = dt_clf_gini.predict(X_test)

print("Decision Tree using Gini Index\nAccuracy is ", accuracy_score(y_test, y_pred_gini)*100)

# Save the model
import pickle
pickle.dump(dt_clf_gini, open("model.pkl", "wb"))