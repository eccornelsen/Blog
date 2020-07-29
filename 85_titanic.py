import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier

def main():

    train_data = pd.read_csv('datasets/train.csv')
    test_data = pd.read_csv('datasets/test.csv')

    df = train_data.copy()
    dg = test_data.copy()

    y = df['Survived']
    features = ['Pclass','Sex','SibSp','Parch']
    Xo = df[features].fillna(-1)
    print('Print subset of X and the data types of the selected features:','\n',Xo.head(10), '\n')
    print(Xo.dtypes,'\n\n')

    X = pd.get_dummies(df[features].fillna(-1))
    print('Print subset of X using pd.get_dummies on the gender column, and the data types of the selected features:','\n',X.head(10),'\n')
    print(X.dtypes, '\n\n')
    X_test = pd.get_dummies(dg[features].fillna(-1))
    
    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
    model.fit(X, y)
    predictions = model.predict(X_test)
    pd.DataFrame({'PassengerId': dg.PassengerId, 'Survived': predictions}).to_csv('datasets/my_submission'+str(1)+'.csv', index=False)

    Xc =df[features].fillna(-1)
    Xc['Sex'] = Xc['Sex'].astype('category').cat.codes
    print('Print subset of X using df.astype(''category'').cat.codes on the gender column, and the data types of the selected features:','\n',Xc.head(10),'\n')
    print(Xc.dtypes, '\n\n')
    X_test2 = dg[features].fillna(-1)
    X_test2['Sex'] = X_test2['Sex'].astype('category').cat.codes

    model2 = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
    model2.fit(Xc, y)
    predictions2 = model2.predict(X_test2)
    pd.DataFrame({'PassengerId': dg.PassengerId, 'Survived': predictions2}).to_csv('datasets/my_submission'+str(2)+'.csv', index=False)



if __name__ == '__main__':
    main()
