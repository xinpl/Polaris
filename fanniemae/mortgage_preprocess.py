import sys
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
import seaborn as sns
import pandas as pd
import numpy as np

import tensorflow as tf

import matplotlib.pyplot as mp

DROP_THRE = 0.3

USE_CLDS = False

def dropnan(df):
    return df.dropna()

def fillnan(df):
    columns = df.columns[df.isnull().any()]
    for name in columns:
        y = df.loc[df[name].notnull(), name].values
        X = df.loc[df[name].notnull()].drop(columns, axis=1).values
        X_test = df.loc[df[name].isnull()].drop(columns, axis=1).values
        if df[name].dtypes == 'object':
            model = RandomForestClassifier(n_estimators=400, max_depth=3)
            model.fit(X, y)
            df.loc[df[name].isnull(), name] = model.predict(X_test)
        else:
            model = RandomForestRegressor(n_estimators=400, max_depth=3)
            model.fit(X, y)
            df.loc[df[name].isnull(), name] = model.predict(X_test)
    return df

def getdummies(df, one_hot):
    columns = df.columns[df.isnull().any()]
    nan_cols = df[columns]

    df.drop(nan_cols.columns, axis=1, inplace=True)

    cat = df.select_dtypes(include=['object'])
    num = df.drop(cat.columns, axis=1)

    data = pd.DataFrame()
    for i in cat.columns:
        tmp = pd.get_dummies(cat[i], drop_first=True)
        data = pd.concat([data, tmp], axis=1)

    one_hot.extend(data.columns.values)

    df = pd.concat([num,data,nan_cols], axis=1).reset_index(drop=True)
    return df

col_acq = ['LoanID','Channel','SellerName','OrInterestRate','OrUnpaidPrinc','OrLoanTerm',
        'OrDate','FirstPayment','OrLTV','OrCLTV','NumBorrow','DTIRat','CreditScore',
        'FTHomeBuyer','LoanPurpose','PropertyType','NumUnits','OccStatus','PropertyState',
        'Zip','MortInsPerc','ProductType','CoCreditScore','MortInsType','RelMortInd'];

col_per = ['LoanID','MonthRep','Servicer','CurrInterestRate','CAUPB','LoanAge','MonthsToMaturity',
        'AdMonthsToMaturity','MaturityDate','MSA','CLDS','ModFlag','ZeroBalCode','ZeroBalDate',
        'LastInstallDate','ForeclosureDate','DispositionDate','ForeclosureCosts','PPRC','AssetRecCost','MHRC',
        'ATFHP','NetSaleProceeds','CreditEnhProceeds','RPMWP','OFP','NIBUPB','PFUPB','RMWPF',
        'FPWA','ServicingIndicator'];

one_hot = ['Default']

def main(acq_path, per_path, out_path):
    df_acq = pd.read_csv(acq_path, sep='|', names=col_acq, index_col=False)
    if USE_CLDS :
        df_per = pd.read_csv(per_path, sep='|', names=col_per, usecols=[0,10,15], index_col=False)
    else:
        df_per = pd.read_csv(per_path, sep='|', names=col_per, usecols=[0,15], index_col=False)


    df_per.rename(index=str, columns={"ForeclosureDate": 'Default'}, inplace=True)

    df_per['Default'].fillna(0, inplace=True)
    df_per.loc[df_per['Default'] != 0, 'Default'] = 1

    if USE_CLDS:
        df_per['CLDS'].fillna(0,inplace=True)
        df_per.loc[df_per['CLDS'] == 'X', 'CLDS'] = 0
        df_per['CLDS'] = df_per['CLDS'].astype(int)
        df_per.loc[df_per['CLDS'] > 0, 'Default'] = 1
        df_per.drop(['CLDS'], axis = 1, inplace = True)
        df_per.sort_values('Default', inplace = True)

    df_acq.drop_duplicates(subset='LoanID', keep='last', inplace=True)
    df_per.drop_duplicates(subset='LoanID', keep='last', inplace=True)
    df_per = df_per.sample(frac = 1.0)

    print(df_per.groupby('Default').size())

    df = pd.merge(df_acq, df_per, on='LoanID', how='inner')

    print(df.groupby('Default').size())

    df['Default'] = df['Default'].astype(int)

    miss_rates = df.apply(lambda x: x.isnull().sum() / float(df.shape[0]), axis=0)

    print(miss_rates)

    print(df.columns.values)

    df.drop(['LoanID'], axis = 1, inplace = True)

    to_be_dropped = ['MortInsPerc', 'CoCreditScore', 'MortInsType']

    #for indx, v in miss_rates.iteritems():
    #    if v > DROP_THRE:
    #        to_be_dropped.append(indx)

    print("Drop the following columns: "+str(to_be_dropped)+"\n")

    df.drop(to_be_dropped, axis=1, inplace=True)

    print(df.columns.values)

    df = dropnan(df)

    y = df['Default'].values
    X = df.drop(['Default'], axis=1).values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = int((0.25*len(y))), random_state=0)

    feat_labels = list(df.drop('Default', axis=1).columns.values)

    feat_labels.append('Default')

    train_table = X_train[:]

    train_table = np.append(train_table,[ [e] for e in y_train], axis=1)

    train_df = pd.DataFrame(data=train_table, columns = feat_labels)

    print("Training set distribution.\n")
    print(train_df.groupby("Default").size())

    for col_name in one_hot:
        train_df[col_name] = train_df[col_name].astype(int)

    train_df.to_csv(out_path+".train", index=False)

    test_table = X_test[:]

    test_table = np.append(test_table, [[e] for e in y_test], axis=1)

    test_df = pd.DataFrame(data=test_table, columns = feat_labels)

    for col_name in one_hot:
        test_df[col_name] = test_df[col_name].astype(int)

    test_df.to_csv(out_path+".test", index=False)

if __name__ == '__main__':
    acq_path = sys.argv[1]
    per_path = sys.argv[2]
    out_path = sys.argv[3]
    main(acq_path,per_path,out_path)