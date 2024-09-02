import pandas as pd
import numpy as np
import os

from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.metrics import make_scorer, accuracy_score, recall_score, precision_score, f1_score, confusion_matrix
from imblearn.under_sampling import RandomUnderSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import StratifiedKFold, LeaveOneOut

all_scent = [
'SCENT_01_CORRECT', 'SCENT_01_RESPONSE', 'SCENT_02_CORRECT', 'SCENT_02_RESPONSE', 'SCENT_03_CORRECT', 'SCENT_03_RESPONSE', 'SCENT_04_CORRECT', 'SCENT_04_RESPONSE', 'SCENT_05_CORRECT', 'SCENT_05_RESPONSE', 'SCENT_06_CORRECT', 'SCENT_06_RESPONSE', 'SCENT_07_CORRECT', 'SCENT_07_RESPONSE', 'SCENT_08_CORRECT', 'SCENT_08_RESPONSE', 'SCENT_09_CORRECT', 'SCENT_09_RESPONSE', 'SCENT_10_CORRECT', 'SCENT_10_RESPONSE', 'SCENT_11_CORRECT', 'SCENT_11_RESPONSE', 'SCENT_12_CORRECT', 'SCENT_12_RESPONSE', 'SCENT_13_CORRECT', 'SCENT_13_RESPONSE', 'SCENT_14_CORRECT', 'SCENT_14_RESPONSE', 'SCENT_15_CORRECT', 'SCENT_15_RESPONSE', 'SCENT_16_CORRECT', 'SCENT_16_RESPONSE', 'SCENT_17_CORRECT', 'SCENT_17_RESPONSE', 'SCENT_18_CORRECT', 'SCENT_18_RESPONSE', 'SCENT_19_CORRECT', 'SCENT_19_RESPONSE', 'SCENT_20_CORRECT', 'SCENT_20_RESPONSE', 'SCENT_21_CORRECT', 'SCENT_21_RESPONSE', 'SCENT_22_CORRECT', 'SCENT_22_RESPONSE', 'SCENT_23_CORRECT', 'SCENT_23_RESPONSE', 'SCENT_24_CORRECT', 'SCENT_24_RESPONSE', 'SCENT_25_CORRECT', 'SCENT_25_RESPONSE', 'SCENT_26_CORRECT', 'SCENT_26_RESPONSE', 'SCENT_27_CORRECT', 'SCENT_27_RESPONSE', 'SCENT_28_CORRECT', 'SCENT_28_RESPONSE', 'SCENT_29_CORRECT', 'SCENT_29_RESPONSE', 'SCENT_30_CORRECT', 'SCENT_30_RESPONSE', 'SCENT_31_CORRECT', 'SCENT_31_RESPONSE', 'SCENT_32_CORRECT', 'SCENT_32_RESPONSE', 'SCENT_33_CORRECT', 'SCENT_33_RESPONSE', 'SCENT_34_CORRECT', 'SCENT_34_RESPONSE', 'SCENT_35_CORRECT', 'SCENT_35_RESPONSE', 'SCENT_36_CORRECT', 'SCENT_36_RESPONSE', 'SCENT_37_CORRECT', 'SCENT_37_RESPONSE', 'SCENT_38_CORRECT', 'SCENT_38_RESPONSE', 'SCENT_39_CORRECT', 'SCENT_39_RESPONSE', 'SCENT_40_CORRECT', 'SCENT_40_RESPONSE'
]
correct = [col for col in all_scent if "CORRECT" in col]
response = [col for col in all_scent if "RESPONSE" in col]

def stratify_dataset_split(X, y, v1='target', v2=False, seed=42):
    #this function separates the data into bins and use them to stratify the dataset 
    #FIX FOR V2
    X_train, X_test, y_train, y_test =  train_test_split(X, y, test_size=0.2, random_state=seed, stratify=v1)
    
    return X_train, X_test, y_train, y_test

def prepare_experiment(df, target_column, remove_columns, select_columns=None, log_transform=False, seed = 42, train=True, hyposmia_cutoff=30, impute = False, fs = False):
    df = df.copy()
    
    rus = RandomUnderSampler(random_state=seed)

    df.drop(columns=remove_columns, inplace=True)
    y = df[target_column]
    X = df.drop(columns=[target_column])
    
    print(y.value_counts())
    
    print(f"Original shapes: {X.shape}, {y.shape}")
    #print unique values from y
    print(f"Unique values in y: {np.unique(y, return_counts=True)}")
    
    #if hyposmia is 2023, use already split data
    if hyposmia_cutoff in [2023, '2023']:
        if impute:
            print("Using already stratified data for age, sex and target with imput data")

            #load
            train_data = pd.read_csv('data/processed/train_data_hyposmia2023.csv')
            test_data = pd.read_csv('data/processed/test_data_hyposmia2023.csv')
            
            #X_train are patnos that are X Patnos that are in train_data
            X_train = X[X['PATNO'].isin(train_data['PATNO'])]
            y_train = y[X_train.index]
            X_test = X[X['PATNO'].isin(test_data['PATNO'])]
            y_test = y[X_test.index]
        else:
            print("Using already stratified data for age, sex and target WITHOUT imput data")
            #load
            train_data = pd.read_csv('data/processed/train_data_without_imput.csv')
            test_data = pd.read_csv('data/processed/test_data_without_imput.csv')
            
            #X_train are patnos that are X Patnos that are in train_data
            X_train = X[X['PATNO'].isin(train_data['PATNO'])]
            y_train = y[X_train.index]
            X_test = X[X['PATNO'].isin(test_data['PATNO'])]
            y_test = y[X_test.index]
            
    elif hyposmia_cutoff in [None, 'None']:
        print("Using already stratified data for age, sex and target")
        #CREATE new variable str with target and birthsex for stratification
        #load
        train_data = pd.read_csv('data/processed/train_data_all.csv')
        test_data = pd.read_csv('data/processed/test_data_all.csv')
        
        #X_train are patnos that are X Patnos that are in train_data
        X_train = X[X['PATNO'].isin(train_data['PATNO'])]
        y_train = y[X_train.index]
        X_test = X[X['PATNO'].isin(test_data['PATNO'])]
        y_test = y[X_test.index]
        
    elif hyposmia_cutoff == "SAA":
       
        #just split the data
        X_train, X_test, y_train, y_test = stratify_dataset_split(df, y, v1=y, v2=None, seed=seed)
    
    else:
    
        gender_column = 'BIRTHSEX'
        
        #create a specific df where target is 0 and another where target is 1
        df_0 = df[df[target_column] == 0]
        df_1 = df[df[target_column] == 1]
        
        #Get the counts of 'BIRTHSEX' in df_1
        counts_df_1 = df_1[gender_column].value_counts()

        # Initialize the RandomUnderSampler with a dictionary that specifies the number of instances for each 'BIRTHSEX'
        rus = RandomUnderSampler(random_state=seed, sampling_strategy={0: counts_df_1[0], 1: counts_df_1[1]})

        # Perform undersampling on df_0
        df_0_resampled, _ = rus.fit_resample(df_0, df_0[gender_column])

        # Concatenate df_0_resampled and df_1
        df_resampled = pd.concat([df_0_resampled, df_1])   
        y_resampled = df_resampled[target_column]
        
        #CREATE new variable str with target and birthsex for stratification
        sex_target = df_resampled[target_column].astype(str) + '_' + df_resampled['BIRTHSEX'].astype(str)
        
        #df_resampled.drop(columns=[target_column], inplace=True)
        
        #print shapes
        print(f"Resampled shapes: {df_resampled.shape}, {y_resampled.shape}")
        
        # v1 is target, v2 is whathever we want. If v2 is None, it will be ignored
        X_train, X_test, y_train, y_test = stratify_dataset_split(df_resampled, y_resampled, v1=sex_target, v2=None, seed=seed)
        
    #print all shapes
    print(f"Train shapes: {X_train.shape}, {y_train.shape}")
    print(f"Test shapes: {X_test.shape}, {y_test.shape}")
    
    print(f"Unique values of BIRTHSEX in X_train: {np.unique(X_train['BIRTHSEX'], return_counts=True)}")
    print(f"Unique values of BIRTHSEX in X_test: {np.unique(X_test['BIRTHSEX'], return_counts=True)}")
    
    if train:
        X = X_train
        y = y_train
    else:
        X = X_test
        y = y_test
    
    if select_columns:
        X = X[select_columns]
        
    #if AGE is in the columns, round it:
    if 'AGE' in X.columns:
        X['AGE'] = X['AGE'].round()
        
    #check if there are columns ending with _RESPONSE
    if any([col for col in X.columns if "RESPONSE" in col]):
        #one hot encode the response columns
        X = pd.get_dummies(X, columns=response)
    #binary encode the correct columns
    if any([col for col in X.columns if "CORRECT" in col]):
        #check which correct columns are in the data
        correct_columns = [col for col in X.columns if "CORRECT" in col]
        X[correct_columns] = X[correct_columns].astype(bool)
    
    if log_transform:
        y = np.log(y)
        
    return X, y

def specificity_score(y_true, y_pred):
    """Calculate specificity (true negative rate)."""
    tn = ((y_true == 0) & (y_pred == 0)).sum()
    fp = ((y_true == 0) & (y_pred == 1)).sum()
    return tn / (tn + fp)

def train_and_evaluate(model, X_train, y_train, preprocess=None, show=True, train=False):
    model_name = model.__class__.__name__

    if preprocess:
        model = make_pipeline(preprocess, model)

    loo = LeaveOneOut()
    
    # Initialize counters for TP, FP, TN, FN
    TP = FP = TN = FN = 0
    labels = [0, 1]  # Adjust according to the known labels in your dataset

    for train_index, test_index in loo.split(X_train):
        # Use .iloc for position-based indexing
        X_train_fold, X_test_fold = X_train.iloc[train_index], X_train.iloc[test_index]
        y_train_fold, y_test_fold = y_train.iloc[train_index], y_train.iloc[test_index]
        
        model.fit(X_train_fold, y_train_fold)
        y_pred = model.predict(X_test_fold)

        tn, fp, fn, tp = confusion_matrix(y_test_fold, y_pred, labels=labels).ravel()

        TP += tp
        FP += fp
        TN += tn
        FN += fn

    # Compute final metrics
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    specificity = TN / (TN + FP)
    sensitivity = TP / (TP + FN)  # Sensitivity is the same as recall
    precision = TP / (TP + FP)
    f1 = 2 * TP / (2 * TP + FP + FN)

    if show:
        print('{} : Accuracy: {:.3f} | Specificity: {:.3f} | Sensitivity: {:.3f} | Precision: {:.3f} | F1: {:.3f}'.format(
            model_name, accuracy, specificity, sensitivity, precision, f1))
    
    return {
        'accuracy': accuracy,
        'specificity': specificity,
        'sensitivity': sensitivity,
        'precision': precision,
        'f1': f1
    }

# Updated train_and_evaluate function
def old_train_and_evaluate(model, X_train, y_train, preprocess=None, cv=5, show=True, train=False, random_state=None):
    model_name = model.__class__.__name__

    if preprocess:
        model = make_pipeline(preprocess, model)

    scoring = {
        'accuracy': make_scorer(accuracy_score),
        'specificity': make_scorer(specificity_score),
        'sensitivity': make_scorer(recall_score),
        'precision': make_scorer(precision_score),
        'f1': make_scorer(f1_score),
    }

    if isinstance(cv, int) and random_state is not None:
        cv = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)

    results = cross_validate(model, X_train, y_train, 
                             cv=cv, 
                             scoring=scoring,
                             return_train_score=train,
                             error_score='raise')
    
    if show and not train:
        print('{} : Accuracy: {:.3f} | Specificity: {:.3f} | Sensitivity: {:.3f} | Precision: {:.3f} | F1: {:.3f}'.format(
            model_name,
            results['test_accuracy'].mean(),
            results['test_specificity'].mean(),
            results['test_sensitivity'].mean(),
            results['test_precision'].mean(),
            results['test_f1'].mean()))
        
    if show and train:
        print('{} Train : Accuracy: {:.3f} | Specificity: {:.3f} | Sensitivity: {:.3f} | Precision: {:.3f} | F1: {:.3f}'.format(
            model_name,
            results['train_accuracy'].mean(),
            results['train_specificity'].mean(),
            results['train_sensitivity'].mean(),
            results['train_precision'].mean(),
            results['train_f1'].mean()))
        print("{} Test : Accuracy: {:.3f} | Specificity: {:.3f} | Sensitivity: {:.3f} | Precision: {:.3f} | F1: {:.3f}".format(
            model_name,
            results['test_accuracy'].mean(),
            results['test_specificity'].mean(),
            results['test_sensitivity'].mean(),
            results['test_precision'].mean(),
            results['test_f1'].mean()))    
    
    return results

def test(model, X_train, y_train, X_test, y_test, preprocess=None, show=True, return_model=False):
    model_name = model.__class__.__name__

    if preprocess:
        model = make_pipeline(preprocess, model)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    specificity = specificity_score(y_test, y_pred)
    sensitivity = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    if show:
        print('{} : Accuracy: {:.3f} | Specificity: {:.3f} | Sensitivity: {:.3f} | Precision: {:.3f} | F1: {:.3f}'.format(
            model_name,
            accuracy,
            specificity,
            sensitivity,
            precision,
            f1))
    
    results = {
        'test_accuracy': accuracy,
        'test_specificity': specificity,
        'test_sensitivity': sensitivity,
        'test_precision': precision,
        'test_f1': f1,
    }
    
    if return_model:
        return model, results
    else:
        return results
    

def test_nopipeline(model, X_train, y_train, X_test, y_test, preprocess=None, show=True, return_model=False):
    model_name = model.__class__.__name__

    #check if there are missing columns in X_test 
    missing_features = set(X_train.columns) - set(X_test.columns)
    for feature in missing_features:
        X_test[feature] = 0
    #match order
    X_test = X_test[X_train.columns]
    
    if preprocess:
        # Fit the preprocess step on the training data
        preprocess.fit(X_train)
        # Transform both training and test data
        X_train = preprocess.transform(X_train)
        X_test = preprocess.transform(X_test)

    # Fit the model on the preprocessed training data
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    #verify if folder exists
    if not os.path.exists("results/predictions"):
        os.makedirs("results/predictions")
        
    #save the predictions 
    np.savetxt(f"results/predictions/{model_name}_predictions.csv", y_pred, delimiter=",")
    
    accuracy = accuracy_score(y_test, y_pred)
    specificity = specificity_score(y_test, y_pred)
    sensitivity = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    if show:
        print('{} : Accuracy: {:.3f} | Specificity: {:.3f} | Sensitivity: {:.3f} | Precision: {:.3f} | F1: {:.3f}'.format(
            model_name,
            accuracy,
            specificity,
            sensitivity,
            precision,
            f1))
    
    results = {
        'test_accuracy': accuracy,
        'test_specificity': specificity,
        'test_sensitivity': sensitivity,
        'test_precision': precision,
        'test_f1': f1
    }
    
    if return_model:
        return model, results
    else:
        return results