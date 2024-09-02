from argparse import ArgumentParser
from pathlib import Path
import pandas as pd
import json
import numpy as np

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import ExtraTreesClassifier
import xgboost as xgb


from database import Database, filter_hyposmia
from train_functions import prepare_experiment, test_nopipeline

parser = ArgumentParser()
parser.add_argument('--clinical', type=str, default="data/processed/clinical_processed.csv")
parser.add_argument('--remote', type=str, default="data/processed/remote_processed.csv")
parser.add_argument('--config_file', type=str, default="config_files/config.json")
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--age_sex', type=bool, default=False)
parser.add_argument('--impute', type=bool, default=False)

VARIABLES_TO_REMOVE = []#["AGE","BIRTHSEX"]

all_scent = [
'SCENT_01_CORRECT', 'SCENT_01_RESPONSE', 'SCENT_02_CORRECT', 'SCENT_02_RESPONSE', 'SCENT_03_CORRECT', 'SCENT_03_RESPONSE', 'SCENT_04_CORRECT', 'SCENT_04_RESPONSE', 'SCENT_05_CORRECT', 'SCENT_05_RESPONSE', 'SCENT_06_CORRECT', 'SCENT_06_RESPONSE', 'SCENT_07_CORRECT', 'SCENT_07_RESPONSE', 'SCENT_08_CORRECT', 'SCENT_08_RESPONSE', 'SCENT_09_CORRECT', 'SCENT_09_RESPONSE', 'SCENT_10_CORRECT', 'SCENT_10_RESPONSE', 'SCENT_11_CORRECT', 'SCENT_11_RESPONSE', 'SCENT_12_CORRECT', 'SCENT_12_RESPONSE', 'SCENT_13_CORRECT', 'SCENT_13_RESPONSE', 'SCENT_14_CORRECT', 'SCENT_14_RESPONSE', 'SCENT_15_CORRECT', 'SCENT_15_RESPONSE', 'SCENT_16_CORRECT', 'SCENT_16_RESPONSE', 'SCENT_17_CORRECT', 'SCENT_17_RESPONSE', 'SCENT_18_CORRECT', 'SCENT_18_RESPONSE', 'SCENT_19_CORRECT', 'SCENT_19_RESPONSE', 'SCENT_20_CORRECT', 'SCENT_20_RESPONSE', 'SCENT_21_CORRECT', 'SCENT_21_RESPONSE', 'SCENT_22_CORRECT', 'SCENT_22_RESPONSE', 'SCENT_23_CORRECT', 'SCENT_23_RESPONSE', 'SCENT_24_CORRECT', 'SCENT_24_RESPONSE', 'SCENT_25_CORRECT', 'SCENT_25_RESPONSE', 'SCENT_26_CORRECT', 'SCENT_26_RESPONSE', 'SCENT_27_CORRECT', 'SCENT_27_RESPONSE', 'SCENT_28_CORRECT', 'SCENT_28_RESPONSE', 'SCENT_29_CORRECT', 'SCENT_29_RESPONSE', 'SCENT_30_CORRECT', 'SCENT_30_RESPONSE', 'SCENT_31_CORRECT', 'SCENT_31_RESPONSE', 'SCENT_32_CORRECT', 'SCENT_32_RESPONSE', 'SCENT_33_CORRECT', 'SCENT_33_RESPONSE', 'SCENT_34_CORRECT', 'SCENT_34_RESPONSE', 'SCENT_35_CORRECT', 'SCENT_35_RESPONSE', 'SCENT_36_CORRECT', 'SCENT_36_RESPONSE', 'SCENT_37_CORRECT', 'SCENT_37_RESPONSE', 'SCENT_38_CORRECT', 'SCENT_38_RESPONSE', 'SCENT_39_CORRECT', 'SCENT_39_RESPONSE', 'SCENT_40_CORRECT', 'SCENT_40_RESPONSE'
]
correct = [col for col in all_scent if "CORRECT" in col]
response = [col for col in all_scent if "RESPONSE" in col]

var_dict = {'all': all_scent, 'correct': correct, 'response': response}

if __name__ == '__main__':
    args = parser.parse_args()
    
    seed = args.seed
    np.random.seed(seed)
    
    if args.impute == True:
        args.clinical = "data/processed/clinical_processed_imputed.csv"
    clinical_dir = Path(args.clinical)
    remote_dir = Path(args.remote)
    config_file = Path(args.config_file)

    MODELS = {
        #   'RidgeClassifier': RidgeClassifier(random_state=seed),
          'RandomForestClassifier': RandomForestClassifier(random_state=seed),
          'XGBClassifier': xgb.XGBClassifier(eval_metric='auc', objective='binary:logistic', seed=seed, n_estimators=100, max_depth=3),
        #   'SVC': SVC(random_state=seed),
        #   'KNeighborsClassifier': KNeighborsClassifier(n_neighbors=5),
        #   'DecisionTreeClassifier': DecisionTreeClassifier(random_state=seed),
          'GradientBoostingClassifier': GradientBoostingClassifier(random_state=seed),
          'ExtraTreesClassifier': ExtraTreesClassifier(random_state=seed)
          } 

    data = Database(clinical_dir, remote_dir, False)
    data.prepare_data()

    #get data with target
    data_all = data.get_data()
    
    print(len(data_all))
    
    #TODO still use JSON configuration file??
    # Load the JSON configuration file
    with open(config_file, 'r') as file:
        config = json.load(file)
    
    results = []

    
    for experiment in config["experiments"]:
        print(f"\nRunning Experiment: {experiment['name']}")

        # Apply preprocessing methods
        hyposmia_cutoff = experiment['preprocessing']['cutoff']
        preprocessing_methods = experiment["preprocessing"]["methods"]
        variables_to_select = var_dict[experiment["preprocessing"]["variables"]]
        #append AGE and BIRTHSEX
        if args.age_sex == True:
            if "AGE" not in variables_to_select:
                variables_to_select.append("AGE")
            if "BIRTHSEX" not in variables_to_select:
                variables_to_select.append("BIRTHSEX")
        log = experiment["preprocessing"]["log"]
        
        #first, filter the data according to the cutoff
        data_experiment = filter_hyposmia(data_all, hyposmia_cutoff)
    
        if not variables_to_select:
            X_train, y_train = prepare_experiment(data_experiment.copy(), 'target', VARIABLES_TO_REMOVE, log_transform=log, seed=seed, train=True, hyposmia_cutoff=hyposmia_cutoff, impute = args.impute)
            X_test, y_test = prepare_experiment(data_experiment.copy(), 'target', VARIABLES_TO_REMOVE, log_transform=log, seed=seed, train=False, hyposmia_cutoff=hyposmia_cutoff, impute = args.impute)
        else:
            X_train, y_train = prepare_experiment(data_experiment.copy(), 'target', VARIABLES_TO_REMOVE, variables_to_select, log_transform=log, seed=seed, train=True, hyposmia_cutoff=hyposmia_cutoff, impute = args.impute)
            X_test, y_test = prepare_experiment(data_experiment.copy(), 'target', VARIABLES_TO_REMOVE, variables_to_select, log_transform=log, seed=seed, train=False, hyposmia_cutoff=hyposmia_cutoff, impute = args.impute)
        
        columns = X_train.columns
        
        # Apply specified preprocessing methods
        for method in preprocessing_methods:
            if method == "StandardScaler":
                preprocess = StandardScaler()
            elif method == "MinMaxScaler":
                preprocess = MinMaxScaler()
                
        # Iterate over models
        for model_name, model in MODELS.items():
            #check if model has random_state
            if model_name in ['RandomForestClassifier', 'XGBClassifier', 'SVC', 'DecisionTreeClassifier', 'GradientBoostingClassifier', 'ExtraTreesClassifier']:
                model.set_params(random_state=seed)
            
            print(f"\nRunning Model: {model_name}")
            model, exp_results = test_nopipeline(model, X_train, y_train, X_test, y_test, preprocess=preprocess, show=True, return_model=True)
            
            # Save results to a csv file
            results.append({
                "experiment": experiment['name'],
                "cutoff": hyposmia_cutoff,
                "preprocess": preprocessing_methods,
                "model": model_name,
                #accuracy with only 3 digits
                "accuracy": exp_results['test_accuracy'],
                "specificity": exp_results['test_specificity'],
                "sensitivity": np.sqrt(exp_results['test_sensitivity']),
                "precision": exp_results['test_precision'],
                "f1": exp_results['test_f1'],
                "instances": len(y_train),
                "columns": list(columns) 
            })

    # save results to a csv file
    results_df = pd.DataFrame(results)
    #sort by accuracy, and keep only 4 digits after the comma
    results_df = results_df.sort_values(by='accuracy', ascending=False)
    results_df = results_df.round(4)
    
    if args.age_sex == True:
        results_df.to_csv(f"results/test_{hyposmia_cutoff}_seed{seed}_age&sex_{args.impute}.csv", index=False)
    else:
        results_df.to_csv(f"results/test_{hyposmia_cutoff}_seed{seed}_{args.impute}.csv", index=False)