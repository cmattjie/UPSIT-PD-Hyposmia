# Use SHAP values and other tools for interpretability of ML models
import pandas as pd
import os
import json
import matplotlib.pyplot as plt
from database import Database, filter_hyposmia
from train_functions import prepare_experiment,test_nopipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import xgboost as xgb
import numpy as np
from xgboost import plot_tree as xgb_plot_tree
from sklearn.tree import plot_tree, export_graphviz
import graphviz
import shap
from pathlib import Path
from matplotlib.colors import LinearSegmentedColormap

#create all these args variables directly
args_clinical = "data/processed/clinical_processed_imputed.csv"
args_remote = "data/processed/remote_processed.csv"
args_config_file = "config_files/config.json"
args_seed = 42
args_age_sex = [True, False]
args_impute = True

all_scent = [
'SCENT_01_CORRECT', 'SCENT_01_RESPONSE', 'SCENT_02_CORRECT', 'SCENT_02_RESPONSE', 'SCENT_03_CORRECT', 'SCENT_03_RESPONSE', 'SCENT_04_CORRECT', 'SCENT_04_RESPONSE', 'SCENT_05_CORRECT', 'SCENT_05_RESPONSE', 'SCENT_06_CORRECT', 'SCENT_06_RESPONSE', 'SCENT_07_CORRECT', 'SCENT_07_RESPONSE', 'SCENT_08_CORRECT', 'SCENT_08_RESPONSE', 'SCENT_09_CORRECT', 'SCENT_09_RESPONSE', 'SCENT_10_CORRECT', 'SCENT_10_RESPONSE', 'SCENT_11_CORRECT', 'SCENT_11_RESPONSE', 'SCENT_12_CORRECT', 'SCENT_12_RESPONSE', 'SCENT_13_CORRECT', 'SCENT_13_RESPONSE', 'SCENT_14_CORRECT', 'SCENT_14_RESPONSE', 'SCENT_15_CORRECT', 'SCENT_15_RESPONSE', 'SCENT_16_CORRECT', 'SCENT_16_RESPONSE', 'SCENT_17_CORRECT', 'SCENT_17_RESPONSE', 'SCENT_18_CORRECT', 'SCENT_18_RESPONSE', 'SCENT_19_CORRECT', 'SCENT_19_RESPONSE', 'SCENT_20_CORRECT', 'SCENT_20_RESPONSE', 'SCENT_21_CORRECT', 'SCENT_21_RESPONSE', 'SCENT_22_CORRECT', 'SCENT_22_RESPONSE', 'SCENT_23_CORRECT', 'SCENT_23_RESPONSE', 'SCENT_24_CORRECT', 'SCENT_24_RESPONSE', 'SCENT_25_CORRECT', 'SCENT_25_RESPONSE', 'SCENT_26_CORRECT', 'SCENT_26_RESPONSE', 'SCENT_27_CORRECT', 'SCENT_27_RESPONSE', 'SCENT_28_CORRECT', 'SCENT_28_RESPONSE', 'SCENT_29_CORRECT', 'SCENT_29_RESPONSE', 'SCENT_30_CORRECT', 'SCENT_30_RESPONSE', 'SCENT_31_CORRECT', 'SCENT_31_RESPONSE', 'SCENT_32_CORRECT', 'SCENT_32_RESPONSE', 'SCENT_33_CORRECT', 'SCENT_33_RESPONSE', 'SCENT_34_CORRECT', 'SCENT_34_RESPONSE', 'SCENT_35_CORRECT', 'SCENT_35_RESPONSE', 'SCENT_36_CORRECT', 'SCENT_36_RESPONSE', 'SCENT_37_CORRECT', 'SCENT_37_RESPONSE', 'SCENT_38_CORRECT', 'SCENT_38_RESPONSE', 'SCENT_39_CORRECT', 'SCENT_39_RESPONSE', 'SCENT_40_CORRECT', 'SCENT_40_RESPONSE'
]
correct = [col for col in all_scent if "CORRECT" in col]
response = [col for col in all_scent if "RESPONSE" in col]

var_dict = {'all': all_scent, 'correct': correct, 'response': response}

#blue and red cmap
cmap = LinearSegmentedColormap.from_list('mycmap', ['red', 'blue'])

VARIABLES_TO_REMOVE = []#["AGE","BIRTHSEX"]

if not os.path.exists('images/trees'):
    os.makedirs('images/trees')

seed = args_seed
np.random.seed(seed)

if args_impute == True:
    args_clinical = "data/processed/clinical_processed_imputed.csv"
clinical_dir = Path(args_clinical)
remote_dir = Path(args_remote)
config_file = Path(args_config_file)

# Load the JSON configuration file
config_file = Path(args_config_file)
with open(config_file, 'r') as file:
    config = json.load(file)

data = Database(clinical_dir, remote_dir, False)
data.prepare_data()

# Create models dictionary
MODELS = {
    'RandomForestClassifier': RandomForestClassifier(random_state=seed),
    'XGBClassifier': xgb.XGBClassifier(eval_metric='auc', objective='binary:logistic', seed=seed, n_estimators=100, max_depth=3),
    'GradientBoostingClassifier': GradientBoostingClassifier(random_state=seed),
    'ExtraTreesClassifier': ExtraTreesClassifier(random_state=seed)
}

# Load feature names
names = pd.read_csv('data/scents/scent_correct_names.csv')
scent_names = [name.split('_')[1] for name in names.columns]
correct_dict = {name: names[f'SCENT_{name}_RESPONSE'][1] for name in scent_names}

# Load responses
scents = pd.read_csv('data/scents/scent_response_clean.csv')
#create dict using scent names as keys and the second row as values
response_dict = {}
for name in scents.columns:
    response_dict[name] = scents[name][0]

#remove first experiment from config
config["experiments"].pop(0)
for experiment in config["experiments"]:
    for age_sex in args_age_sex:
        data_all = data.get_data()
        hyposmia_cutoff = experiment['preprocessing']['cutoff']
        preprocessing_methods = experiment["preprocessing"]["methods"]
        variables_to_select = []
        variables_to_select = var_dict[experiment["preprocessing"]["variables"]]

        if age_sex == True:
            if "AGE" not in variables_to_select:
                variables_to_select.append("AGE")
            if "BIRTHSEX" not in variables_to_select:
                variables_to_select.append("BIRTHSEX")     
                
        else:
            if "AGE" in variables_to_select:
                variables_to_select.remove("AGE")
            if "BIRTHSEX" in variables_to_select:
                variables_to_select.remove("BIRTHSEX")
        
        log = experiment["preprocessing"]["log"]
        data_experiment = filter_hyposmia(data_all, hyposmia_cutoff)

        X_train, y_train = prepare_experiment(data_experiment.copy(), 'target', VARIABLES_TO_REMOVE, variables_to_select, log_transform=log, seed=seed, train=True, hyposmia_cutoff=hyposmia_cutoff, impute=args_impute)
        X_test, y_test = prepare_experiment(data_experiment.copy(), 'target', VARIABLES_TO_REMOVE, variables_to_select, log_transform=log, seed=seed, train=False, hyposmia_cutoff=hyposmia_cutoff, impute=args_impute)

        feature_names = X_train.columns.tolist()
        for i in range(len(feature_names)):
            for key in response_dict.keys():
                if key in feature_names[i]:
                    partial = feature_names[i].split('_')[:-1]
                    partial = '_'.join(partial)
                    feature_names[i] = feature_names[i].replace(key, f'{partial}_{response_dict[key]}')
            for key in correct_dict.keys():
                if key in feature_names[i]:
                    feature_names[i] = feature_names[i].replace(key, correct_dict[key])
            #remove "SCENT_" from feature names
            feature_names[i] = feature_names[i].replace("SCENT_", "")
            #remove "REPONSE" from feature names
            feature_names[i] = feature_names[i].replace("_RESPONSE", "")
            #remove spaces
            feature_names[i] = feature_names[i].replace(" ", "")
            
        #assert both train and test have the same number of features
        assert X_train.shape[1] == X_test.shape[1], "Mismatch between train and test features."
            
        #now we have the adapted feature names, rename the columns
        X_train.columns = feature_names
        X_test.columns = feature_names

        for method in preprocessing_methods:
            if method == "StandardScaler":
                preprocess = StandardScaler()
            elif method == "MinMaxScaler":
                preprocess = MinMaxScaler()
        
        shap_values_list = []
        
        for model_name, model in MODELS.items():
            if model_name in ['RandomForestClassifier', 'XGBClassifier', 'GradientBoostingClassifier', 
                              'ExtraTreesClassifier'
                              ]:
                model.set_params(random_state=seed)

            model, exp_results = test_nopipeline(model, X_train, y_train, X_test, y_test, preprocess=preprocess, show=True, return_model=True)

            accuracy = round(exp_results['test_accuracy'], 2)
            save_name = f'{hyposmia_cutoff}_{model_name}_{accuracy}_{vars}_{age_sex}'

            # Visualize decision trees
            if model_name == 'XGBClassifier':
                # Plot the first tree of the XGBoost model
                plt.figure(figsize=(20, 20))
                xgb_plot_tree(model, num_trees=0, rankdir='LR')
                plt.savefig(f'images/trees/{model_name}_tree_0_{save_name}.png', dpi=300, bbox_inches='tight')
                plt.clf()  # Clear the current figure

            elif model_name == 'GradientBoostingClassifier':
                # Plot the first tree of the GradientBoosting model
                # Plot the first tree of the GradientBoosting model
                # Plot the first tree of the GradientBoosting model
                # Export the tree to a DOT format
                # Export the tree to a DOT format
                tree_depth = model.estimators_[0, 0].get_depth()
                print(f"The depth of the first tree is: {tree_depth}")
                
                #
                
                # Iterate over each estimator in the ensemble
                for i, est in enumerate(model.estimators_.ravel()):
                    
                    dot_data = export_graphviz(
                        est,
                        out_file=None,
                        feature_names=feature_names,
                        filled=True,
                        rounded=True,
                        special_characters=True,
                        impurity=False
                    )

                    # Create a graph from DOT data
                    graph = graphviz.Source(dot_data)


                    output_dir = 'images/trees'
                    # Save the graph to a file
                    file_path = os.path.join(output_dir, f'tree_{i}')
                    graph.format = 'png'
                    graph.render(file_path)

                    print(f'Tree {i} saved as {file_path}.png')
                    
                    #clean plot
                    plt.clf()
                    
                    #clean graph
                    graph = None


            # X_test_renamed = X_test.copy()
            # X_test_renamed.columns = feature_names
            vars = experiment["preprocessing"]["variables"]
            #get accuracy from exp_results dictionary with 2 decimals
            
            #verify if folder exists
            if not os.path.exists('images/shap'):
                os.makedirs('images/shap')
                
            #clean model name
            clean_model_name = model_name.replace("Classifier", "")
            #insert space before CAPITAL LETTERS ONLY
            if clean_model_name == 'XGB':
                if age_sex == True:
                    clean_model_name = 'XGBoost Response_SA'
                else:
                    clean_model_name = 'XGBoost Response'
            elif clean_model_name == 'RandomForest':
                if age_sex == True:
                    clean_model_name = 'Random Forest Response_SA' 
                else:
                    clean_model_name = 'Random Forest Response'
            elif clean_model_name == 'GradientBoosting':
                if age_sex == True:
                    clean_model_name = 'Gradient Boost Response_SA'
                else:
                    clean_model_name = 'Gradient Boost Response'
            
            if model_name in ['RandomForestClassifier', 'ExtraTreesClassifier']:
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X_test)

                if isinstance(shap_values, list):
                    shap_values = shap_values[1]

                shap_values = np.array(shap_values)
                
                # Select SHAP values for the positive class (assuming binary classification)
                shap_values = shap_values[:, :, 1]
               
                # Ensure SHAP values and X_test_renamed have the same number of features
                assert shap_values.shape[1] == X_test.shape[1], "Mismatch between SHAP values and features."

                # Limit to top 10 features
                shap_values = shap.Explanation(values=shap_values, base_values=explainer.expected_value[1], data=X_test)
                shap_values_list.append(shap_values)

                # Plot the SHAP summary plot with top 10 features and reversed color map
                shap.summary_plot(shap_values, X_test, show=False, max_display=10, cmap=cmap, color_bar=False)

                #insert title 
                plt.title(f'{clean_model_name}')
                plt.savefig(f'images/shap/summaryplot_{save_name}.png', dpi=300, bbox_inches='tight')
                plt.clf()  # Clear the current figure
                
                # Plot bar
                shap.summary_plot(shap_values, X_test, plot_type="bar", show=False, max_display=10)

                #replace legend with "Feature importance"
                plt.legend(["mean(|SHAP value|)", "Feature importance"])
 
                #inset title
                plt.title(f'{clean_model_name}')
                plt.savefig(f'images/shap/barplot_{save_name}.png', dpi=300, bbox_inches='tight')
                plt.clf()  # Clear the current figure

                # Correct indexing for the force plot
                # shap_values_to_plot = shap_values[0, :].reshape(1, -1)

                # Plot force
                # shap.initjs()
                # force_plot = shap.force_plot(explainer.expected_value[1], shap_values_to_plot, X_test_renamed.iloc[0, :], show=False)
                # shap.save_html(f'images/shap/forceplot_{save_name}.html', force_plot)
                # plt.clf()  # Clear the current figure

                # Plot decision
                # shap.decision_plot(explainer.expected_value[1], shap_values, X_test_renamed, show=False)
                # plt.title(f'{clean_model_name}')
                # plt.savefig(f'images/shap/decisionplot_{save_name}.png', dpi=300, bbox_inches='tight')
                # plt.clf()  # Clear the current figure

                # Plot waterfall
                #shap.waterfall_plot(shap_exp[0])
                #plt.savefig(f'images/shap/waterfallplot_{save_name}.png', dpi=300, bbox_inches='tight')

            else:
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X_test)
                shap_values_list.append(shap_values)
                shap_exp = shap.Explanation(values=shap_values, base_values=explainer.expected_value, data=X_test)
                
                # Change feature names
                shap.summary_plot(shap_values, X_test, show=False, max_display=10, cmap=cmap, color_bar=False)
                plt.title(f'{clean_model_name}')
                plt.savefig(f'images/shap/summaryplot_{save_name}.png', dpi=300, bbox_inches='tight')
                plt.clf()  # Clear the current figure
                
                # Plot bar
                shap.summary_plot(shap_values, X_test, plot_type="bar", show=False, max_display=10)
                plt.title(f'{clean_model_name}')
                #replace legend with "Feature importance"
                plt.legend(["mean(|SHAP value|)", "Feature importance"])
 
                plt.savefig(f'images/shap/barplot_{save_name}.png', dpi=300, bbox_inches='tight')
                plt.clf()  # Clear the current figure
                
                # Plot force
                # shap.initjs()
                # force_plot = shap.force_plot(explainer.expected_value, shap_values[0,:], X_test_renamed.iloc[0,:], show=False)
                # shap.save_html(f'images/shap/forceplot_{save_name}.html', force_plot)
                # plt.clf()  # Clear the current figure
                
                # Plot decision
                # shap.decision_plot(explainer.expected_value, shap_values, X_test_renamed, show=False)
                # plt.title(f'{clean_model_name}')
                # plt.savefig(f'images/shap/decisionplot_{save_name}.png', dpi=300, bbox_inches='tight')
                # plt.clf()  # Clear the current figure
                
        # fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        # model_names = ['Random Forest', 'Gradient Boost', 'XGBoost']
        # for i in range(len(shap_values_list)):
        #     plt.sca(axs[i])
        #     shap.summary_plot(shap_values_list[i], X_test_renamed, show=False, plot_size=None)
        #     axs[i].set_title(model_names[i])
        
        # plt.tight_layout()
        # plt.savefig(f'images/shap/combined_summary_plot.png', dpi=300, bbox_inches='tight')
        # plt.clf()  # Clear the current figure