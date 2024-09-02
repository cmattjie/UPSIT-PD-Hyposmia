## This repository contains the necessary code to reproduce the results of the research paper 'Hyposmia in Parkinson’s disease; exploring selective odour loss'.

This project uses machine learning to try to identify Parkinson's disease (PD) in a Hyposmic cohort using the University of Pennsylvania Smell Identification (UPSIT) tests. For comparison, it also provides all the steps necessary to identify PD from UPSIT tests from individuals with normosmia or hyposmia. 

# Data

Data should be downloaded directly from the [PPMI website](https://www.ppmi-info.org/). You must request access to download the required data. This project used PPMI Clinical and PPMI Remote data.

The following csv files should be downloaded from PPMI Clinical and moved to the _data/PPMI_clinical_ folder:
- Age_at_visit
- Demographics
- Participant_Status
- Socio-Economics
- University_of_Pennsylvania_Smell_Identification_Test_UPSIT
- PD_Diagnosis_History

The following csv files should be downloaded from PPMI Remote and moved to the _data/PPMI_remote/_ folder:
- Screening_High_Interest
- Screening_Participant_Progress
- Screening_Screener
- Screening_Smell_Test_Direct_Screener
- Screening_UPSIT_Screening
- University_of_Pennsylvania_Smell_Identification_Test

The following file is unavailable on the platform and was requested directly to PPMI: _Remote_Screening_BirthYear.csv_. For complete reproducibility, the birth year of the PPMI remote should be requested. 

# Installation

Download this repository and create an environment using Python 3.9. Then, install all necessary libraries using 'pip install -r requirements.txt'.

# Breakdown of Code

The order to properly preprocess data, prepare the training and test sets and run the experiments is ordered on the file names:

1. 1_merge_organize_data.ipynb
2. 2_match_&_stratify.ipynb
3. 3_script_train.sh
4. 4_script_test.sh

## 1_merge_organize_data.ipynb

This jupyter notebook organizes the files downloaded from PPMI and generates 3 CSV files used for training out models: _clinical_processed.csv_, _remote_processed.csv_, and _clinical_processed_imputed.csv_. After running this code, the files can be found at _data/processed/_.

## 2_match_&_stratify.ipynb

This jupyter notebook creates the training and test sets for the three main experiments: Hyposmoc PD x non-PD with imputed data; Hyposmic PD x non-PD without imputed data; Normosmic/Hyposmic PD x non-PD with imputed data. 

To create a balanced dataset (equal proportion of PD and non-PD), we find each individual from PPMI Remote who best matches age and sex from PPMI Clinical, as there is a lot more data available from non-PDs (PPMI Remote) than PDs (PPMI Clinical). Afterwards, it generates training and test sets balanced for age, sex, and presence of PD. 

## 3_script_train.sh

This script runs our trainer.py -- experimenting with 8 different ML models using leave-one-out cross-validation -- for each of the three datasets. Then, it repeats them, including age and sex variables as predictors in addition to UPSIT results. Each of the six experiments also runs twice: Once using only the correct features (if the smell was correctly identified) and once using all response features (which of the four smell alternatives was guessed).

## 4. 4_script_test.sh

This script runs our test.py for all experiments. Only models that reached the top performances on leave-one-out cross-validation should be considered. 

## shap_values.py 

This Python code provides SHAP summary plots in the _images/shap/_ folder and can be adapted for different experiments.

# Contact

If you need any support with the code, you can contact me here or through email: christian.mattjie@gmail.com