## This repository contains the necessary code to reproduce the results of the research paper 'Hyposmia in Parkinson’s disease; exploring selective odour loss'.

# Data

Data should be downloaded directly from the PPMI website (https://www.ppmi-info.org/). You must request access to download the required data. This project used PPMI Clinical and PPMI Remote data.

The following csv files should be downloaded from PPMI Clinical and moved to the 'data/PPMI_clinical' folder:
- Age_at_visit
- Demographics
- Participant_Status
- Socio-Economics
- University_of_Pennsylvania_Smell_Identification_Test_UPSIT
- PD_Diagnosis_History

The following csv files should be downloaded from PPMI Remote and moved to the 'data/PPMI_remote' folder:
- Screening_High_Interest
- Screening_Participant_Progress
- Screening_Screener
- Screening_Smell_Test_Direct_Screener
- Screening_UPSIT_Screening
- University_of_Pennsylvania_Smell_Identification_Test

The following file is not available on the platform and was requested directly to PPMI: _Remote_Screening_BirthYear.csv_. For complete reproducibility, the birth year of the PPMI remote should be requested. 

# Installation

Download this repository and create an environment using Python 3.9. Then, install all necessary libraries using 'pip install -r requirements.txt'.

# Breakdown of Code

The order to properly preprocess data, prepare the training and test sets and run the experiments is ordered on the file names:

1. 1_merge_organize_data.ipynb
2. 2_match_&_stratify.ipynb
3. 3_script_train.sh
4. 4_script_test.sh

## 1_merge_organize_data

This jupyter notebook organizes the files downloaded from PPMI and generates 3 csv files used for training out models: _clinical_processed.csv_, _remote_processed.csv_, and _clinical_processed_imputed.csv_. After running this code, the files can be found at _data/processed/_
