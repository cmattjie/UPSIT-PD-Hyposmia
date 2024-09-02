#Main experiment only using patients with Hyposmia including imputed data
python test.py --config config_files/config.json --impute True
#Experiment only using patietns with Hyposmia without including imputed data
python test.py --config config_files/config.json
#Experiments using data from all patients regardless of their hyposmia status. Includes imputed data
python test.py --config config_files/config_alldata.json --impute True

#repeated experiments with sex and age variables

python test.py --config config_files/config.json --impute True --age_sex True
python test.py --config config_files/config.json --age_sex True
python test.py --config config_files/config_alldata.json --impute True --age_sex True