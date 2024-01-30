# SemEval 2024 - Track A

This track consists of the supervised part of the problem statement which involves calculation of the semantic textual relatedness score for 9 different languages. This approach uses the `Constrastive learning` approach using `distillroberta` model. The script `simcse_trial.py` has the entire pipeline developed for training on the *train* and the *dev_with_labels* data and inferencing on the provided *test* data for each of the languages.

The model is trained for 15 epochs using the contrastive loss function. 

Steps to run:

1. Set the language for which you want to run the script by changing the variable name `lang` to the desired language.

2. Run the script

```py
python3 simcse_trial.py
```

The script automatically produces a folder `preds` which contains the *IDs* and their *semantic relatedness scores* for each sentence pair in each of the languages. The folder structure:

```
├── pred_amh_a.csv
├── pred_arq_a.csv
├── pred_ary_a.csv
├── pred_eng_a.csv
├── pred_esp_a.csv
├── pred_hau_a.csv
├── pred_kin_a.csv
├── pred_mar_a.csv
└── pred_tel_a.csv
```