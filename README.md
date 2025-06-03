# Predicting Drug Sensitivity using Basal Gene Expression in Cancer Cell Lines
This project aims to develop machine learning models to predict the IC50 values of drug-treated cancer cell lines using basal gene expression levels.

# Sources
Data was retrieved from [Cell Model Passports](https://cellmodelpassports.sanger.ac.uk/), [Genomics for Drug Sensitivity in Cancer](https://www.cancerrxgene.org/), and the [Drug-Gene Interaction database](https://dgidb.org/about/overview/introduction).

# Model Generation
Binary and regression models can be created through different parameters. Binary models report performance, and regression models report performance and save the produced elastic net model.
### Creating One Model
`python3 workflow.py -dOI [dOI] -dID [dID] -oP [path/for/outputs/] -gV [1/2] -b [0/1] -v [0/1] -m [0/1] -dM [0/1]`
### Parameters
`-dOI --drugOfInterest`: name of drug to produce models for  
`-dID --drugID`: ID of the drug to produce models for  
`-oP --outputPath`: directory to store any outputs to  
`-gV --gdscVer`: version of the GDSC dataset to use. 1 or 2  
`-b --binary`: 0/1 value indicating whether to make binary models. Set to 0 (regression models) by default.  
`-v --visuals`: 0/1 value indicating whether to produce visuals. Set to 0 (no visuals) by default.  
`-m --metadata`: 0/1 value indicating whether to write properties of the data to a file. Set to 0 by default.  
`-dM --developerMode`: 0/1 value indicating whether to produce more in-depth, time-consuming visuals. Set to 0 by default.  
### Outputs
If creating binary models:  
`[outputPath]all_classification_metrics.csv`: Accuracy and AUC-PR of Logistic Regression, LDA, and KNN models when tested on the test set in the format  
> `Model, Drug, Accuracy, AUCPR, Random Accuracy, Random AUCPR`

> If run again for binary models and the same output path, the results will be appended to the same file.

If creating regression models:  
`[outputPath]all_regression_metrics.csv`: R2, MSE, RMSE, and Pearson Correlation Coefficients of Linear Regression and Elastic Network models when tested on the test set in the format
> `Model, Drug, R2, MSE, RMSE, Pearson Correlation`

> If run again for regression models and the same output path, the results will be appended to the same file.

`[outputPath]models/GDSC[gdscVer]_[drugOfInterest]_[drugID]_elastnet_model.pkl`: The elastic net model produced  
`[outputPath]model_genes/GDSC[gdscVer]_[drugOfInterest]_[drugID]_model_genes.csv`: The genes used as features for the models  
`[outputPath]model_coefs/GDSC[gdscVer]_[drugOfInterest]_[drugID]_top_coefs.csv`: All non-zero gene coefficients for the elastic net model  

If visuals are being produced:  
`[outputPath]GDSC[gdscVer]_[drugOfInterest]_[drugID].pdf`: visuals produced during data pre-processing  
`[outputPath]GDSC[gdscVer]_[drugOfInterest]_[drugID]_classification.pdf`: visuals produced during the production of binary models  
`[outputPath]GDSC[gdscVer]_[drugOfInterest]_[drugID]_regression.pdf`: visuals produced during the production of regression models  
`[outputPath]GDSC[gdscVer]_[drugOfInterest]_[drugID]__y_train_set.csv`: List of cell lines used in training the models  

If metadata is being written:  
`[outputPath]metadata.csv`: Properties of the data being used to develop the model in the format  
> `Drug, Total_cell_lines, Total_genes, Total_ic50_var, Train_cell_lines, Train_ic50_var, Test_cell_lines, Test_ic50_var`

> If run again with the same output path, the results will be appended to the same file.  

## Create Models for all Drugs
Binary or regression models are created for every drug in a specified GDSC version. Binary models report performance, and regression models report performance and save the produced elastic net model. This is done using a SLURM HPC.  
`./modelsbatch.sh [OUTPUT_PATH] [GDSC_VER] [METADATA] [BINARY] [EMAIL] [CONDA_PATH]`  

### Parameters  
`OUTPUT_PATH`: directory to store any outputs to  
`GDSC_VER`: version of the GDSC dataset to use. 1 or 2  
`METADATA`: 0/1 value indicating whether to write properties of the data to a file.  
`BINARY`: 0/1 value indicating whether to make binary models.  
`EMAIL`: email address to send updates regarding the run of the job to.  
`CONDA_PATH`: miniconda directory  

### Outputs
If creating binary models:  
`[outputPath]all_classification_metrics.csv`: Accuracy and AUC-PR of Logistic Regression, LDA, and KNN models when tested on the test set in the format  
> `Model, Drug, Accuracy, AUCPR, Random Accuracy, Random AUCPR`

> If run again for binary models and the same output path, the results will be appended to the same file.

If creating regression models:  
`[outputPath]all_regression_metrics.csv`: R2, MSE, RMSE, and Pearson Correlation Coefficients of Linear Regression and Elastic Network models when tested on the test set in the format
> `Model, Drug, R2, MSE, RMSE, Pearson Correlation`  

> If run again for regression models and the same output path, the results will be appended to the same file.  

`[outputPath]models/GDSC[gdscVer]_[drugOfInterest]_[drugID]_elastnet_model.pkl`: The elastic net models produced. One model per drug.  
`[outputPath]model_genes/GDSC[gdscVer]_[drugOfInterest]_[drugID]_model_genes.csv`: The genes used as features for the models. One file per drug model.  
`[outputPath]model_coefs/GDSC[gdscVer]_[drugOfInterest]_[drugID]_top_coefs.csv`: All non-zero gene coefficients for the elastic net model. One file per drug model.  

# One Model Against All Drugs
Takes one elastic net model and assesses its predictive performance for all drugs.

## For one Model
`python3 one_v_all.py -m [model] -g [genes] -oP [path/for/outputs/]`
### Parameters
`-m --model`: model to be assessed. Output from `workflow.py`  
`-g --genes`: the genes that were used as features for the model. Output from `workflow.py`  
`-op --outputPath`:  directory to store any outputs to  

### Outputs
`[outputPath]one_all_metrics/[drug_model_name]_oneall_metrics.csv`: R2, RMSE, and Pearson Correlation for the performance of the model on all the drugs. In the format  
> `Drug, R2, RMSE, Pearson Correlation`

## For all Models
Done using a SLURM HPC.  
`./oneallsbatch.sh [OUTPUT_PATH] [GDSC_VER] [EMAIL] [CONDA_PATH]`  
### Parameters
`OUTPUT_PATH`: directory to store any outputs to  
`GDSC_VER`: version of the GDSC dataset to use. 1 or 2
`EMAIL`: email address to send updates regarding the run of the job to.  
`CONDA_PATH`: miniconda directory  

### Outputs
`[outputPath]one_all_metrics/[drug_model_name]_oneall_metrics.csv`: R2, RMSE, and Pearson Correlation for the performance of the models on all the drugs. One file per drug model. In the format  
> `Drug, R2, RMSE, Pearson Correlation`  

# Workflow  
Example workflow, how I produced my models.

## Model Production  
Binary models for GDSC1:  
> `./modelsbatch.sh ./Outputs 1 0 1 [EMAIL] [CONDA_PATH]`  

Binary models for GDSC2:  
> `./modelsbatch.sh ./Outputs 2 0 1 [EMAIL] [CONDA_PATH]`  

Regression models for GDSC1:  
> `./modelsbatch.sh ./Outputs 1 1 0 [EMAIL] [CONDA_PATH]`

Regression models for GDSC2:
> `./modelsbatch.sh ./Outputs 2 1 0 [EMAIL] [CONDA_PATH]`  

## Testing Two Drug Models Against All Drugs  
GDSC2, Afatinib, 1032  
`python3 one_v_all.py -m ./Outputs/models/GDSC2_Afatinib_1032_elastnet_model.pkl -g ./Outputs/model_genes/GDSC2_Afatinib_1032_model_genes.csv -oP ./Outputs`  
GDSC2, Selumetinib, 1062  
`python3 one_v_all.py -m ./Outputs/models/GDSC2_Selumetinib_1062_elastnet_model.pkl -g ./Outputs/model_genes/GDSC2_Selumetinib_1062_model_genes.csv -oP ./Outputs`  

## Visual Production  
### Metadata  
Knit `r_visuals/metadata_vis.Rmd` with parameters. Set `dir` to the directory containing `metadata.csv`.  
### Classification Models  
Knit `r_visuals/all_class_metrics_vis.Rmd` with parameters. Set `metricsdir` to the directory containing `all_classification_metrics.csv`. Set `metadir` to the directory containing `metadata.csv`.  
### Regression Models  
Knit `r_visuals/all_metrics_vis.Rmd` with parameters. Set `metricsdir` to the directory containing `all_regression_metrics.csv`. Set `metadir` to the directory containing `metadata.csv`.  
### Model Coefficients  
Knit `r_visuals/top_coefs_vis.Rmd` with parameters. Set `coefsdir` to the directory containing the `model_coefs/GDSC[gdscVer]_[drugOfInterest]_[drugID]_top_coefs.csv` files for every drug.  
### One Model, All Drugs  
Knit `r_visuals/onall_visuals.Rmd` with parameters. Set `drug_dataset` to either `GDSC1` or `GDSC2`. Set `drug_name` to the name of the drug the model is for. Set `drug_id` to the ID of the drug the model is for. Set `metricsdir` to the directory containing `[drug_dataset]_[drug_name]_[drug_id]_oneall_metrics.csv`.

:shipit:
