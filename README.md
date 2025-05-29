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
