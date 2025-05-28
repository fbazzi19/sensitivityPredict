# Predicting Drug Sensitivity using Basal Gene Expression in Cancer Cell Lines
This project aims to develop machine learning models to predict the IC50 values of drug-treated cancer cell lines using basal gene expression levels.

## Sources
Data was retrieved from [Cell Model Passports](https://cellmodelpassports.sanger.ac.uk/), [Genomics for Drug Sensitivity in Cancer](https://www.cancerrxgene.org/), and the [Drug-Gene Interaction database](https://dgidb.org/about/overview/introduction).

## Model Generation
Binary and regression models can be created through different parameters. Binary models report performance, and regression models report performance and save the produced elastic net model.
### Creating One Model
#### Parameters
`-dOI --drugOfInterest`: name of drug to produce models for

### Create Models for all Drugs
