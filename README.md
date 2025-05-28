# Predicting Drug Sensitivity using Basal Gene Expression in Cancer Cell Lines
This project aims to develop machine learning models to predict the IC50 values of drug-treated cancer cell lines using basal gene expression levels.

## Sources
Data was retrieved from [Cell Model Passports](https://cellmodelpassports.sanger.ac.uk/), [Genomics for Drug Sensitivity in Cancer](https://www.cancerrxgene.org/), and the [Drug-Gene Interaction database](https://dgidb.org/about/overview/introduction).

## Model Generation
Binary and regression models can be created through different parameters. Binary models report performance, and regression models report performance and save the produced elastic net model.
### Creating One Model
#### Parameters
`-dOI --drugOfInterest`: name of drug to produce models for
`-dID --drugID`: ID of the drug to produce models for
`-oP --outputPath`: directory to store any outputs to
`-gV --gdscVer`: version of the GDSC dataset to use. 1 or 2
`-b --binary`: 0/1 value indicating whether to make binary models. Set to 0 (regression models) by default.
`-v --visuals`: 0/1 value indicating whether to produce visuals. Set to 0 (no visuals) by default.
`-m --metadata`: 0/1 value indicating whether to write properties of the data to a file. Set to 0 by default.
`-dM --developerMode`: 0/1 value indicating whether to produce more in-depth, time-consuming visuals. Set to 0 by default.
#### Output
### Create Models for all Drugs
