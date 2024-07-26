## 03 MLOps Pipeline

#### Objective

This document explains the most important aspects of 03_newbatch.py, 04_train_xgboost.py, and 05_api_redeployment.py.

#### Instructions for Code Execution

Open 03_newbatch.py, 04_train_xgboost.py and 05_api_redeployment.py in your CML Session. Familiarize yourself with the code and update the DBNAME, STORAGE, and CONNECTION_NAME variables as instructed by your HOL Lead.

Do not run the scripts individually. Create a CML Job for each instead. Do not run the jobs yet.

Create Job "New Batch" with the following configurations:

```
Name: New Batch Paul
Script: 03_newbatch.py
Editor: Workbench
Kernel: Python 3.9
Spark Add On: Spark 3.2 or 3.3
Edition: Standard
Version: 2024.02
Schedule: Manual
Resource Profile: 2 vCPU / 4 Gib / 0 GPU
```

Create Job "New Batch" with the following configurations:

```
Name: Retrain XGBoost Paul
Script: 04_train_xgboost.py
Editor: Workbench
Kernel: Python 3.9
Spark Add On: Spark 3.2 or 3.3
Edition: Standard
Version: 2024.02
Schedule: Dependent on New Batch Paul
Resource Profile: 2 vCPU / 4 Gib / 0 GPU
```

Create Job "New Batch" with the following configurations:

```
Name: API Redeployment Paul
Script: 05_api_redeployment.py
Editor: Workbench
Kernel: Python 3.9
Spark Add On: Spark 3.2 or 3.3
Edition: Standard
Version: 2024.02
Schedule: Dependent on Retrain XGBoost Paul
Resource Profile: 2 vCPU / 4 Gib / 0 GPU
```

Once you created all three jobs, manually trigger the New Batch job. Monitor execution in the Job History tab, and observe that once it is complete the next job in the MLOps pipeline, Retrain XGBoost, is triggered, and finally the last job, API Redeployment, is executed.

#### Code Highlights

* 03_newbatch.py is mostly identical to 00_datagen.py.

* 04_train_xgboost.py is nearly identical to "01_train_xgboost.py". However, at lines 67-69 Iceberg Snapshot metadata is stored as variables. This metadata is used at lines 71-75 in order to perform an Incremental Read i.e. only loading data from the Iceberg table within a start and end time boundary. The metadata is then saved as MLFlow Tags during Experiment Run execution.

* 05_api_redeployment.py includes both methods from the mlops util and code to execute the MLOps pipeline. This is also nearly identical to the code in "02_api_deployment.py".

#### References and Related Articles

* To learn more about MLFlow in CML:
* To learn more about CML Model Deployments:
* To learn more about the CML API:
