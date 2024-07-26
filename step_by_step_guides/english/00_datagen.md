## 00 Datagen

#### Objective

This document explains the most important aspects of 00_datagen.py.

#### Instructions for Code Execution

Open 00_datagen.py in your CML Session and update the DBNAME, STORAGE, and CONNECTION_NAME variables at lines 160-162 as instructed by your HOL Lead.

Next, press the play button in order to run the whole script. You will be able to observe code output on the right side of your screen.

#### Code Highlights

* Line 50: the cml.data_v1 library is imported. This library allows you to take advantage of CML Data Connections in order to launch a Spark Session and connect to the Data Lake. The DataConnection is used at lines 103 - 109 within the "createSparkConnection" module.

* Lines 64 - 95: the "dataGen" module is used to create synthetic data for the classification use case. Observe the data attributes that are being created, and their respective types and value ranges.

* Lines 141 and 143: the PySark API for Apache Iceberg is used to create or append data to an Iceberg table format table from a PySpark dataframe.

#### References and Related Articles

* To learn more about CML Data Connections:
* To learn more about Apache Iceberg:
* To learn more about the PySpark API for Iceberg:
