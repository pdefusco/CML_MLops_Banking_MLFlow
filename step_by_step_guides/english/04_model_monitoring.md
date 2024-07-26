## 04 Model Monitoring

#### Objective

This document explains the most important aspects of 06_model_simulation.py and 07_model_monitoring.py.

#### Instructions for Code Execution

Open 06_model_simulation.py and 07_model_monitoring.py in your CML Session. Familiarize yourself with the code and update the DBNAME, STORAGE, and CONNECTION_NAME variables as instructed by your HOL Lead.

Execute 06_model_simulation.py and immediately navigate out of the Session to the CML Model Deployment. Open the Monitoring tab and watch the monitoring dashboard get updated in real time as requests are received by the model endpoint.

Next, run 07_model_monitoring.py. Explore the model monitoring diagrams on the right side of the page. How is your model performing?

#### Code Highlights

06_model_simulation.py creates more synthetic data and leverages the CDSW SDK to interact with the deployed endpoints. The data is submitted to the model along with ground truth in order to simulate a wave of requests to the endpoint.

* Line 41: the cdsw sdk is imported. This does not need to be installed.
* Lines 91-129: a method to submit request batches to the specified Model Deployment endpoint.
* Lines 134-154: a method to submit ground truth to the specified Model Deployment endpoint.

07_model_monitoring.py shows you how you can monitor model performance programmatically. CML features a Postgres Database called "Model Metrics Store" that automatically stores metadata for each request by deployed model. In this script, the cdsw SDK is used again in order to read model metadata and access the Model Metrics Store.

* Lines 66-67: the ApiUtility is instantiated to obtain model metadata. The util's source code is located in the src folder. Similarly to the "mlops" util you leveraged in 02_api_deployment.py, this util allows you to build custom methods in order to obtain model metadata as needed by your use case.

* Line 76: cdsw.read_metrics() is used to read the tracked model requests from the Model Metrics Store.

* Seaborn and Pandas are used throughout the rest of the script in order to create the model performance plots.

#### References and Related Articles

* To learn more about MLFlow in CML:
* To learn more about CML Model Deployments:
* To learn more about the CML API:
