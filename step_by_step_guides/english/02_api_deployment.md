## 02 API Deployment

#### Objective

This document explains the most important aspects of 02_api_deployment.py.

#### Instructions for Code Execution

Open 02_api_deployment.py in your CML Session. Familiarize yourself with the code.

Open mlops.py. Familiarize yourself with the code.

Next, reutrn to 02_api_deployment.py and press the play button in order to run the whole script. You will be able to observe code output on the right side of your screen.

#### Code Highlights

* Line 46: the "ModelDeployment" class is imported from the "mlops" util. This util has been placed in the "/home/cdsw" folder.  

* Line 49: the CML API client is instantiated. The API provides you with over 100 Python methods to execute actions such as creating projects, launching jobs, and a lot more. In this example, the API is used to "list_projects()".

* Line 62: the API Client is passed as an argument to the ModelDeployment instance. The mlops.py util includes a few methods that extend and override API methods. Typically, CML Machine Learning Engineers create Python Interfaces to build custom MLOps pipelines as required by their use case.

* Line 68: the latest MLFlow Experiment Run is used to register the Model in the CML MLFlow Registry.

* Lines 74, 78, 81: the registered Model is used to create a new CML Model Deployment. The Model is first created, then built, and finally deployed.

#### References and Related Articles

* To learn more about MLFlow in CML:
* To learn more about CML Model Deployments:
* To learn more about the CML API:
