# Model Process Control Simulator

A chemical process simulator using the [Chemical Process Monitoring Time-Series Dataset](https://www.kaggle.com/datasets/rohit8527kmr7518/chemical-process-monitoring-time-series-dataset) from Kaggle. The modular workflow goes through cleaning the data and defining variables, creating the ML and RL models, and deploying them in a simulator environment, all tracked and logged 

The project has the following components:

- Machine Learning in Pytorch and Reinforcement Learning using Gym to create predictive and responsive models
- Process Dynamic and Control loop using trained models to simulate a live process
- Model saving and atomic model switching using Databricks
- Redundancy using local model saves and local databases prior to cloud uploads  
- FastAPI frontend to track parameters over time

---

## Background

With automation becoming more prevalent in not only traditional industrial applications but also newer novel industries including Biotech, plants are containing more constituent parts that make up an entire process. While individual PI controls are generally enough for these parts, the whole process may rely on parts that are not talking to each other. Interacting variables and dead-time can cause sub-optimal solutions to be taken by downstream controllers. The solution to this is to create a model that can evaluate the entire plant with regard to all other smaller parts. Outside of the traditional method of creating mathematical models, with enough data, machine learning models can be trained to model the plant and all its end to end parts. This project shows a small demonstration of what this would look like using state, target, and action variables to train to RL model to control the outputs of the dynamic model and maximize quality.  

---

## Architecture

```text
Module 0 (Data Cleaning and Variable Definition) 
    ↓
Module 1 (Dynamic Model Training)
    ↓
Module 2 (RL Model Training)
    ↓
FastAPI
    ↓
MLflow and Databricks
    ↓
Process Dynamic and Process Control Loop 

```

The FastAPI app is deployed on AWS EC2, along with the cloud TimeScaleDB. A local Redis server coordinates inference between models and a SQLite database stores process data locally and buffers it until it is transferred to the cloud.

---

## Tech stack

| Layer | Tooling |
|---|---|
| API | FastAPI, Pydantic, Uvicorn |
| Frontend | FastAPI | Jinja | 
| Model Training | PyTorch | Gym |
| Model Lifecycle | MLFlow | Databricks |
| Process Loop | Redis | SQLite | TimeScale DB |
| Deployment | Docker Compose, AWS EC2 |

---

## Key Features

### Variable Defining in Data Cleaning

- Guided process for defining state, target, and action variables while performing data cleaning

### Model training Modules for PyTorch Dynamic and Gym RL Models

### Redundent Data and Inference Pathways

- Inference data is stored in local SQLite database
- Model configs pulled from Databricks, not loaded until local model is saved for deployment
- Model paths and champion model stored in local config file as well as Databricks

### Deployment
The project includes a compose file for a local on-premise server for the inference loop, a compose file for the cloud server and FastAPI app, and a compose file for hosting it all locally on one machine for testing.

## Development Requirements

### Prerequisites

- Python 3.13+
- Docker
  
### Install dependencies

```bash
uv sync 
```

### Run the inference loop

```bash
bash server_run.sh
```

### Run the EC2 App

```bash
bash ec2_run.sh
```

### Run everything locally

```bash
bash local_test.sh
```


