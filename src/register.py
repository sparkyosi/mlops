import json
from mlflow.tracking import MlflowClient
import mlflow

import dagshub

# Initialize DagsHub and MLflow integration
dagshub.init(repo_owner='asmaa.hnaien', repo_name='mlops', mlflow=True)

mlflow.set_experiment("MLOps")

# Set the tracking URI for MLflow to log the experiment in DagsHub
mlflow.set_tracking_uri("https://dagshub.com/asmaa.hnaien/mlops.mlflow") 

# Load the run ID and model name from the saved JSON file
reports_path = "./models/run_info.json"
with open(reports_path, 'r') as file:
    run_info = json.load(file)

run_id = run_info['run_id'] # Fetch run id from the JSON file
model_name = run_info['model_name']  # Fetch model name from the JSON file

# Create an MLflow client
client = MlflowClient()

# Create the model URI
model_uri = f"runs:/{run_id}/artifacts/{model_name}"

# Register the model
reg = mlflow.register_model(model_uri, model_name)

# Get the model version
model_version = reg.version  # Get the registered model version

# Transition the model version to Staging
new_stage = "Staging"

client.transition_model_version_stage(
    name=model_name,
    version=model_version,
    stage=new_stage,
    archive_existing_versions=True
)

print(f"Model {model_name} version {model_version} transitioned to {new_stage} stage.")