import mlflow
import json
import os


def model():
    mlflow.set_tracking_uri("https://dagshub.com/asmaa.hnaien/mlops.mlflow") 
     # Set DagsHub credentials as environment variables
    os.environ['MLFLOW_TRACKING_USERNAME'] = 'asmaa.hnaien'
    os.environ['MLFLOW_TRACKING_PASSWORD'] = '9070d7e0b57df1f9f23801b60c6a3e962c0a075b'

    reports_path = "models/run_info.json"
    with open(reports_path, 'r') as file:
        run_info = json.load(file)

    model_name = run_info['model_name']  


    try:
        # Create an MlflowClient to interact with the MLflow server
        client = mlflow.tracking.MlflowClient()

        # Get the latest version of the model in the Production stage
        versions = client.get_latest_versions(model_name, stages=["Production"])

        if versions:
            latest_version = versions[0].version
            run_id = versions[0].run_id  # Fetching the run ID from the latest version
            print(f"Latest version in Production: {latest_version}, Run ID: {run_id}")

            # Construct the logged_model string
            logged_model = f'runs:/{run_id}/{model_name}'
            print("Logged Model:", logged_model)

            # Load the model using the logged_model variable
            loaded_model = mlflow.pyfunc.load_model(logged_model)
            print(f"Model loaded from {logged_model}")

            return loaded_model
        else:
            print("No model found in the 'Production' stage.")

    except Exception as e:
        print(f"Error fetching model: {e}")


    
model()