from kube_jobs import Storage, submit_job

submit_job(
    job_name="ciao-explanation-generation",
    username="dhalmazna",
    cpu=4,
    memory="10Gi",
    gpu="mig-1g.10gb",
    public=False,
    script=[
        "echo '=== Environment Information ==='",
        "echo 'MLFLOW_TRACKING_URI:' $MLFLOW_TRACKING_URI",
        "echo 'MLFLOW_EXPERIMENT_NAME:' $MLFLOW_EXPERIMENT_NAME",
        "echo 'MLFLOW_ARTIFACT_STORE:' $MLFLOW_ARTIFACT_STORE",
        "echo 'Working directory:' $(pwd)",
        "echo '================================'",
        "git clone https://github.com/dhalmazna/ciao-method.git",
        "cd ciao-method",
        "uv sync",  # Install dependencies using uv like counterfactuals
        "mkdir -p ./output ./explanations",  # Create output directories
        "uv run python -c 'import mlflow; print(mlflow.get_tracking_uri())'",
        "uv run python -m ciao",  # Uses default prostate cancer config
        "echo '=== Output Files ==='",
        "ls -la explanations/",
        "find explanations/ -name '*.png' -o -name '*.yaml' | head -10",
    ],
    storage=Storage(mou=True),
)
