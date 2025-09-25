from kube_jobs import Storage, submit_job

submit_job(
    job_name="ciao-explanation-generation",
    username="dhalmazna",
    cpu=4,
    memory="10Gi",
    gpu="mig-1g.10gb",
    public=False,
    script=[
        "git clone https://github.com/dhalmazna/ciao-method.git",
        "cd ciao-method",
        "uv sync",  # Install dependencies using uv like counterfactuals
        "mkdir -p ./output ./explanations",  # Create output directories
        "uv run python -m ciao",  # Uses default prostate cancer config
    ],
    storage=Storage(mou=True),
)
