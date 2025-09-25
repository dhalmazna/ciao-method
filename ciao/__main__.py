"""
Main entry point for CIAO explanations.
"""

from datetime import datetime
from pathlib import Path
from random import randint

import hydra
import mlflow
from lightning import seed_everything
from omegaconf import DictConfig, OmegaConf

from ciao.components.explainer import CIAOExplainer
from ciao.components.factory import make_classifier
from ciao.data import CIAODataModule

OmegaConf.register_new_resolver(
    "random_seed", lambda: randint(0, 2**31), use_cache=True
)


@hydra.main(config_path="../configs", config_name="default", version_base=None)
def main(config: DictConfig) -> None:
    """Main CIAO explanation generation function."""
    seed_everything(config.seed, workers=True)

    # Setup data module
    data: CIAODataModule = hydra.utils.instantiate(
        config.data,
        _recursive_=False,
        _target_=CIAODataModule,
    )
    data.setup("explain")

    # Create explainer components
    classifier = make_classifier(config.variant)
    segmenter = hydra.utils.instantiate(config.ciao.segmentation)
    obfuscator = hydra.utils.instantiate(config.ciao.obfuscation)

    # Create CIAO explainer
    explainer = CIAOExplainer(
        classifier=classifier,
        segmenter=segmenter,
        obfuscator=obfuscator,
        **config.ciao.get("explainer_params", {}),
    )

    # Get dataloader
    dataloader = data.explain_dataloader()

    # Setup output directory
    output_dir = Path(config.explanation.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate timestamp for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_output_dir = output_dir / f"ciao_explanations_{timestamp}"
    run_output_dir.mkdir(parents=True, exist_ok=True)

    print("Starting CIAO explanation generation...")
    print(f"Variant: {config.variant}")
    print(f"Segmentation: {config.ciao.segmentation._target_}")
    print(f"Obfuscation: {config.ciao.obfuscation._target_}")
    print(f"Output directory: {run_output_dir}")

    # Process images
    explanations = []
    for i, (images, labels, metadata) in enumerate(dataloader):
        if i >= config.explanation.num_images:
            break

        # Get single image (batch size = 1)
        image = images[0]
        label = labels[0] if labels is not None else None

        print(f"Processing image {i + 1}/{config.explanation.num_images}...")

        # Generate explanation
        output_path = (
            run_output_dir / f"explanation_{i:03d}.png"
            if config.explanation.save_visualizations
            else None
        )

        explanation = explainer.explain(
            image=image,
            target_class=label.item() if label is not None else None,
            save_visualization=config.explanation.save_visualizations,
            output_path=output_path,
        )

        explanations.append(explanation)

        # Print results
        print(f"  Predicted class: {explanation['predicted_class']}")
        print(f"  Confidence: {explanation['confidence']:.3f}")
        print(f"  Target class: {explanation['target_class']}")
        print(f"  Found {explanation['n_feature_groups']} feature groups")
        print(f"  Used η = {explanation['eta_used']:.4f}")
        print()

    print(f"✅ Generated {len(explanations)} CIAO explanations")
    print(f"📁 Results saved to: {run_output_dir}")

    # Save summary
    summary = {
        "config": OmegaConf.to_yaml(config),
        "num_explanations": len(explanations),
        "timestamp": timestamp,
        "variant": config.variant,
        "segmentation_method": config.ciao.segmentation._target_,
        "obfuscation_method": config.ciao.obfuscation._target_,
    }

    summary_path = run_output_dir / "summary.yaml"
    with open(summary_path, "w") as f:
        OmegaConf.save(summary, f)

    print(f"📋 Summary saved to: {summary_path}")

    # Log artifacts to MLflow
    try:
        print("🔄 Logging artifacts to MLflow...")
        
        # Set experiment name from config
        experiment_name = config.metadata.get(
            "experiment_name", "Explainability"
        )
        mlflow.set_experiment(experiment_name)
        
        # Start MLflow run
        run_name = config.metadata.get(
            "run_name", f"CIAO_{config.variant}_{timestamp}"
        )
        with mlflow.start_run(run_name=run_name):
            # Log hyperparameters
            mlflow.log_param("variant", config.variant)
            mlflow.log_param(
                "segmentation_method", config.ciao.segmentation._target_
            )
            mlflow.log_param(
                "obfuscation_method", config.ciao.obfuscation._target_
            )
            mlflow.log_param("eta", config.ciao.explainer_params.eta)
            mlflow.log_param("num_images", config.explanation.num_images)
            mlflow.log_param("seed", config.seed)
            
            # Log metrics
            n_explanations = len(explanations)
            avg_feature_groups = (
                sum(exp['n_feature_groups'] for exp in explanations)
                / n_explanations
            )
            avg_confidence = (
                sum(exp['confidence'] for exp in explanations)
                / n_explanations
            )
            mlflow.log_metric("avg_feature_groups", avg_feature_groups)
            mlflow.log_metric("avg_confidence", avg_confidence)
            mlflow.log_metric("total_explanations", n_explanations)
            
            # Log all output files as artifacts
            mlflow.log_artifacts(
                str(run_output_dir), artifact_path="explanations"
            )
            
        print(f"✅ Artifacts logged to MLflow experiment '{experiment_name}'")
        
    except Exception as e:
        print(f"⚠️  MLflow logging failed: {e}")
        print("Continuing without MLflow logging...")


if __name__ == "__main__":
    main()
