import torch
from util.yaml_check import yaml_add_or_update
import mlflow
from mlflow.models import infer_signature
import mlflow.onnx
import onnx

def dynamic_model_export_to_onnx(model, model_directory, registered_model_name: str):
    model_cpu = model.to("cpu")
    model.eval()
    dummy_input = torch.randn(1, 1, model.features)
    model_directory.mkdir(parents=True, exist_ok=True)
    model_directory.as_posix()
    onnx_dynamic_model_path = f"{model_directory}/VAE_industrial"
    torch.onnx.export(
        model_cpu,
        dummy_input,
        onnx_dynamic_model_path,
        export_params=True,
        input_names=[f"input_name"],
        output_names=["output_name", "output_mu", "output_logvar"],
        dynamic_axes={
            "input_name": {0: "batch_size", 1: "sequence_length"},
            "output_name": {0: "batch_size", 1: "sequence_length"},
            "output_mu": {0: "batch_size"},
            "output_logvar": {0: "batch_size"},
        },
    )

    yaml_add_or_update(key="onnx_dynamic_model_path", value=onnx_dynamic_model_path)

    # --- model size, as a param so it's comparable across architectures ---
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    mlflow.log_params(
        {"total_params": total_params, "trainable_params": trainable_params}
    )

    # --- log + register the model in the MLflow Model Registry ---
    # Unity Catalog requires a signature (input/output schema) on every
    # registered model version, so build one from the same dummy input used
    # for ONNX export, paired with a real forward pass for the outputs.
    with torch.no_grad():
        example_out, example_mu, example_logvar = model_cpu(dummy_input)

    input_example = dummy_input.detach().cpu().numpy()
    output_example = {
        "output_name": example_out.detach().cpu().numpy(),
        "output_mu": example_mu.detach().cpu().numpy(),
        "output_logvar": example_logvar.detach().cpu().numpy(),
    }
    signature = infer_signature(input_example, output_example)

    onnx_model = onnx.load(onnx_dynamic_model_path)
    mlflow.onnx.log_model(
        onnx_model,
        artifact_path="model",
        registered_model_name=registered_model_name,
        signature=signature,
        input_example=input_example,
    )

    # Write this run's id into config.yaml so any downstream consumer of the
    # ONNX dynamics model (e.g. the RL training script) can trace exactly
    # which VAE run -- hyperparameters, loss curves, registered version --
    # produced the simulator it trained against.
    active_run = mlflow.active_run()
    if active_run is not None:
        yaml_add_or_update(key="dynamics_model_run_id", value=active_run.info.run_id)

    print(f"Model Saved and registered as '{registered_model_name}'")