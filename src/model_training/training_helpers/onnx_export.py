import torch
from util.yaml_check import yaml_add_or_update
import mlflow
from mlflow.models import infer_signature
import mlflow.onnx
import onnx
from pathlib import Path

def dynamic_model_export_to_onnx(model, model_directory, registered_model_name: str):
    model_cpu = model.to("cpu")
    model.eval()
    dummy_input = torch.randn(1, 1, model.features)
    model_directory.mkdir(parents=True, exist_ok=True)
    model_directory.as_posix()
    onnx_dynamic_model_path = f"{model_directory}/{registered_model_name}"
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
    # if active_run is not None:
    #     yaml_add_or_update(key="dynamics_model_run_id", value=active_run.info.run_id)

    print(f"Model Saved and registered as '{registered_model_name}'")

    return active_run

def rl_model_export_to_onnx(model, model_directory, registered_model_name: str, model_name, onnxable_policy, env):


    onnxable_model = onnxable_policy(
        model.policy.actor,
        action_low=env.action_space.low,
        action_high=env.action_space.high,
    ).to(device="cpu")
    onnxable_model.actor.set_training_mode(False)
    agent_shape = onnxable_model.actor.observation_space["agent"].shape[0]
    quality_shape = onnxable_model.actor.observation_space["target"].shape[0]
    agent_input = torch.randn(1, agent_shape)
    quality_input = torch.randn(1, quality_shape)
    save_directory = Path(f"{model_directory}")
    save_directory.mkdir(parents=True, exist_ok=True)
    onnx_RL_model_path = f"{save_directory.as_posix()}/{model_name}.onnx"
    print("exporting model to ONNX")
    torch.onnx.export(
        onnxable_model,
        (agent_input, quality_input),
        onnx_RL_model_path,
        opset_version=18,
        input_names=["agent", "target"],
        output_names=["action"],
        dynamic_axes={
            "agent": {0: "batch_size"},
            "target": {0: "batch_size"},
            "action": {0: "batch_size"},
        },
    )
    # --- model size ---
    total_params = sum(p.numel() for p in onnxable_model.parameters())
    trainable_params = sum(
        p.numel() for p in onnxable_model.parameters() if p.requires_grad
    )
    mlflow.log_params(
        {"total_params": total_params, "trainable_params": trainable_params}
    )

    with torch.no_grad():
        example_action = onnxable_model(agent_input, quality_input)

    input_example = {
        "agent": agent_input.detach().cpu().numpy(),
        "target": quality_input.detach().cpu().numpy(),
    }
    output_example = example_action.detach().cpu().numpy()
    signature = infer_signature(input_example, output_example)

    onnx_model = onnx.load(onnx_RL_model_path)
    mlflow.onnx.log_model(
        onnx_model,
        artifact_path="model",
        registered_model_name=registered_model_name,
        signature=signature,
        input_example=input_example,
    )

    active_run = mlflow.active_run()
    # if active_run is not None:
    #     yaml_add_or_update(key="rl_model_run_id", value=active_run.info.run_id)

    print(f"model saved and registered as '{registered_model_name}'")

    return active_run