"""Model definitions and factory helpers for Lorenz dynamics learning."""

from src.models.mlp_predictor import MLPPredictor
from src.models.neural_ode import NeuralODEModel
from src.models.resnet_predictor import ResNetPredictor
from src.models.vector_field_net import VectorFieldNet


MODEL_REGISTRY = {
    "mlp": MLPPredictor,
    "resnet": ResNetPredictor,
    "neural_ode": NeuralODEModel,
}


def build_model(model_name: str, model_kwargs: dict):
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{model_name}'.")
    return MODEL_REGISTRY[model_name](**model_kwargs)


__all__ = [
    "MLPPredictor",
    "ResNetPredictor",
    "VectorFieldNet",
    "NeuralODEModel",
    "MODEL_REGISTRY",
    "build_model",
]
