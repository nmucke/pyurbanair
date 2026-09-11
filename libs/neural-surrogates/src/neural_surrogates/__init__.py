from neural_surrogates import architectures
from neural_surrogates.architectures import (
    P3D,
    UPT,
    DomainDecomposed,
    GeometryBranch,
    LatentEncoding,
    ParamConditionedSubnetwork,
    SimpleConv,
    TadpoleAE,
    TadpoleDiscriminator,
    TadpoleLatentGenerator,
    TadpoleTimeStepper,
    UNetConvNeXt,
)
from neural_surrogates.datasets import (
    PatchTransitionDataset,
    SnapshotDataset,
    SnapshotHistoryDataset,
    TrajectoryBatchSampler,
    TransitionDataset,
    snapshot_collate,
    snapshot_history_collate,
)
from neural_surrogates.dd_loss import DomainDecompositionLoss
from neural_surrogates.decomposition import DomainDecomposition
from neural_surrogates.ensemble_forward_model import NeuralSurrogateEnsembleForwardModel
from neural_surrogates.forward_model import NeuralSurrogateForwardModel
from neural_surrogates.generative_spinup import GenerativeSpinup
from neural_surrogates.sdf import (
    n_sdf_feature_channels,
    normalize_sdf_mode,
    sdf_features,
)
from neural_surrogates.training import (
    AutoencoderTrainer,
    BaseTraining,
    LatentFlowMatchingTrainer,
    PatchTrainer,
    Trainer,
)

__all__ = [
    "TransitionDataset",
    "PatchTransitionDataset",
    "TrajectoryBatchSampler",
    "SnapshotDataset",
    "snapshot_collate",
    "SnapshotHistoryDataset",
    "snapshot_history_collate",
    "DomainDecompositionLoss",
    "BaseTraining",
    "PatchTrainer",
    "Trainer",
    "AutoencoderTrainer",
    "LatentFlowMatchingTrainer",
    "architectures",
    "SimpleConv",
    "UNetConvNeXt",
    "UPT",
    "P3D",
    "DomainDecomposed",
    "TadpoleAE",
    "TadpoleDiscriminator",
    "GeometryBranch",
    "TadpoleTimeStepper",
    "ParamConditionedSubnetwork",
    "TadpoleLatentGenerator",
    "LatentEncoding",
    "DomainDecomposition",
    "NeuralSurrogateForwardModel",
    "NeuralSurrogateEnsembleForwardModel",
    "GenerativeSpinup",
    "sdf_features",
    "n_sdf_feature_channels",
    "normalize_sdf_mode",
]
