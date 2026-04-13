# Phase 5: Federated Learning Module
from .fl_client import FLClient, FLClientConfig
from .fl_server import FLServer, FLServerConfig, FLSimulator, FederatedIntegratedSimulator
from .aggregator import FedAvgAggregator, AggregationConfig, QuantizationHelper
from .drift_aware_fl import (
    DriftAwareFlSimulation,
    DynamicDriftConfig,
    DriftEvent,
    run_dynamic_drift_fl
)

__all__ = [
    'FLClient',
    'FLClientConfig',
    'FLServer',
    'FLServerConfig',
    'FLSimulator',
    'FederatedIntegratedSimulator',
    'FedAvgAggregator',
    'AggregationConfig',
    'QuantizationHelper',
    # Dynamic drift-aware FL
    'DriftAwareFlSimulation',
    'DynamicDriftConfig',
    'DriftEvent',
    'run_dynamic_drift_fl'
]
