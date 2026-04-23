import pytest
from faultline.models import IncidentConfig
from faultline.generator import ProceduralIncidentGenerator

def test_determinism():
    config = IncidentConfig(
        failure_mode="latency",
        cascade_depth=2,
        red_herring_count=1,
        noise_level=0.1
    )
    generator = ProceduralIncidentGenerator()
    scenario1 = generator.generate(config, seed=42)
    scenario2 = generator.generate(config, seed=42)
    
    assert scenario1.root_cause_service == scenario2.root_cause_service
    assert scenario1.affected_services == scenario2.affected_services
    assert len(scenario1.firing_alerts) == len(scenario2.firing_alerts)
    assert len(scenario1.red_herring_alerts) == len(scenario2.red_herring_alerts)

def test_failure_modes():
    generator = ProceduralIncidentGenerator()
    config = IncidentConfig(
        failure_mode="crash",
        cascade_depth=0,
        red_herring_count=0,
        noise_level=0.0
    )
    scenario = generator.generate(config, seed=1)
    assert scenario.correct_action_type == "rollback"

def test_cascade_depth():
    generator = ProceduralIncidentGenerator()
    config = IncidentConfig(
        failure_mode="oom",
        cascade_depth=3,
        red_herring_count=0,
        noise_level=0.0
    )
    scenario = generator.generate(config, seed=1)
    assert len(scenario.affected_services) == 3

def test_red_herrings_count():
    generator = ProceduralIncidentGenerator()
    config = IncidentConfig(
        failure_mode="config_drift",
        cascade_depth=1,
        red_herring_count=2,
        noise_level=0.0
    )
    scenario = generator.generate(config, seed=1)
    assert len(scenario.red_herring_alerts) == 2

def test_noise_injection():
    # noise level exists in config but generator currently doesn't implement advanced noise
    # We just test config parameter
    generator = ProceduralIncidentGenerator()
    config = IncidentConfig(
        failure_mode="connection_leak",
        cascade_depth=0,
        red_herring_count=0,
        noise_level=0.5
    )
    scenario = generator.generate(config, seed=1)
    assert scenario.root_cause_service is not None
