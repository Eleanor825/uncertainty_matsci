"""Fixed MADE executor declarations, separate from scientific model settings."""
from collections.abc import Mapping


def declared_mace_workers(protocol: Mapping) -> int:
    """Legacy protocols used the single official calculator; new runs declare it."""
    execution = protocol.get("made_execution", {})
    if not isinstance(execution, Mapping):
        raise ValueError("MADE execution configuration must be a mapping")
    workers = execution.get("mace_num_workers", 1)
    if type(workers) is not int or workers not in (1, 4):
        raise ValueError("Only validated MACE executor sizes 1 and 4 are supported")
    if type(execution.get("orb_num_workers", 1)) is not int or execution.get("orb_num_workers", 1) != 1:
        raise ValueError("The final ORB oracle must retain one worker")
    return workers


def verify_mace_workers(arguments: Mapping, metadata: Mapping, expected: int) -> None:
    """Reject mixed executor conditions; only legacy serial may omit the fields."""
    if type(expected) is not int or expected not in (1, 4):
        raise ValueError("Expected MACE workers must be 1 or 4")
    requested = arguments.get("mace_num_workers", 1)
    recorded = metadata.get("mace_num_workers", 1)
    orb = metadata.get("orb_num_workers", 1)
    if (type(requested) is not int or type(recorded) is not int or type(orb) is not int
            or requested != expected or recorded != expected or orb != 1):
        raise ValueError("MADE init/metadata executor differs from the fixed paired condition")
    if expected == 4:
        from .mace_parallel import mace_execution_metadata
        if metadata.get("orb_num_workers") != 1 or metadata.get("mace_execution") != mace_execution_metadata(4):
            raise ValueError("Parallel MADE executor lacks its complete lifecycle and ORB declaration")
