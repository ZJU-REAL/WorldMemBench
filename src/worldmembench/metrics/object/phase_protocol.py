"""Object uses the shared, reference-preserving revisit protocol."""

from worldmembench.metrics.revisit_protocol import (
    TargetPhase, MAX_SHORT_MISSED_FRAMES, MIN_REAPPEAR_MATCH_RUN,
    MAX_EXPECTED_REVISITS, REVISIT_BOUNDARY_BUFFER_FRAMES,
    validate_expected_revisits, infer_target_phase, phase_from_revisit_runs,
)
