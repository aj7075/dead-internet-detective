from __future__ import annotations

from dead_internet_detective.models import State


def synthetic_detection_rate(state: State, filed_report: dict) -> float:
    flagged = set(state.case_file.synthetic_sources)
    total_synthetic = [
        url for url, page in state.synthetic_internet.items()
        if page.ground_truth_label != "credible"
    ]
    if not total_synthetic:
        return 1.0
    correctly_flagged = sum(
        1 for url in total_synthetic if url in flagged
    )
    return correctly_flagged / len(total_synthetic)
