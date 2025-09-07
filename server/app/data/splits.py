from typing import List, Dict, Set

def check_patient_leak(train_pids: List[str], val_pids: List[str], test_pids: List[str]) -> Dict[str, int]:
    """Check for patient overlap across train, validation, and test partitions."""
    s_train: Set[str] = set(train_pids)
    s_val: Set[str] = set(val_pids)
    s_test: Set[str] = set(test_pids)

    return {
        "train_val_overlap": len(s_train & s_val),
        "train_test_overlap": len(s_train & s_test),
        "val_test_overlap": len(s_val & s_test),
    }
