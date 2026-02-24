"""
MIT-BIH beat annotation mapping focused on main arrhythmia groups.

Main (AAMI-style) beat groups:
- N: normal + bundle/escape variants
- S: supraventricular ectopic
- V: ventricular ectopic
- F: fusion beat
- Q: paced/unclassifiable

Rhythm annotations (e.g. "(AFIB"), noise/quality annotations ("qq", "U"), and
other non-beat markers are intentionally ignored.
"""

# Main arrhythmia class ids (optional multiclass use).
MAIN_CLASS_TO_ID = {
    "N": 0,
    "S": 1,
    "V": 2,
    "F": 3,
    "Q": 4,
}
ID_TO_MAIN_CLASS = {v: k for k, v in MAIN_CLASS_TO_ID.items()}

# MIT-BIH beat symbols grouped into main arrhythmia classes.
SYMBOL_TO_MAIN_CLASS = {
    # N class
    "·": "N",  # Normal beat (middle dot variant)
    ".": "N",  # Normal beat
    "N": "N",  # Normal beat
    "L": "N",  # Left bundle branch block beat
    "R": "N",  # Right bundle branch block beat
    "e": "N",  # Atrial escape beat
    "j": "N",  # Nodal escape beat
    # S class
    "A": "S",  # Atrial premature beat
    "a": "S",  # Aberrated atrial premature beat
    "J": "S",  # Nodal premature beat
    "S": "S",  # Supraventricular premature beat
    # V class
    "V": "V",  # Premature ventricular contraction
    "E": "V",  # Ventricular escape beat
    # F class
    "F": "F",  # Fusion of ventricular and normal beat
    # Q class
    "/": "Q",  # Paced beat
    "f": "Q",  # Fusion of paced and normal beat
    "Q": "Q",  # Unclassifiable beat
}


def map_symbol_main_class(symbol):
    """Return main arrhythmia class id (N/S/V/F/Q -> 0..4), else None."""
    main_class = SYMBOL_TO_MAIN_CLASS.get(symbol)
    if main_class is None:
        return None
    return MAIN_CLASS_TO_ID[main_class]


def map_symbol(symbol):
    """
    Binary mapping used by current window labeling pipeline:
    - 0: normal-like (N class)
    - 1: abnormal (S/V/F/Q classes)
    - None: ignored symbol
    """
    main_class_id = map_symbol_main_class(symbol)
    if main_class_id is None:
        return None
    if main_class_id == MAIN_CLASS_TO_ID["N"]:
        return 0
    return 1
