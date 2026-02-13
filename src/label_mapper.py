# def map_symbol(symbol):
#     mapping = {
#         'N': 0,  # Normal
#         'V': 1,  # PVC
#         'A': 2,  # Atrial premature beat
#         'L': 3,  # Left bundle branch block (optional later)
#         'R': 4   # Right bundle branch block (optional later)
#     }

#     return mapping.get(symbol, None)  # Ignore others
def map_symbol(symbol):
    if symbol == 'N':
        return 0  # Normal
    elif symbol in ['V', 'A', 'L', 'R']:
        return 1  # Abnormal
    else:
        return None  # Ignore others
