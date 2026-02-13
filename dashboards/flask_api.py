from flask import Flask, jsonify
from src.data_loader import load_record
from src.pipeline import process_signal

app = Flask(__name__)
#dummy file ignore this
# Load first record
signal, fs = load_record("data/mitdb", "100")
windows = process_signal(signal, fs)

@app.route("/get_window/<int:idx>")
def get_window(idx):
    if idx < 0 or idx >= len(windows):
        return jsonify({"error": "Index out of range"})
    # Dummy label
    label = 1 if idx % 10 == 0 else 0
    return jsonify({
        "window_idx": idx,
        "label": label,
        "data": windows[idx].tolist()
    })

if __name__ == "__main__":
    app.run(debug=True)
