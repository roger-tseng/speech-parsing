import json
import os

def save_experiment(experiment_file, step):
    data = dict(step=step)
    data = json.dumps(data, indent=4, sort_keys=True)
    with open(experiment_file, 'w') as f:
        f.write(data)

def update_best_model(path, out_path):
    if os.path.exists(out_path):
        os.remove(out_path)
    os.symlink(os.path.abspath(path), os.path.abspath(out_path))

def load_experiment(experiment_file):
    with open(experiment_file, 'r') as f:
        data = json.loads(f.read())
    return data
