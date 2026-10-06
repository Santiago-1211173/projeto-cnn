import os
import shutil

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ckpt_dir = os.path.join(PROJECT_ROOT, "outputs", "cifar100", "checkpoints")
legacy_dir = os.path.join(PROJECT_ROOT, "outputs", "cifar100", "checkpoints_v1_legacy")
new_ckpt_dir = os.path.join(PROJECT_ROOT, "outputs", "cifar100", "checkpoints_opt2_150e")

# 1. Backup existing to legacy
os.makedirs(legacy_dir, exist_ok=True)
for item in os.listdir(ckpt_dir):
    s = os.path.join(ckpt_dir, item)
    d = os.path.join(legacy_dir, item)
    if os.path.isfile(s):
        shutil.copy2(s, d)
print(f"Backed up {len(os.listdir(legacy_dir))} files to {legacy_dir}")

# 2. Copy new files into ckpt_dir
for item in os.listdir(new_ckpt_dir):
    s = os.path.join(new_ckpt_dir, item)
    d = os.path.join(ckpt_dir, item)
    if os.path.isfile(s):
        shutil.copy2(s, d)
print(f"Promoted new 74.27% model files into {ckpt_dir}")
