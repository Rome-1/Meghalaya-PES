import os
import wandb
import subprocess

PROJECT = "sweep"
SWEEP_CONFIG = "mkezw377"

def main():
    subprocess.run(["python3", "-m", "anything_model", "--wandb_project", PROJECT], cwd="DeepForestcast/src", check=False)

if __name__ == "__main__":
    
    wandb.agent(SWEEP_CONFIG, function=main, project=PROJECT, count=1)
