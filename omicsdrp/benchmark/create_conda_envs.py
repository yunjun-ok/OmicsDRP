import subprocess
import sys

def run_cmd(cmd):
    print(f"Running command: {' '.join(cmd)}")
    result = subprocess.run(cmd, stdout=sys.stdout, stderr=sys.stderr)
    if result.returncode != 0:
        print(f"Command failed with code {result.returncode}")
        raise Exception(f"Command failed: {' '.join(cmd)}")

def main():
    print("Starting conda environment creation for GraphDRP, DeepTTA, and PaccMann...")

    # 1. GraphDRP env setup
    print("\n==================== Setting up benchmark_graphdrp ====================")
    run_cmd(["conda", "create", "-y", "-n", "benchmark_graphdrp", "python=3.9"])
    python_graphdrp = "/home/mjcho/miniconda3/envs/benchmark_graphdrp/bin/python"
    pip_graphdrp = "/home/mjcho/miniconda3/envs/benchmark_graphdrp/bin/pip"
    run_cmd([pip_graphdrp, "install", "torch==2.3.0", "--index-url", "https://download.pytorch.org/whl/cu121"])
    run_cmd([pip_graphdrp, "install", "torch-scatter", "torch-sparse", "torch-cluster", "torch-spline-conv", "torch-geometric", "-f", "https://data.pyg.org/whl/torch-2.3.0+cu121.html"])
    run_cmd([pip_graphdrp, "install", "rdkit", "pandas", "numpy", "scipy", "matplotlib"])

    # 2. DeepTTA env setup
    print("\n==================== Setting up benchmark_deeptta ====================")
    run_cmd(["conda", "create", "-y", "-n", "benchmark_deeptta", "python=3.9"])
    pip_deeptta = "/home/mjcho/miniconda3/envs/benchmark_deeptta/bin/pip"
    run_cmd([pip_deeptta, "install", "torch==2.3.0", "--index-url", "https://download.pytorch.org/whl/cu121"])
    run_cmd([pip_deeptta, "install", "rdkit", "subword-nmt", "prettytable", "pandas", "numpy", "scikit-learn", "biopython", "pubchempy"])

    # 3. PaccMann env setup
    print("\n==================== Setting up benchmark_paccmann ====================")
    run_cmd(["conda", "create", "-y", "-n", "benchmark_paccmann", "python=3.9"])
    pip_paccmann = "/home/mjcho/miniconda3/envs/benchmark_paccmann/bin/pip"
    run_cmd([pip_paccmann, "install", "torch==2.3.0", "--index-url", "https://download.pytorch.org/whl/cu121"])
    run_cmd([pip_paccmann, "install", "rdkit", "pytoda", "pandas", "numpy", "scikit-learn"])
    # Run pip install -e . in PaccMann repo
    print("Installing PaccMann package in editable mode...")
    subprocess.run([pip_paccmann, "install", "-e", "."], cwd="/project/OmicsDRP_Review/Benchmark/PaccMann", stdout=sys.stdout, stderr=sys.stderr)

    print("\nAll environments created and configured successfully!")

if __name__ == "__main__":
    main()
