#!/bin/bash
#SBATCH --job-name=darwin_sweep
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=20GB
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --account=pr_60_tandon_advanced
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

# === Command ===
# sbatch --array=0-9 /scratch/wx2178/SNN/DarwinNeuron/DarwinNeuron_hpc.SBATCH

# === Mode ===
# MODE="offline"
MODE="online"

# === environment ===
module purge
export SSL_CERT_FILE=/scratch/wx2178/cacert.pem
export HOME=/scratch/wx2178
export JUPYTER_RUNTIME_DIR=$HOME/jupyter_runtime
mkdir -p $JUPYTER_RUNTIME_DIR

export WANDB_DIR=$HOME/wandb_local
export WANDB_CACHE_DIR=$WANDB_DIR/cache
export WANDB_ARTIFACT_DIR=$WANDB_DIR/artifacts
mkdir -p $WANDB_ARTIFACT_DIR

# XDG_RUNTIME_DIR
unset XDG_RUNTIME_DIR
if [ "$SLURM_JOBTMP" != "" ]; then
    export XDG_RUNTIME_DIR=$SLURM_JOBTMP
fi

# === Singularity Overlay ===
overlay_ext3=$HOME/pytorch-example/my_pytorch.ext3
container_img=/scratch/work/public/singularity/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif

# === training ===
singularity exec --nv \
    --overlay $overlay_ext3:ro \
    $container_img \
    /bin/bash -c "
        source /share/apps/anaconda3/2020.07/etc/profile.d/conda.sh
        conda activate /scratch/wx2178/.conda/envs/cns

        export HOME=/scratch/wx2178
        export PATH=\$HOME/.local/bin:\$PATH
        export PYTHONUNBUFFERED=1
        export WANDB_API_KEY=
        export WANDB_DIR=$WANDB_DIR
        export WANDB_CACHE_DIR=$WANDB_CACHE_DIR
        export WANDB_ARTIFACT_DIR=$WANDB_ARTIFACT_DIR

        # online/offline switch
        if [ \"$MODE\" = \"offline\" ]; then
            export WANDB_MODE=offline
        fi

        echo '=== Run Sweep ==='
        cd /scratch/wx2178/SNN/DarwinNeuron
        python sweep_on_HPC.py

        # if offline：sync runs
        if [ \"$MODE\" = \"offline\" ]; then
            echo '=== Syncing offline logs ==='
            for d in $WANDB_DIR/offline-run-*; do
                /scratch/wx2178/.conda/envs/cns/bin/wandb sync \"\$d\"
            done
        fi
    "
