#!/bin/bash
#SBATCH --job-name=pymoo-grid
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16GB
#SBATCH --time=50:00:00
#SBATCH --gres=gpu:0
#SBATCH --account=pr_60_tandon_advanced
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
#SBATCH -a 0-63    # 8 problems × 8 algorithms = 64 array jobs

module purge

export SSL_CERT_FILE=/scratch/wx2178/cacert.pem
export HOME=/scratch/wx2178

unset XDG_RUNTIME_DIR
if [ "$SLURM_JOBTMP" != "" ]; then
    export XDG_RUNTIME_DIR=$SLURM_JOBTMP
fi

overlay_ext3=/scratch/wx2178/pytorch-example/my_pytorch.ext3

singularity exec --nv \
    --overlay $overlay_ext3:ro \
    /scratch/work/public/singularity/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif \
    /bin/bash -c "
        source /share/apps/anaconda3/2020.07/etc/profile.d/conda.sh
        conda activate /scratch/wx2178/.conda/envs/cns
        export PYTHONUNBUFFERED=1

        cd /scratch/wx2178/SNN/DarwinNeuron

        # === config ===
        DB=data/landscape-analysis.db
        CLI=src/OptimizerAnalysis/run_batch_compare.py
        OUT=data/MethodComparisonRuns

        TERM_ID=1
        PROBLEMS=(0 1 2 3 4 5 6 7)
        ALGOS=(0 1 2 3 4 5 6 7)

        NP=\${#PROBLEMS[@]}
        NA=\${#ALGOS[@]}
        PIDX=\$(( SLURM_ARRAY_TASK_ID % NP ))
        AIDX=\$(( SLURM_ARRAY_TASK_ID / NP ))

        PID=\${PROBLEMS[\$PIDX]}
        ALG_ID=\${ALGOS[\$AIDX]}

        echo '[task] pid='\$PID' algo_id='\$ALG_ID' term_id='\$TERM_ID

        python \$CLI \\
          --db \$DB \\
          --ids \$PID \\
          --termination-id \$TERM_ID \\
          --algorithm-ids \$ALG_ID \\
          --batch-size 516 \\
          --seeds 5,6,7 \\
          --multi-run \\
          --verbose \\
          --out \$OUT
    "
