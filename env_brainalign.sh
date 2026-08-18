# Bare-metal environment for the BrainAlign DevAI grid on this shared A100 box.
# Sourced by run_tiers_detached.sh and safe to source by hand.
#
# GPUs 0,1,2 are OURS. GPU 3 is reserved. GPUs 4-7 run a 96-hour merge sweep for
# another project (/root/mergeability) -- touching them destroys days of work.
export CUDA_VISIBLE_DEVICES=0,1,2

# Dedicated HF cache. $SCRATCH is unset on this box, so the runbook's
# "$SCRATCH/hf_cache" would resolve to /hf_cache. Do not share another
# project's cache -- that has caused failures here before.
export HF_HOME=/root/hf_cache_brainalign
export BACKUP_HF_REPO=BrainAlign/cdl-devai-results

# Token lives in a mode-600 file and is never hardcoded, printed or committed.
[ -f /root/.ms_hf_env ] && . /root/.ms_hf_env

export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
