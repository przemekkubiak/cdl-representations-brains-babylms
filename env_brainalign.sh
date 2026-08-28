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

# Run only the sessions that fit alongside the merge sweep on the shared filesystem.
# ses-7 is the large one (217 subjects, ~250 GB of transient patterns) and has aborted two runs;
# ses-5 and ses-9 complete inside ~90 GB. Clear ONLY_SESSIONS once the sweep has finished and the
# disk is ours, then rerun -- sessions with an RDM already present are skipped, so it resumes.
# With MAX_SUBJECTS capping every batch to the same size, the large session costs no more than a
# small one, so full task x session coverage is affordable again. Left empty deliberately.
export ONLY_SESSIONS="${ONLY_SESSIONS:-}"
export DISK_FLOOR_GB="${DISK_FLOOR_GB:-350}"

# Cap subjects per session batch. A session RDM is an atom -- it aggregates across every subject
# in the session -- so the ONLY way to shrink a batch is to build it from fewer subjects.
# Measured cost per subject, which varies a lot by task and is what three aborted runs got wrong:
#     Sem/ses-5   91 subj -> 180 patterns,  ~60 GB  = 0.66 GB/subject
#     Phon/ses-5 122 subj -> 231 patterns,  173 GB  = 1.42 GB/subject
# Sharing the filesystem with a merge sweep that aborts below 250 GB, the headroom above our own
# 350 GB floor is ~80 GB, so 40 subjects (~57 GB at the Phon rate) fits every task with margin.
# 40 subjects is a real RDM, just a noisier one than 122 would give. It gets full task x session
# coverage onto the Hub in about an hour and unblocks Tier 1, instead of a fourth abort with
# nothing to show. Clear MAX_SUBJECTS and rerun once the sweep is done to rebuild at full N --
# rdm_cache_hf.py push overwrites, so the cached RDMs upgrade in place.
# CLEARED 2026-08-28. The condition this cap existed for is gone: the merge sweep
# that aborted below 250 GB has finished (PICKUP.md), `/` reports ~810 GB free
# against our 350 GB floor, so the headroom is ~460 GB rather than ~80 GB. At the
# Phon rate above, ds001894's largest batch is 188 subjects x 1.42 GB = ~267 GB,
# which fits. Leaving it at 40 was silently costing real statistical power on
# every dataset -- ds002236's age bins came out at n=8..11 subjects where the
# cohort supports roughly double, and the noise ceiling IS a function of n, so it
# was deflating the denominator that every ceiling-normalised alignment number is
# divided by. rdm_cache_hf.py push overwrites, so the Hub-cached RDMs upgrade in
# place, exactly as the note above anticipated. 0 = all subjects.
export MAX_SUBJECTS="${MAX_SUBJECTS:-0}"
