# CAIS Cluster Debug Summary (Runs 119390 & 120632)

## The Core Issue: Persistent Distributed Deadlock
Both training runs (119390 and 120632) failed due to a severe PyTorch Distributed deadlock, specifically an `ALLREDUCE` timeout triggered by the NCCL watchdog after 1,800 seconds (30 minutes).

`Watchdog caught collective operation timeout: WorkNCCL(SeqNum=9, OpType=ALLREDUCE, NumelIn=1, NumelOut=1, Timeout(ms)=1800000) ran for 1800056 milliseconds before timing out.`

## What Has Been Fixed So Far:
1. **vLLM Engine Initialization**: OOM and CUDA context collisions were resolved by shifting to the older vLLM V0 engine and forcibly isolating PyTorch `CUDA_VISIBLE_DEVICES` in the multiprocessing `spawn` function earlier in the pipeline.
2. **DeepSpeed ZeRO Corrupted LoRA Saving**: `save_pretrained` on Rank 0 was replacing the LoRAs with corrupted parameter shards because of ZeRO partitioning. This was fixed by using `accelerator.save_model(...)` which safely gathers shards across the network before saving.
3. **Distribution Strategy Mismatch**: Originally, Ranks 0/1 had the Victim, and Ranks 2/3 had the Attacker. This caused a local evaluation deadlock where ranks waited for missing models. We changed this to `all_ranks` so every GPU has every model.

## Why 120632 Failed (The Remaining Bug):
Despite `all_ranks` being enabled, the NCCL deadlock STILL occurs at `SeqNum=9` (around step 0 or step 1). The crash implies that one or more PyTorch ranks are becoming desynchronized during the local generation loop (MTSA simulation) taking longer than 30 minutes, or a specific `dist.barrier()` or Gradient `ALLREDUCE` is being hit asynchronously.

### Potential Culprits to Investigate Next:
* **vLLM Engine VRAM Exhaustion under `all_ranks`**: Loading 3 vLLM engines (Victim, Attacker, Judge) natively on EACH A100 GPU might be silently locking up or thrashing memory, causing a single rank to silently stall.
* **Hanging at local evaluation**: Deep iteration inside `_generate_rollouts_vllm` or `_get_victim_response`, possibly because vLLM `generate` requests are queueing endlessly on one rank but finishing on another.
* **RLHF Rollout Synching**: During Hugging Face `TRL` or `Accelerate` optimization steps, exactly 1 tensor or gradient might not be correctly shaped if a batch was prematurely skipped on a rank.
