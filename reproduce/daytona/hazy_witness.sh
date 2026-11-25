# Native witness on Hazy's megakernel, no instrumentation: before each decode step a buffer is filled with NaN (or 1e4);
# a consumer that reads it before the producer's writes are visible makes the logits NaN. Original vs release/acquire build.
# Args: N (generations per run) CANARIES (comma list)
N=${1:-1000}; CANARIES=${2:-attn_out,silu_out}
. /work/syncsan/reproduce/daytona/common.sh
. /work/syncsan/reproduce/daytona/hazy_setup.sh
cd /opt/mk
for c in ${CANARIES//,/ }; do for b in low-latency-llama acqrel; do
  step "witness $c nan $b"
  CANARY=$c CANARY_VALUE=nan timeout 1800 python /work/syncsan/reproduce/hazy/canary.py mode=mk model=unsloth/Llama-3.2-1B-Instruct ntok=64 num_warmup=0 num_iters=$N mk_dir=/opt/mk/demos/$b 2>&1 | tee /work/out/witness_${c}_$b.log | grep -a "WITNESS canary"
done; done
