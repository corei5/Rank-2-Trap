#!/bin/bash
#SBATCH --job-name=gjepa_w126
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=300G
#SBATCH --gres=gpu:l40s:2
#SBATCH --partition=p_48G
#SBATCH --output=/nfs/home/rabbyg/JEPA/Graph-JEPA/log/w126_%j.out
#SBATCH --error=/nfs/home/rabbyg/JEPA/Graph-JEPA/log/w126_%j.err

# W1 / W2 / W6 factorial test.
#   2 trainers {fallback, real} x 2 feature spaces {text, hetero}
# Exactly one factor differs between comparable cells, so a disagreement is
# attributable. A cell that cannot get what it asked for emits NO verdict.

set -uo pipefail

PROJECT_DIR="/nfs/home/rabbyg/JEPA/Graph-JEPA"
LOG_DIR="${PROJECT_DIR}/log"
OUT="${PROJECT_DIR}/checkpoints/w126"
CORPUS="papers"

# ── REQUIRED for the 'real' cells. Leave unset to run fallback cells only. ──
# export GJEPA_TRAIN_HOOK="train.paper_reason_gjepa:train_for_diag"

# ── Set this if the graph's node types are not named claim/method/result. ──
# Discover them from the [graph] node types line, then e.g.:
# export GJEPA_NODE_MAP="claim=claims,method=methods,result=key_results"

EPOCHS=100
INTERVENTION_EPOCHS=40
DOSES="0,0.5,1,3,10,30,100"
NQUERIES=4000
SEED=0
SMOKE=0

mkdir -p "${LOG_DIR}" "${OUT}"
STAMP="$(date +%Y%m%d_%H%M%S)"; JOBTAG="${SLURM_JOB_ID:-local}"
RUN_LOG="${LOG_DIR}/w126_${JOBTAG}_${STAMP}.log"

source /nfs/home/rabbyg/miniconda3/etc/profile.d/conda.sh
conda activate kgjepa
cd "${PROJECT_DIR}" || exit 1
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export HF_HOME="${PROJECT_DIR}/_hf_cache"
export SENTENCE_TRANSFORMERS_HOME="${PROJECT_DIR}/_st_cache"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MPLBACKEND=Agg
export MPLCONFIGDIR="${PROJECT_DIR}/_mpl_cache"
mkdir -p "${HF_HOME}" "${SENTENCE_TRANSFORMERS_HOME}" "${MPLCONFIGDIR}"

{
echo "════════════════════════════════════════════════════════════"
echo "  W1 / W2 / W6  FACTORIAL TEST"
echo "════════════════════════════════════════════════════════════"
echo "  Date   : $(date)"
echo "  Job    : ${JOBTAG}   Host: $(hostname)"
echo "  Out    : ${OUT}"
echo "  Corpus : ${CORPUS}"
nvidia-smi --query-gpu=name,memory.total --format=csv || true
python -c "import torch;print(' torch',torch.__version__,'cuda?',torch.cuda.is_available())"

# ── which cells are runnable ──
CELLS="fallback+text,fallback+hetero"
if [[ -n "${GJEPA_TRAIN_HOOK:-}" ]]; then
  CELLS="${CELLS},real+text,real+hetero"
  echo "  Trainer: ${GJEPA_TRAIN_HOOK}  -> running all 4 cells"
else
  echo "  Trainer: UNSET -> fallback cells only."
  echo "           W1/W2/W6 CANNOT be adjudicated for Graph-JEPA without this."
  echo "           export GJEPA_TRAIN_HOOK='train.paper_reason_gjepa:train_for_diag'"
fi
echo "  Nodemap: ${GJEPA_NODE_MAP:-<auto-resolve>}"
echo "  Cells  : ${CELLS}"

# ── show the graph's actual node types before anything else ──
echo ""
echo "  ---- graph introspection ----"
python - <<'PY'
import os, torch
p = os.path.join(os.environ["PROJECT_DIR"], "dataset", "papers", "processed",
                 "hetero_graphA.pt") if "PROJECT_DIR" in os.environ else None
try:
    from diag import adapters
    g = adapters.load_graph(adapters.get_spec("papers"))
    from diag.stages_w126 import _node_store, _get_x
    st = _node_store(g)
    for nt, e in st.items():
        x = _get_x(e)
        print(f"    node type {nt!r:24s} x={tuple(x.shape) if x is not None else None}")
    if hasattr(g, "edge_types"):
        for et in g.edge_types:
            print(f"    edge type {et}")
except Exception as e:
    print(f"    introspection failed: {e}")
PY

FLAGS=(--corpus "${CORPUS}" --out-dir "${OUT}" --cells "${CELLS}"
       --epochs "${EPOCHS}" --intervention-epochs "${INTERVENTION_EPOCHS}"
       --doses "${DOSES}" --n-queries "${NQUERIES}" --seed "${SEED}")
if [[ "${SMOKE}" == "1" ]]; then
  OUT="${OUT}_smoke"; mkdir -p "${OUT}"
  FLAGS=(--corpus "${CORPUS}" --out-dir "${OUT}" --cells "${CELLS}"
         --epochs 5 --intervention-epochs 3 --doses "0,1,10,100"
         --n-queries 300 --seed 0 --limit-papers 400)
fi

echo ""
echo "════════════════════════════════════════════════════════════"
python -m diag.stages_w126 "${FLAGS[@]}"
RC=$?

echo ""
echo "  ---- output ----"
ls -la "${OUT}"/*.json 2>/dev/null || true
echo ""
echo "════════════════════════════════════════════════════════════"
case ${RC} in
  0) echo "  ALL CELLS PRODUCED VERDICTS. rc=0" ;;
  2) echo "  SOME CELLS BLOCKED ON PROVENANCE (rc=2). This is the correct"
     echo "  outcome when a cell could not obtain the trainer or features it"
     echo "  asked for. Read the 'reason' fields in w126_${CORPUS}.json." ;;
  *) echo "  FAILED rc=${RC}" ;;
esac
echo "  prereg : ${OUT}/w126_prereg.json"
echo "  results: ${OUT}/w126_${CORPUS}.json"
echo "════════════════════════════════════════════════════════════"
exit ${RC}
} 2>&1 | tee "${RUN_LOG}"