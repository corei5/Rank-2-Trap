#!/bin/bash
#SBATCH --job-name=gjepa_rebuttal
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=300G
#SBATCH --gres=gpu:l40s:2
#SBATCH --partition=p_48G
#SBATCH --output=/nfs/home/rabbyg/JEPA/Graph-JEPA/log/rebuttal_%j.out
#SBATCH --error=/nfs/home/rabbyg/JEPA/Graph-JEPA/log/rebuttal_%j.err

# NOTE: no `-e`. Stage failures are handled explicitly; `-e` plus `| tee` was
# producing the misleading "[FATAL] line 0: tee (rc=1)" messages.
set -uo pipefail

# ═══════════════════════════════════════════════════════════════ PATHS
PROJECT_DIR="/nfs/home/rabbyg/JEPA/Graph-JEPA"
LOG_DIR="${PROJECT_DIR}/log"
OUT="${PROJECT_DIR}/checkpoints/rebuttal_diag"
PAPER_FIG_DIR="${PROJECT_DIR}/paper/figures"

# ═══════════════════════════════════════════════════════════════ CORPORA
export GJEPA_ROOT="${PROJECT_DIR}"
export GJEPA_ARXIV_RAW="${GJEPA_ARXIV_RAW:-${PROJECT_DIR}/dataset/arxiv/raw}"
export GJEPA_ARXIV_PROC="${GJEPA_ARXIV_PROC:-${PROJECT_DIR}/dataset/arxiv/processed}"
export GJEPA_EXTRACTOR_A="llm:gpt-4o-mini"      # state this truthfully (Q4)
export GJEPA_EXTRACTOR_B="llm:gpt-4o-mini"

# <<< CHANGED: papers only for this run. Add "arxiv" back when the data lands;
# the autodetect block below will skip any corpus with no *.json anyway.
WANT_CORPORA="papers"

# Wire your own trainer here to diagnose the REAL model instead of the fallback:
# export GJEPA_TRAIN_HOOK="train.paper_reason_gjepa:train_for_diag"

# ═══════════════════════════════════════════════════════════════ STAGES
STAGES_PER_CORPUS="baseline,ridge,lossfloor,gradaudit,capacity,posctrl,poolladder,hardpool,extraction,phase"
STAGES_ONCE="faithful"

SEEDS="0,1,2,3,4"
CAPACITY_SEEDS="0"
POSCTRL_SEEDS="0,1,2,3,4"
EPOCHS=100
NQUERIES=4000
NLEXQUERIES=2000        # BM25 is ~500x costlier per query than a dot product
NEXTRACTDOCS=8000
NPERM=200
HARD_POOL_K="10,100,1000"
LADDER="2,5,10,50,100,500,1000"
SIM_N=20000
CLAIMED_RHO=0.9961      # the value printed in the submission
STAGE_TIMEOUT=5400      # 90 min per stage; a hang costs one stage, not the job

SMOKE=0
FORCE=0
FIGURES_ONLY=0

mkdir -p "${LOG_DIR}" "${OUT}"
STAMP="$(date +%Y%m%d_%H%M%S)"; JOBTAG="${SLURM_JOB_ID:-local}"
RUN_LOG="${LOG_DIR}/rebuttal_${JOBTAG}_${STAMP}.log"

# ═══════════════════════════════════════════════════════════════ ENV
source /nfs/home/rabbyg/miniconda3/etc/profile.d/conda.sh
conda activate kgjepa
cd "${PROJECT_DIR}" || exit 1
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
# NOTE: do NOT override CUDA_VISIBLE_DEVICES; Slurm already scoped the 2 GPUs.
export HF_HOME="${PROJECT_DIR}/_hf_cache"
export SENTENCE_TRANSFORMERS_HOME="${PROJECT_DIR}/_st_cache"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MPLBACKEND=Agg
export MPLCONFIGDIR="${PROJECT_DIR}/_mpl_cache"
mkdir -p "${HF_HOME}" "${SENTENCE_TRANSFORMERS_HOME}" "${MPLCONFIGDIR}"

FLAGS=(--seeds "${SEEDS}" --capacity-seeds "${CAPACITY_SEEDS}"
       --posctrl-seeds "${POSCTRL_SEEDS}" --epochs "${EPOCHS}"
       --n-queries "${NQUERIES}" --n-lex-queries "${NLEXQUERIES}"
       --n-extract-docs "${NEXTRACTDOCS}" --n-perm "${NPERM}"
       --hard-pool-k "${HARD_POOL_K}" --ladder "${LADDER}"
       --sim-n "${SIM_N}" --paper-claimed-rho "${CLAIMED_RHO}"
       --stage-timeout "${STAGE_TIMEOUT}" --log-grads)
[[ "${FORCE}" == "1" ]] && FLAGS+=(--force)

if [[ "${SMOKE}" == "1" ]]; then
  OUT="${OUT}_smoke"; mkdir -p "${OUT}"
  FLAGS=(--seeds 0 --capacity-seeds 0 --posctrl-seeds 0 --epochs 5
         --n-queries 500 --n-lex-queries 200 --n-extract-docs 500
         --n-perm 20 --hard-pool-k 10,100 --ladder 2,10,100
         --sim-n 4000 --paper-claimed-rho "${CLAIMED_RHO}"
         --stage-timeout 900 --limit-papers 400 --log-grads --force)
fi

{
echo "════════════════════════════════════════════════════════════"
echo "  GRAPH-JEPA — ICLR REBUTTAL DIAGNOSTICS"
echo "════════════════════════════════════════════════════════════"
echo "  Date    : $(date)"
echo "  Job     : ${JOBTAG}   Host: $(hostname)"
echo "  Out     : ${OUT}"
echo "  Trainer : ${GJEPA_TRAIN_HOOK:-diag.refmodel (FALLBACK)}"
echo "------------------------------------------------------------"
nvidia-smi --query-gpu=name,memory.total --format=csv || true
python -c "import torch,sklearn,scipy,matplotlib;print(' torch',torch.__version__,'cuda?',torch.cuda.is_available())"
python -c "import rank_bm25" 2>/dev/null || pip install -q rank_bm25
python -c "import scipy.sparse; print(' scipy.sparse OK')"
python -c "import py_spy" 2>/dev/null || pip install -q py-spy

# ---- corpus autodetect: skip anything with no JSON ----
CORPORA=""
for C in ${WANT_CORPORA}; do
  case "${C}" in
    arxiv) D="${GJEPA_ARXIV_RAW}" ;;
    *)     D="${PROJECT_DIR}/dataset/${C}/raw" ;;
  esac
  if compgen -G "${D}/*.json" > /dev/null 2>&1; then
    CORPORA+="${C} "
  else
    echo "  [skip] corpus '${C}': no *.json in ${D}"
  fi
done
CORPORA="${CORPORA% }"
if [[ -z "${CORPORA}" ]]; then
  echo "ERROR: no usable corpus. Set GJEPA_ARXIV_RAW or populate dataset/*/raw."
  exit 1
fi
echo "  Corpora : ${CORPORA}"
echo "  Stages  : ${STAGES_PER_CORPUS}  (+ once: ${STAGES_ONCE})"
echo "════════════════════════════════════════════════════════════"

python -c "import diag.cli" || { echo "ERROR: diag not importable from ${PROJECT_DIR}"; exit 1; }

if [[ "${FIGURES_ONLY}" == "1" ]]; then
  python -m diag.cli --figures-only --corpus "${CORPORA// /,}" --out-dir "${OUT}"
  exit 0
fi

RC=0
for C in ${CORPORA}; do
  echo ""
  echo "############################################################"
  echo "#  CORPUS ${C}"
  echo "############################################################"
  LOG="${LOG_DIR}/diag_${C}_${JOBTAG}_${STAMP}.log"
  python -m diag.cli --corpus "${C}" --stages "${STAGES_PER_CORPUS}" \
         --out-dir "${OUT}" "${FLAGS[@]}" 2>&1 | tee "${LOG}"
  r=${PIPESTATUS[0]}
  echo "  corpus ${C} finished rc=${r}  (log ${LOG})"
  [[ ${r} -ne 0 ]] && RC=${r}
done

if [[ -n "${STAGES_ONCE}" ]]; then
  echo ""
  echo "###### FAITHFULNESS (Graph-JEPA on its own benchmarks) ######"
  FIRST="${CORPORA%% *}"
  python -m diag.cli --corpus "${FIRST}" --stages "${STAGES_ONCE}" \
         --out-dir "${OUT}" "${FLAGS[@]}" 2>&1 | tee "${LOG_DIR}/faithful_${JOBTAG}.log"
fi

echo ""
echo "###################### CLAIM LEDGER #########################"
python -m diag.cli --corpus "${CORPORA// /,}" --stages ledger \
       --out-dir "${OUT}" --no-figures

echo ""; echo "  ---- results ----"; ls -la "${OUT}"/*.json 2>/dev/null || true
echo "  ---- figures ----"; ls -la "${OUT}/figures" 2>/dev/null || true
if [[ -n "${PAPER_FIG_DIR}" && -d "${OUT}/figures" ]]; then
  mkdir -p "${PAPER_FIG_DIR}"
  cp -f "${OUT}/figures/"*.pdf "${PAPER_FIG_DIR}/" 2>/dev/null || true
  cp -f "${OUT}/table_rebuttal.tex" "${PROJECT_DIR}/paper/" 2>/dev/null || true
  echo "  copied figures -> ${PAPER_FIG_DIR}/ and table_rebuttal.tex -> paper/"
fi

echo ""
echo "════════════════════════════════════════════════════════════"
echo "  DONE $(date)   rc=${RC}"
echo "  Paste into the rebuttal:  ${OUT}/table_rebuttal.tex"
echo "  Claim ledger           :  ${OUT}/claim_ledger.json"
echo "════════════════════════════════════════════════════════════"
exit ${RC}
} 2>&1 | tee "${RUN_LOG}"