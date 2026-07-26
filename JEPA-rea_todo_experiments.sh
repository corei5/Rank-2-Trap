#!/bin/bash
#SBATCH --job-name=papers_jepa
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=300G
#SBATCH --gres=gpu:l40s:2
#SBATCH --partition=p_48G
#SBATCH --output=/nfs/home/rabbyg/JEPA/Graph-JEPA/log/papers_%j.out
#SBATCH --error=/nfs/home/rabbyg/JEPA/Graph-JEPA/log/papers_%j.err

# -E  : ERR trap is inherited by functions (needed for the diagnostics below)
set -Eeuo pipefail

# ═══════════════════════════════════════════════════════════════
#  FATAL-ERROR DIAGNOSTICS
#  A bare `[[ cond ]] && cmd` that evaluates false returns 1. If it is the last
#  command of a function, `set -e` kills the shell with NO message. That is what
#  killed job 28935. The trap below makes any such death self-reporting.
# ═══════════════════════════════════════════════════════════════
p4_err () {
  local rc=$?
  echo "" >&2
  echo "[FATAL] ${BASH_SOURCE[0]}: line ${BASH_LINENO[0]}: command failed (rc=${rc})" >&2
  echo "[FATAL] last command: ${BASH_COMMAND}" >&2
  echo "[FATAL] call stack  : ${FUNCNAME[*]:-main}" >&2
  echo "" >&2
}
arm_err ()  { trap p4_err ERR; }
mute_err () { trap - ERR; }
arm_err

# ═══════════════════════════════════════════════════════════════
#  PATHS — EDIT THESE
# ═══════════════════════════════════════════════════════════════
PROJECT_DIR="/nfs/home/rabbyg/JEPA/Graph-JEPA"
LOG_DIR="${PROJECT_DIR}/log"
RAW_DIR="${PROJECT_DIR}/dataset/papers/raw"
PROC_DIR="${PROJECT_DIR}/dataset/papers/processed"
CONFIG="${PROJECT_DIR}/train/configs/papers.yaml"

HETERO_CACHE="${PROC_DIR}/hetero_graphA.pt"
RWSE_CACHE="${PROC_DIR}/rwse_pe.pt"

# Figures are copied here for the paper. Leave empty to skip the copy.
PAPER_FIG_DIR="${PROJECT_DIR}/paper/figures"

# ═══════════════════════════════════════════════════════════════
#  PART TOGGLES  (1 = run, 0 = skip)
# ═══════════════════════════════════════════════════════════════
RUN_PART1=0        # Graph-JEPA self-supervised training
RUN_PART2=0        # reasoning graph + link prediction
RUN_PART3=0        # hyperbolic Graph-JEPA ablation
RUN_PART4=1        # TODO experiments v2 + FIGURES

REBUILD_CACHE=0
REBUILD_HETERO=0
REBUILD_RWSE=0

# ═══════════════════════════════════════════════════════════════
#  PART 4 — PRE-PASSES
# ═══════════════════════════════════════════════════════════════
P4_SMOKE=0                # all stages, few epochs, 1 seed, separate out-dir
P4_SMOKE_EPOCHS=20
P4_TRIAGE=1               # rankinit + centering, 1 seed  (cheap + decisive)
P4_DRYRUN=0               # 1 = print the python commands, run nothing

# ═══════════════════════════════════════════════════════════════
#  PART 4 — main stage toggles     (approx. cost at 100 epochs, 5 seeds)
# ═══════════════════════════════════════════════════════════════
P4_RANKINIT=1      # untrained rank/DC vs node degree, depths 0-3     ~2 min, no training
P4_PROTOCOLR=1     # Protocol R baseline + whitened, raw & centred    ~7 min
P4_PERASPECT=1     # per-aspect rank / MRR / DC                       ~4 min
P4_CENTERING=1     # retrieval-frame sweep raw/center/rm-top/zca      ~4 min (3 seeds)
P4_ENCODER=1       # encoder depth 0-3 + residual + bypass            ~40 min (3 seeds)
P4_ORACLE=1        # oracle + BM25 / no-summary / no-overlap          ~17 min
P4_FIX=1           # pooling x loss x target-VICReg grid              ~21 min
P4_RHO=1           # synthetic inter-paper-redundancy sweep           ~1 min

# ---- stage options ----
P4_TEXT_CONTROLS=1
P4_FIX_WHITEN=0
P4_ENC_WHITEN=0
P4_ENC_LOSS="cos"          # cos | infonce
P4_ENC_MAX_SEEDS=3
P4_CENTERING_MAX_SEEDS=3

P4_EPOCHS=""               # "" = inherit EPOCHS from paper_reason_gjepa
P4_SEEDS="0,1,2,3,4"
P4_NQUERIES=4000
P4_OUT="${PROJECT_DIR}/checkpoints/todo_experiments"

# ═══════════════════════════════════════════════════════════════
#  PART 4 — figure toggles
# ═══════════════════════════════════════════════════════════════
P4_FIGURES=1
P4_FIGURES_ONLY=0
P4_ARCHIVE=1

mkdir -p "${LOG_DIR}"
STAMP="$(date +%Y%m%d_%H%M%S)"
JOBTAG="${SLURM_JOB_ID:-local}"
RUN_LOG="${LOG_DIR}/papers_run_${JOBTAG}_${STAMP}.log"
P4_LOG="${LOG_DIR}/p4_${JOBTAG}_${STAMP}.log"
P4_TRIAGE_LOG="${LOG_DIR}/p4_triage_${JOBTAG}_${STAMP}.log"
P4_SMOKE_LOG="${LOG_DIR}/p4_smoke_${JOBTAG}_${STAMP}.log"

OVERALL_RC=0

# ═══════════════════════════════════════════════════════════════
#  ENVIRONMENT
# ═══════════════════════════════════════════════════════════════
source /nfs/home/rabbyg/miniconda3/etc/profile.d/conda.sh
conda activate kgjepa || { echo "Conda environment activation failed"; exit 1; }
cd "${PROJECT_DIR}"

export PYTHONUNBUFFERED=1
# one visible GPU keeps NUM_GPUS_BILLED=1 in the python cost model honest
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export HF_HOME="${PROJECT_DIR}/_hf_cache"
export SENTENCE_TRANSFORMERS_HOME="${PROJECT_DIR}/_st_cache"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export MPLBACKEND=Agg
export MPLCONFIGDIR="${PROJECT_DIR}/_mpl_cache"
mkdir -p "${HF_HOME}" "${SENTENCE_TRANSFORMERS_HOME}" "${MPLCONFIGDIR}"

# ═══════════════════════════════════════════════════════════════
#  HELPERS      (every one ends with an explicit `return 0`)
# ═══════════════════════════════════════════════════════════════
P4_MODULE="train.paper_reason_todo_experiments"
EXTRA=()

# FIX: if/fi blocks instead of `cond && cmd`, plus explicit return 0.
p4_build_extra () {
  EXTRA=()
  if [[ "${P4_TEXT_CONTROLS}" == "1" ]]; then EXTRA+=(--text-controls); fi
  if [[ "${P4_FIX_WHITEN}"    == "1" ]]; then EXTRA+=(--fix-whiten);    fi
  if [[ "${P4_ENC_WHITEN}"    == "1" ]]; then EXTRA+=(--enc-whiten);    fi
  if [[ "${P4_FIGURES}"       == "0" ]]; then EXTRA+=(--no-figures);    fi
  EXTRA+=(--enc-loss "${P4_ENC_LOSS}")
  EXTRA+=(--enc-max-seeds "${P4_ENC_MAX_SEEDS}")
  EXTRA+=(--centering-max-seeds "${P4_CENTERING_MAX_SEEDS}")
  if [[ -n "${P4_EPOCHS}" ]]; then EXTRA+=(--epochs "${P4_EPOCHS}"); fi
  return 0
}

# p4_run <stages> <seeds> <outdir> [extra flags...]   -> returns python's rc
p4_run () {
  local stages="$1"; shift
  local seeds="$1";  shift
  local outdir="$1"; shift
  mkdir -p "${outdir}"
  echo "  >>> python -m ${P4_MODULE} --stages ${stages} --seeds ${seeds}" \
       "--n-queries ${P4_NQUERIES} --out-dir ${outdir} $*"
  if [[ "${P4_DRYRUN}" == "1" ]]; then
    echo "  (dry run: not executed)"
    return 0
  fi
  REBUILD_HETERO=0 python -m "${P4_MODULE}" \
    --stages "${stages}" \
    --seeds "${seeds}" \
    --n-queries "${P4_NQUERIES}" \
    --out-dir "${outdir}" \
    "$@"
  return $?
}

# run p4_run, tee to a log, never abort the script, echo the rc
# p4_run_logged <logfile> <stages> <seeds> <outdir> [extra...]
p4_run_logged () {
  local log="$1"; shift
  local rc=0
  mute_err
  set +e
  p4_run "$@" 2>&1 | tee "${log}"
  rc=${PIPESTATUS[0]}
  set -e
  arm_err
  echo "  (exit code ${rc}; log: ${log})"
  return "${rc}"
}

# decision-relevant lines out of a captured log
p4_verdicts () {
  local f="${1:-}"
  if [[ -z "${f}" || ! -f "${f}" ]]; then
    echo "  (no log to summarise)"; return 0
  fi
  echo "  ---- key lines ----"
  grep -E -e '\[rankinit\]' -e 'in-degree per node type' -e '\[centering\]' \
          -e 'best frame' -e 'FRAME MATTERS' -e 'frame does not help' \
          -e '\[enc ' -e 'ENCODER IS THE BOTTLENECK' -e 'depth is not the binding' \
          -e 'NOT distinguishable' -e 'trained baseline MRR' \
          -e 'training-free oracle' -e 'best encoder variant' \
          -e 'collapse threshold' -e '\[redundancy\]' -e '\[texts\]' \
          -e '\[oracle:' -e 'FIX SUMMARY' -e 'ENCODER SWEEP' \
          -e 'RETRIEVAL-FRAME SWEEP' \
          "${f}" 2>/dev/null | sed 's/^/    /' || true
  echo "  ---- headline summary ----"
  # FIX: 25 lines after the marker (the old sed range anchored `^==========$`,
  # which never matches python's 90-character rule).
  grep -m1 -A 25 'HEADLINE SUMMARY' "${f}" 2>/dev/null | sed 's/^/    /' || true
  return 0
}

p4_has () {   # p4_has <log> <pattern>
  local f="$1" pat="$2"
  [[ -f "${f}" ]] && grep -q -- "${pat}" "${f}" 2>/dev/null
}

{
  echo "════════════════════════════════════════════════════════════"
  echo "  KG-JEPA (papers) — run log   [PART 1-4]   (experiments v2)"
  echo "════════════════════════════════════════════════════════════"
  echo "  Date        : $(date)"
  echo "  Host        : $(hostname)"
  echo "  Job ID      : ${JOBTAG}"
  echo "  Bash        : ${BASH_VERSION}"
  echo "  PARTS       : P1=${RUN_PART1} P2=${RUN_PART2} P3=${RUN_PART3} P4=${RUN_PART4}"
  echo "  P4 pre      : smoke=${P4_SMOKE} triage=${P4_TRIAGE} dryrun=${P4_DRYRUN}"
  echo "  P4 stages   : rankinit=${P4_RANKINIT} protocolr=${P4_PROTOCOLR}" \
       "peraspect=${P4_PERASPECT} centering=${P4_CENTERING} encoder=${P4_ENCODER}" \
       "oracle=${P4_ORACLE} fix=${P4_FIX} rho=${P4_RHO}"
  echo "  P4 opts     : text=${P4_TEXT_CONTROLS} enc_loss=${P4_ENC_LOSS}" \
       "enc_seeds=${P4_ENC_MAX_SEEDS} ctr_seeds=${P4_CENTERING_MAX_SEEDS}" \
       "epochs='${P4_EPOCHS}'"
  echo "  P4 figures  : ${P4_FIGURES}  (figures_only=${P4_FIGURES_ONLY})"
  echo "  CUDA devices: ${CUDA_VISIBLE_DEVICES}"
  echo "  Python      : $(which python) ($(python --version 2>&1))"
  echo "------------------------------------------------------------"
  nvidia-smi || echo "  (nvidia-smi unavailable)"
  python -c "import torch, matplotlib, sklearn, scipy; print('   torch', torch.__version__, '| mpl', matplotlib.__version__, '| sklearn', sklearn.__version__, '| scipy', scipy.__version__, '| cuda?', torch.cuda.is_available())"
  echo "════════════════════════════════════════════════════════════"; echo

  NUM_JSON=$( { find "${RAW_DIR}" -maxdepth 1 -type f -name '*.json' 2>/dev/null || true; } | wc -l )
  if [[ "${NUM_JSON}" -eq 0 ]]; then echo "ERROR: no .json in ${RAW_DIR}"; exit 1; fi
  echo "  Found ${NUM_JSON} raw JSON files."; echo

  # ════════════════════════════════════════════════════════════
  #  PART 1
  # ════════════════════════════════════════════════════════════
  if [[ "${RUN_PART1}" == "1" ]]; then
    echo "############ PART 1 : train.papers ############"
    if [[ "${REBUILD_CACHE}" == "1" ]]; then
      rm -f "${PROC_DIR}/papers_coarse.pt" "${PROC_DIR}/papers_full.pt" \
            "${PROC_DIR}/label_map.json" "${PROC_DIR}/pre_transform.pt" \
            "${PROC_DIR}/pre_filter.pt" 2>/dev/null || true
    fi
    python -m train.papers --config "${CONFIG}" device 0
    echo "  PART 1 done at $(date)"; echo
  else echo "  [skip] PART 1"; echo; fi

  # ════════════════════════════════════════════════════════════
  #  PART 2
  # ════════════════════════════════════════════════════════════
  if [[ "${RUN_PART2}" == "1" ]]; then
    echo "############ PART 2 : train.paper_reason ############"
    if [[ "${REBUILD_HETERO}" == "1" ]]; then
      rm -f "${HETERO_CACHE}" || true
    elif [[ ! -f "${HETERO_CACHE}" ]]; then
      echo "  ERROR: cache missing: ${HETERO_CACHE}"; exit 1
    fi
    if [[ "${REBUILD_HETERO}" == "1" ]]; then REBUILD_FLAG="True"; else REBUILD_FLAG="False"; fi
    python - "${RAW_DIR}" "${HETERO_CACHE}" "${REBUILD_FLAG}" <<'PY'
import sys
from core.data_utils.paper_graph import build_hetero_graph
raw_dir, cache, rebuild = sys.argv[1], sys.argv[2], (sys.argv[3] == "True")
data, meta = build_hetero_graph(raw_dir, cache, rebuild=rebuild)
print(">>> meta:", meta); print(data)
PY
    REBUILD_HETERO=0 python -m train.paper_reason
    echo "  PART 2 done at $(date)"; echo
  else echo "  [skip] PART 2"; echo; fi

  # ════════════════════════════════════════════════════════════
  #  PART 3
  # ════════════════════════════════════════════════════════════
  if [[ "${RUN_PART3}" == "1" ]]; then
    echo "############ PART 3 : build_rwse -> paper_reason_gjepa ############"
    if [[ ! -f "${HETERO_CACHE}" ]]; then echo "  ERROR: ${HETERO_CACHE} missing"; exit 1; fi
    if [[ "${REBUILD_RWSE}" == "1" ]]; then
      rm -f "${RWSE_CACHE}" || true
      REBUILD_HETERO=0 python -m train.build_rwse
    elif [[ ! -f "${RWSE_CACHE}" ]]; then
      REBUILD_HETERO=0 python -m train.build_rwse
    fi
    if [[ ! -f "${RWSE_CACHE}" ]]; then echo "  ERROR: RWSE cache not produced"; exit 1; fi
    REBUILD_HETERO=0 python -m train.paper_reason_gjepa
    echo "  PART 3 done at $(date)"; echo
  else echo "  [skip] PART 3"; echo; fi

  # ════════════════════════════════════════════════════════════
  #  PART 4 — TODO EXPERIMENTS v2 + FIGURES
  # ════════════════════════════════════════════════════════════
  if [[ "${RUN_PART4}" == "1" ]]; then
    echo "############################################################"
    echo "#  PART 4 : ${P4_MODULE}"
    echo "############################################################"

    if [[ "${P4_FIGURES_ONLY}" == "1" ]]; then
      # ---------- figures-only fast path ----------
      echo "  P4_FIGURES_ONLY=1  ->  re-rendering figures only"
      for f in "${P4_OUT}/todo_experiments_results.json" "${P4_OUT}/artifacts.npz"; do
        if [[ ! -f "${f}" ]]; then
          echo "  ERROR: missing ${f} (run the full PART 4 once first)"; exit 1
        fi
      done
      python -m "${P4_MODULE}" --figures-only --out-dir "${P4_OUT}"
    else
      # ---------- required caches ----------
      echo "  checking required caches ..."
      for f in "${HETERO_CACHE}" "${RWSE_CACHE}"; do
        if [[ ! -f "${f}" ]]; then
          echo "  ERROR: required cache missing: ${f}"
          echo "         Run PART 2 (hetero) and PART 3 stage 3a (RWSE) first."
          exit 1
        fi
        echo "    OK  ${f}"
      done

      FIRST_SEED="${P4_SEEDS%%,*}"
      echo "  first seed for pre-passes: ${FIRST_SEED}"
      p4_build_extra
      echo "  shared extra args: ${EXTRA[*]:-none}"
      echo

      # ---------- SMOKE TEST ----------
      if [[ "${P4_SMOKE}" == "1" ]]; then
        echo "------------------------------------------------------------"
        echo "  SMOKE TEST: all stages, ${P4_SMOKE_EPOCHS} epochs, seed ${FIRST_SEED}"
        echo "------------------------------------------------------------"
        SMOKE_RC=0
        p4_run_logged "${P4_SMOKE_LOG}" \
          "rankinit,protocolr,peraspect,centering,encoder,oracle,fix,rho" \
          "${FIRST_SEED}" "${P4_OUT}_smoke" \
          --epochs "${P4_SMOKE_EPOCHS}" --enc-max-seeds 1 \
          --centering-max-seeds 1 --no-figures || SMOKE_RC=$?
        if [[ "${SMOKE_RC}" -ne 0 ]]; then
          echo "  ERROR: smoke test failed (rc=${SMOKE_RC}). Not starting the full run."
          exit "${SMOKE_RC}"
        fi
        echo "  SMOKE TEST OK at $(date)"; echo
      fi

      # ---------- TRIAGE ----------
      if [[ "${P4_TRIAGE}" == "1" ]]; then
        echo "------------------------------------------------------------"
        echo "  TRIAGE (cheap + decisive), seed ${FIRST_SEED}:"
        echo "    rankinit  : untrained rank/DC vs node in-degree (no training)"
        echo "    centering : does the retrieval frame rescue MRR?"
        echo "------------------------------------------------------------"
        TRIAGE_RC=0
        p4_run_logged "${P4_TRIAGE_LOG}" "rankinit,centering" \
          "${FIRST_SEED}" "${P4_OUT}_triage" \
          --centering-max-seeds 1 --no-figures || TRIAGE_RC=$?
        if [[ "${TRIAGE_RC}" -ne 0 ]]; then
          echo "  [warn] triage exited rc=${TRIAGE_RC}; continuing to the main sweep"
        fi
        echo
        echo "  ===== TRIAGE VERDICT ====="
        p4_verdicts "${P4_TRIAGE_LOG}"
        echo "  =========================="
        echo
      fi

      # ---------- MAIN SWEEP ----------
      STAGES=""
      if [[ "${P4_RANKINIT}"  == "1" ]]; then STAGES="${STAGES},rankinit";  fi
      if [[ "${P4_PROTOCOLR}" == "1" ]]; then STAGES="${STAGES},protocolr"; fi
      if [[ "${P4_PERASPECT}" == "1" ]]; then STAGES="${STAGES},peraspect"; fi
      if [[ "${P4_CENTERING}" == "1" ]]; then STAGES="${STAGES},centering"; fi
      if [[ "${P4_ENCODER}"   == "1" ]]; then STAGES="${STAGES},encoder";   fi
      if [[ "${P4_ORACLE}"    == "1" ]]; then STAGES="${STAGES},oracle";    fi
      if [[ "${P4_FIX}"       == "1" ]]; then STAGES="${STAGES},fix";       fi
      if [[ "${P4_RHO}"       == "1" ]]; then STAGES="${STAGES},rho";       fi
      STAGES="${STAGES#,}"

      if [[ -z "${STAGES}" ]]; then
        echo "  [skip] PART 4 main sweep: all stage toggles are 0"
      else
        echo "------------------------------------------------------------"
        echo "  MAIN SWEEP"
        echo "    stages    : ${STAGES}"
        echo "    seeds     : ${P4_SEEDS}"
        echo "    n_queries : ${P4_NQUERIES}"
        echo "    extra     : ${EXTRA[*]:-none}"
        echo "    out dir   : ${P4_OUT}"
        echo "------------------------------------------------------------"
        mkdir -p "${P4_OUT}"
        MAIN_RC=0
        p4_run_logged "${P4_LOG}" "${STAGES}" "${P4_SEEDS}" "${P4_OUT}" \
          ${EXTRA[@]+"${EXTRA[@]}"} || MAIN_RC=$?
        if [[ "${MAIN_RC}" -ne 0 ]]; then
          echo "  ERROR: main sweep exited rc=${MAIN_RC} (partial results may exist)"
          OVERALL_RC="${MAIN_RC}"
        fi
      fi
    fi

    # ---------- artefacts ----------
    echo
    echo "  ---- artefacts ----"
    ls -la "${P4_OUT}" 2>/dev/null || true
    echo "  ---- figures ----"
    ls -la "${P4_OUT}/figures" 2>/dev/null || true
    N_FIG=$( { ls -1 "${P4_OUT}/figures/"*.pdf 2>/dev/null || true; } | wc -l )
    echo "  figures produced: ${N_FIG} PDFs (expected up to 15)"

    # ---------- timestamped archive ----------
    if [[ "${P4_ARCHIVE}" == "1" && -f "${P4_OUT}/todo_experiments_results.json" ]]; then
      mkdir -p "${P4_OUT}/archive"
      cp -f "${P4_OUT}/todo_experiments_results.json" \
            "${P4_OUT}/archive/results_${STAMP}.json" || true
      cp -f "${P4_OUT}/artifacts.npz" \
            "${P4_OUT}/archive/artifacts_${STAMP}.npz" 2>/dev/null || true
      echo "  archived -> ${P4_OUT}/archive/results_${STAMP}.json"
    fi

    # ---------- copy figures next to the paper ----------
    if [[ -n "${PAPER_FIG_DIR}" && -d "${P4_OUT}/figures" ]]; then
      mkdir -p "${PAPER_FIG_DIR}"
      cp -f "${P4_OUT}/figures/"*.pdf "${PAPER_FIG_DIR}/" 2>/dev/null || true
      echo "  copied PDFs -> ${PAPER_FIG_DIR}/"
      echo "  use e.g.:  \\includegraphics[width=\\linewidth]{figures/fig12_encoder_depth.pdf}"
    fi

    # ---------- verdicts + next actions ----------
    if [[ -f "${P4_LOG}" ]]; then
      echo
      echo "  ===================== MAIN VERDICT ====================="
      p4_verdicts "${P4_LOG}"
      echo "  ========================================================"
      echo
      echo "  ---- NEXT ACTIONS ----"
      if p4_has "${P4_LOG}" "FRAME MATTERS"; then
        echo "    * The retrieval FRAME matters: the trained representation DOES"
        echo "      contain the signal, in a DC-dominated basis. Rebuild the paper"
        echo "      around 'centre/whiten the retrieval space'; fig13 is the"
        echo "      centrepiece and fig2 supplies the mechanism."
      elif p4_has "${P4_LOG}" "frame does not help"; then
        echo "    * Centring does NOT help => the representation genuinely lacks"
        echo "      instance identity. The encoder story stands; lead with fig12/fig14."
      fi
      if p4_has "${P4_LOG}" "ENCODER IS THE BOTTLENECK"; then
        echo "    * Encoder over-smoothing CONFIRMED (depth 0 >> depth 3). Make fig12"
        echo "      the central experiment; promote the degree-monotone law (fig14)."
      elif p4_has "${P4_LOG}" "depth is not the binding"; then
        echo "    * Depth is not the binding factor. Remaining suspects: the RWSE"
        echo "      query code and the context-mixer path. Add a query-side ablation."
      fi
      if p4_has "${P4_LOG}" "NOT distinguishable"; then
        echo "    * Baseline is indistinguishable from random: report the permutation"
        echo "      p-value, NOT a ratio against chance (no '4690x')."
      fi
      if p4_has "${P4_LOG}" "EDIT TEXT_FIELD_MAP"; then
        echo "    * Text extraction still failing: read the [schema] dump above,"
        echo "      extend TEXT_FIELD_MAP, then re-run:"
        echo "        --stages oracle --text-controls"
      fi
      echo "    * Re-render figures any time:"
      echo "        python -m ${P4_MODULE} --figures-only --out-dir ${P4_OUT}"
    fi

    echo "  PART 4 done at $(date)"; echo
  else echo "  [skip] PART 4 (RUN_PART4=0)"; echo; fi

  echo "════════════════════════════════════════════════════════════"
  echo "  ALL DONE at $(date)   (overall rc=${OVERALL_RC})"
  echo "════════════════════════════════════════════════════════════"

} 2>&1 | tee "${RUN_LOG}"

echo "Full run log      : ${RUN_LOG}"
if [[ -f "${P4_LOG}" ]];        then echo "PART 4 stage log  : ${P4_LOG}";        fi
if [[ -f "${P4_TRIAGE_LOG}" ]]; then echo "PART 4 triage log : ${P4_TRIAGE_LOG}"; fi
if [[ -f "${P4_SMOKE_LOG}" ]];  then echo "PART 4 smoke log  : ${P4_SMOKE_LOG}";  fi
exit "${OVERALL_RC}"
