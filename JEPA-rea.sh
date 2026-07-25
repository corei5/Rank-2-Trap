#!/bin/bash
#SBATCH --job-name=papers_jepa
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=300G
#SBATCH --gres=gpu:l40s:2
#SBATCH --partition=p_48G

# ── log files (separate stdout/stderr, %j = SLURM job id) ──
#SBATCH --output=/nfs/home/rabbyg/JEPA/Graph-JEPA/log/papers_%j.out
#SBATCH --error=/nfs/home/rabbyg/JEPA/Graph-JEPA/log/papers_%j.err

set -euo pipefail

# ═══════════════════════════════════════════════════════════════
#  PATHS — EDIT THESE to match your setup
# ═══════════════════════════════════════════════════════════════
PROJECT_DIR="/nfs/home/rabbyg/JEPA/Graph-JEPA"     # repo root (contains core/, train/, dataset/)
LOG_DIR="${PROJECT_DIR}/log"
RAW_DIR="${PROJECT_DIR}/dataset/papers/raw"
PROC_DIR="${PROJECT_DIR}/dataset/papers/processed"
CONFIG="${PROJECT_DIR}/train/configs/papers.yaml"  # the config train.papers reads

# Part 2 hetero-graph cache (built by core/data_utils/paper_graph.py)
HETERO_CACHE="${PROC_DIR}/hetero_graphA.pt"

# Part 3 RWSE positional-encoding cache (built by train.build_rwse)
RWSE_CACHE="${PROC_DIR}/rwse_pe.pt"

# ═══════════════════════════════════════════════════════════════
#  STAGE TOGGLES  (set to 1 to run, 0 to skip)
# ═══════════════════════════════════════════════════════════════
RUN_PART1=0        # ← OFF : skip Graph-JEPA self-supervised training
RUN_PART2=1        # ← ON  : run reasoning graph + link prediction (v5)
RUN_PART3=1        # ← ON  : run FAITHFUL Graph-JEPA ablation (RWSE + A1 probe + A2 patch-ret)

# Rebuild caches (1 = force clean rebuild).
REBUILD_CACHE=1        # (ignored while RUN_PART1=0)
REBUILD_HETERO=1       # ← 1 : BUILD hetero_graphA.pt (first run). Set 0 to reuse later.
REBUILD_RWSE=1         # ← 1 : BUILD rwse_pe.pt (Part-3 Stage 3a). Set 0 to reuse later.

mkdir -p "${LOG_DIR}"

# combined run-log (everything, with a timestamp) in addition to SLURM .out/.err
RUN_LOG="${LOG_DIR}/papers_run_${SLURM_JOB_ID:-local}_$(date +%Y%m%d_%H%M%S).log"

# ═══════════════════════════════════════════════════════════════
#  ENVIRONMENT
# ═══════════════════════════════════════════════════════════════
source /nfs/home/rabbyg/miniconda3/etc/profile.d/conda.sh
conda activate kgjepa || { echo "Conda environment activation failed"; exit 1; }
cd "${PROJECT_DIR}"

export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export HF_HOME="${PROJECT_DIR}/_hf_cache"
export SENTENCE_TRANSFORMERS_HOME="${PROJECT_DIR}/_st_cache"
export TOKENIZERS_PARALLELISM=false
mkdir -p "${HF_HOME}" "${SENTENCE_TRANSFORMERS_HOME}"

# ═══════════════════════════════════════════════════════════════
#  RUN  (tee everything into RUN_LOG; 2>&1 folds stderr into stdout)
# ═══════════════════════════════════════════════════════════════
{
  echo "════════════════════════════════════════════════════════════"
  echo "  KG-JEPA  (papers)  —  run log   [PART 2 + PART 3]"
  echo "════════════════════════════════════════════════════════════"
  echo "  Date        : $(date)"
  echo "  Host        : $(hostname)"
  echo "  Job ID      : ${SLURM_JOB_ID:-N/A}"
  echo "  Project     : ${PROJECT_DIR}"
  echo "  Config      : ${CONFIG}"
  echo "  Raw dir     : ${RAW_DIR}"
  echo "  Hetero cache: ${HETERO_CACHE}"
  echo "  RWSE cache  : ${RWSE_CACHE}"
  echo "  RUN_PART1   : ${RUN_PART1}   RUN_PART2: ${RUN_PART2}   RUN_PART3: ${RUN_PART3}"
  echo "  REBUILD_HETERO: ${REBUILD_HETERO}   REBUILD_RWSE: ${REBUILD_RWSE}"
  echo "  CUDA devices: ${CUDA_VISIBLE_DEVICES}"
  echo "  Python      : $(which python)  ($(python --version 2>&1))"
  echo "------------------------------------------------------------"
  nvidia-smi || echo "  (nvidia-smi unavailable)"
  echo "  Torch CUDA  :"
  python -c "import torch; print('   torch', torch.__version__, '| cuda?', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
  echo "════════════════════════════════════════════════════════════"
  echo

  # ── sanity: make sure raw data exists ──
  NUM_JSON=$( { find "${RAW_DIR}" -maxdepth 1 -type f -name '*.json' 2>/dev/null || true; } | wc -l )
  if [[ "${NUM_JSON}" -eq 0 ]]; then
    echo "ERROR: no .json files found in ${RAW_DIR}"
    exit 1
  fi
  echo "  Found ${NUM_JSON} raw JSON files. Sample (first 5):"
  find "${RAW_DIR}" -maxdepth 1 -type f -name '*.json' 2>/dev/null | head -5 | sed 's/^/    /' || true
  echo

  # ════════════════════════════════════════════════════════════
  #  PART 1 — Graph-JEPA self-supervised training  (SKIPPED)
  # ════════════════════════════════════════════════════════════
  if [[ "${RUN_PART1}" == "1" ]]; then
    echo "############################################################"
    echo "#  PART 1 : Graph-JEPA  (train.papers)"
    echo "############################################################"

    if [[ "${REBUILD_CACHE}" == "1" ]]; then
      echo "  REBUILD_CACHE=1  ->  removing Part-1 caches only"
      rm -f "${PROC_DIR}/papers_coarse.pt" \
            "${PROC_DIR}/papers_full.pt"   \
            "${PROC_DIR}/label_map.json"   \
            "${PROC_DIR}/pre_transform.pt" \
            "${PROC_DIR}/pre_filter.pt" 2>/dev/null || true
    fi

    echo "------------------------------------------------------------"
    echo "  Pre-flight: dataset build + class count"
    echo "------------------------------------------------------------"
    python - <<'PY'
from core.config import cfg, update_cfg
from core.get_data import create_dataset, calculate_stats
cfg.merge_from_file('train/configs/papers.yaml')
update_cfg(cfg)
dataset, _, _ = create_dataset(cfg)
print(f">>> num graphs : {len(dataset)}")
print(f">>> num_classes: {dataset.num_classes}")
calculate_stats(dataset)
PY
    echo

    echo "------------------------------------------------------------"
    echo "  Training: train.papers"
    echo "------------------------------------------------------------"
    python -m train.papers --config "${CONFIG}" device 0
    echo "  PART 1 done at $(date)"
    echo
  else
    echo "  [skip] PART 1 (RUN_PART1=0)"
    echo
  fi

  # ════════════════════════════════════════════════════════════
  #  PART 2 — Heterogeneous reasoning graph + link prediction (v5)
  # ════════════════════════════════════════════════════════════
  if [[ "${RUN_PART2}" == "1" ]]; then
    echo "############################################################"
    echo "#  PART 2 : Reasoning graph  (train.paper_reason)"
    echo "############################################################"

    # ── citation-coverage audit (do papers cite each other?) ──
    echo "------------------------------------------------------------"
    echo "  Citation-coverage audit (grep over raw JSONs)"
    echo "------------------------------------------------------------"
    N_REFS=$( { grep -rl 'oa_referenced_works": "\[' "${RAW_DIR}" 2>/dev/null || true; } | wc -l )
    N_OAID=$( { grep -rl '"openalex_id": "http'      "${RAW_DIR}" 2>/dev/null || true; } | wc -l )
    echo "  files with non-empty oa_referenced_works : ${N_REFS}"
    echo "  files with openalex_id                   : ${N_OAID}"
    echo "  total json files                         : ${NUM_JSON}"
    echo "  (if openalex_id coverage is low, 'cites' edges will be sparse"
    echo "   and L2 will fall back to predicting 'makes' edges.)"
    echo

    # ── decide build vs reuse ──
    if [[ "${REBUILD_HETERO}" == "1" ]]; then
      echo "  REBUILD_HETERO=1  ->  removing ${HETERO_CACHE} (will rebuild)"
      rm -f "${HETERO_CACHE}" 2>/dev/null || true
    else
      echo "  REBUILD_HETERO=0  ->  REUSING existing ${HETERO_CACHE}"
      if [[ ! -f "${HETERO_CACHE}" ]]; then
        echo "  ERROR: cache not found and REBUILD_HETERO=0."
        echo "         Set REBUILD_HETERO=1 to build it, or check the path."
        exit 1
      fi
    fi

    # ── (1) build (or load) the hetero graph ONCE here ──
    echo "------------------------------------------------------------"
    echo "  Build/Load hetero graph  (core.data_utils.paper_graph)"
    echo "------------------------------------------------------------"
    REBUILD_FLAG=$([[ "${REBUILD_HETERO}" == "1" ]] && echo "True" || echo "False")
    python - "${RAW_DIR}" "${HETERO_CACHE}" "${REBUILD_FLAG}" <<'PY'
import sys
from core.data_utils.paper_graph import build_hetero_graph
raw_dir, cache, rebuild = sys.argv[1], sys.argv[2], (sys.argv[3] == "True")
data, meta = build_hetero_graph(raw_dir, cache, rebuild=rebuild)
print(">>> meta:", meta)
print(">>> node/edge summary:")
print(data)
PY
    echo

    # ── (2) train reasoning (L2a) + link prediction (L2b) ──
    #    graph is already built above -> force REUSE here (no double-embed)
    echo "------------------------------------------------------------"
    echo "  Training: train.paper_reason  (reasoning + link prediction)"
    echo "------------------------------------------------------------"
    REBUILD_HETERO=0 python -m train.paper_reason
    echo "  PART 2 done at $(date)"
    echo
  else
    echo "  [skip] PART 2 (RUN_PART2=0)"
    echo
  fi

  # ════════════════════════════════════════════════════════════
  #  PART 3 — FAITHFUL Graph-JEPA ablation (RWSE + A1 probe + A2 patch-ret)
  #           Reuses the hetero cache built in PART 2 (no re-embed).
  #           Stage 3a : build_rwse.py  -> rwse_pe.pt   (dynamic schema + RWSE)
  #           Stage 3b : paper_reason_gjepa.py          (GINE + masked JEPA)
  # ════════════════════════════════════════════════════════════
  if [[ "${RUN_PART3}" == "1" ]]; then
    echo "############################################################"
    echo "#  PART 3 : Graph-JEPA ablation  (build_rwse -> paper_reason_gjepa)"
    echo "############################################################"
    echo "  Reuses hetero cache: ${HETERO_CACHE}"
    echo "  RWSE cache         : ${RWSE_CACHE}"

    # PART 3 depends on the hetero cache. If PART 2 was skipped, the cache must
    # already exist on disk (from a previous run).
    if [[ ! -f "${HETERO_CACHE}" ]]; then
      echo "  ERROR: ${HETERO_CACHE} not found."
      echo "         PART 3 needs the hetero graph. Run PART 2 first"
      echo "         (RUN_PART2=1, REBUILD_HETERO=1) or point to an existing cache."
      exit 1
    fi

    # ── Stage 3a: build (or reuse) the RWSE positional-encoding cache ──
    echo "------------------------------------------------------------"
    echo "  Stage 3a: DYNAMIC schema analysis + TRUE RWSE  (train.build_rwse)"
    echo "  -> READ the SCHEMA + CONNECTIVITY REPORT it prints below."
    echo "------------------------------------------------------------"
    if [[ "${REBUILD_RWSE}" == "1" ]]; then
      echo "  REBUILD_RWSE=1  ->  removing ${RWSE_CACHE} (will rebuild)"
      rm -f "${RWSE_CACHE}" 2>/dev/null || true
      REBUILD_HETERO=0 python -m train.build_rwse
    else
      echo "  REBUILD_RWSE=0  ->  REUSING existing ${RWSE_CACHE}"
      if [[ ! -f "${RWSE_CACHE}" ]]; then
        echo "  RWSE cache not found; building it now (one-off)."
        REBUILD_HETERO=0 python -m train.build_rwse
      fi
    fi

    # guard: Stage 3b cannot run without the RWSE cache
    if [[ ! -f "${RWSE_CACHE}" ]]; then
      echo "  ERROR: ${RWSE_CACHE} was not produced by Stage 3a."
      echo "         Check the build_rwse output above for errors."
      exit 1
    fi
    echo "  Stage 3a done at $(date)"
    echo

    # ── Stage 3b: faithful Graph-JEPA (GINE + within-paper masked prediction) ──
    echo "------------------------------------------------------------"
    echo "  Stage 3b: Running A1 (reasoning probe) + A2 (patch retrieval)"
    echo "  NOTE: paper_reason_gjepa.py loads BOTH caches with rebuild=False,"
    echo "        so it will NOT re-embed nodes and reuses the RWSE PE."
    echo "------------------------------------------------------------"
    REBUILD_HETERO=0 python -m train.paper_reason_gjepa
    echo "  Stage 3b done at $(date)"
    echo "  PART 3 done at $(date)"
    echo
  else
    echo "  [skip] PART 3 (RUN_PART3=0)"
    echo
  fi

  echo "════════════════════════════════════════════════════════════"
  echo "  ALL DONE at $(date)"
  echo "════════════════════════════════════════════════════════════"

} 2>&1 | tee "${RUN_LOG}"

echo "Full run log saved to: ${RUN_LOG}"