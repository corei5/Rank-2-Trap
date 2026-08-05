import os
import gc
import glob
import json
import torch
import numpy as np
from collections import Counter
from torch_geometric.data import InMemoryDataset, Data

# ---------------------------------------------------------------------------
# Field names differ across the three record formats found in the raw data.
# We list BOTH spellings; whichever exists is used. Each field also maps to a
# canonical NODE TYPE, so reasoning tasks can say "predict the METHOD node".
# (This dict REPLACES the old flat SUMMARY_TEXT_FIELDS list — same fields,
#  now with a type attached to each.)
# ---------------------------------------------------------------------------
SUMMARY_FIELD_TO_TYPE = {
    "executive_summary": "summary",
    "research_context": "context",
    "research_question_and_hypothesis": "question",   # format A/B
    "research_question_hypothesis": "question",        # format C (flat)
    "methodological_details": "method",
    "procedures_and_architectures": "method",          # format A/B
    "procedures_architectures": "method",              # format C
    "key_results": "result",
    "interpretation_and_theoretical_implications": "interpretation",  # A/B
    "interpretation_implications": "interpretation",                  # C
    "contradictions_and_limitations": "limitation",   # A/B
    "contradictions_limitations": "limitation",        # C
    "key_figures_tables": "figures",
    "three_takeaways": "takeaways",
}
SUMMARY_TEXT_FIELDS = list(SUMMARY_FIELD_TO_TYPE.keys())   # preserves order

CLAIM_FIELD_TO_TYPE = {
    "details": "claim",          # format A/B
    "description": "claim",      # format C
    "supporting_evidence": "claim_support",
    "contradicting_evidence": "claim_contradict",
    "implications": "claim_implication",
}
CLAIM_TEXT_FIELDS = list(CLAIM_FIELD_TO_TYPE.keys())

# canonical node-type vocabulary -> integer id (stored on each graph as node_type)
NODE_TYPE_VOCAB = sorted(set(SUMMARY_FIELD_TO_TYPE.values()) |
                         set(CLAIM_FIELD_TO_TYPE.values()))
NODE_TYPE_ID = {t: i for i, t in enumerate(NODE_TYPE_VOCAB)}


def _coerce_summary(paper):
    """
    Return the dict that actually holds the summary fields, plus the label,
    handling all three record formats. Returns (summary_dict, label) or
    (None, None) if the record should be skipped.
    """
    # ---- Format C: flat record (field_subfield already at top level) ----
    if "field_subfield" in paper and "summarization" not in paper:
        label = paper.get("field_subfield")
        if not label:
            return None, None
        return paper, label

    # ---- Format A/B: nested under 'summarization' (a JSON *string*) ----
    raw = paper.get("summarization")
    if raw is None:
        return None, None
    if isinstance(raw, str):
        try:
            outer = json.loads(raw)
        except json.JSONDecodeError:
            return None, None
    elif isinstance(raw, dict):
        outer = raw
    else:
        return None, None

    # Non-scientific records have summary == null -> skip
    summary = outer.get("summary")
    if not summary or not isinstance(summary, dict):
        return None, None
    label = summary.get("field_subfield")
    if not label:
        return None, None
    return summary, label


def _extract_node_texts(summary):
    """
    Collect (text, node_type) pairs from summary fields + claim sub-fields.
    NOTE: old version returned only a list of texts; now each text carries its
    canonical node type so the reasoning evaluator knows what each node is.
    """
    nodes = []
    for k in SUMMARY_TEXT_FIELDS:
        v = summary.get(k)
        if isinstance(v, str) and v.strip():
            nodes.append((v.strip(), SUMMARY_FIELD_TO_TYPE[k]))
    for claim in summary.get("claims", []) or []:
        if not isinstance(claim, dict):
            continue
        for ck in CLAIM_TEXT_FIELDS:
            v = claim.get(ck)
            if isinstance(v, str) and v.strip():
                nodes.append((v.strip(), CLAIM_FIELD_TO_TYPE[ck]))
    return nodes


def parse_paper(paper, coarse_label=False):
    """
    Parse one raw paper record into (node_texts, node_types, label_string).
    Returns None if the record cannot be used (skipped).
    (Old signature returned (node_texts, label); we added node_types.)
    """
    summary, label = _coerce_summary(paper)
    if summary is None:
        return None
    nodes = _extract_node_texts(summary)
    if len(nodes) < 2:               # need >=2 nodes to form a graph
        return None
    node_texts = [t for t, _ in nodes]
    node_types = [ty for _, ty in nodes]
    if coarse_label:
        # "Economics — Development Economics..." -> "Economics"
        for sep in (" — ", " - ", "—", " – "):
            if sep in label:
                label = label.split(sep)[0]
                break
        label = label.strip()
    return node_texts, node_types, label


class PapersDataset(InMemoryDataset):
    """
    One paper -> one graph. Nodes = embedded text sections (summary sections +
    claim parts), each TYPED. Edges = fully connected among the paper's nodes.
    y = integer id of field_subfield (coarse or full).

    Each Data now also carries node_type [N] (int id into NODE_TYPE_VOCAB),
    which the reasoning evaluator uses to mask/predict a chosen section type.

    Rare classes (fewer than min_class_size papers) are dropped so stratified
    splitting and the linear probe behave sensibly.
    """

    def __init__(self, root, transform=None, pre_transform=None,
                 coarse_label=True, embed_model="all-MiniLM-L6-v2",
                 min_class_size=50):
        self.coarse_label = coarse_label
        self.embed_model_name = embed_model
        self.min_class_size = int(min_class_size)   # drop classes below this count
        self.label_map = {}                          # label_string -> int id
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
        # restore label_map saved alongside the processed data
        meta_path = os.path.join(self.processed_dir, "label_map.json")
        if os.path.exists(meta_path):
            self.label_map = json.load(open(meta_path, encoding="utf-8"))

    @property
    def raw_file_names(self):
        files = [os.path.basename(p)
                 for p in glob.glob(os.path.join(self.raw_dir, "*.json"))]
        return files if files else ["papers.json"]

    @property
    def processed_file_names(self):
        suffix = "coarse" if self.coarse_label else "full"
        # '_typed' tag forces a rebuild now that node_type is stored
        return [f"papers_{suffix}_min{self.min_class_size}_typed.pt"]

    def download(self):
        raise RuntimeError(
            f"No raw .json files found in {self.raw_dir}. "
            f"Place your paper JSON file(s) there.")

    @property
    def num_classes(self):
        return len(self.label_map)

    def _iter_raw_records(self):
        """Yield every paper record from every json file (object OR array)."""
        for fp in glob.glob(os.path.join(self.raw_dir, "*.json")):
            try:
                data = json.load(open(fp, encoding="utf-8"))
            except json.JSONDecodeError:
                print(f"[PapersDataset] WARNING: could not parse {fp}, skipping.")
                continue
            except Exception as ex:
                print(f"[PapersDataset] WARNING: error reading {fp}: {ex}, skipping.")
                continue
            if isinstance(data, dict):
                yield data
            elif isinstance(data, list):
                for rec in data:
                    if isinstance(rec, dict):
                        yield rec

    def process(self):
        # lazy imports so these deps are only needed when building the cache
        from sentence_transformers import SentenceTransformer
        try:
            from tqdm import tqdm
        except ImportError:
            def tqdm(x, **k):
                return x

        # -------------------------------------------------------------------
        # PASS 1 — parse every raw record into (node_texts, node_types, label).
        # Cheap (strings only), so we hold it in a list.
        # -------------------------------------------------------------------
        parsed = []
        skipped = 0
        for rec in tqdm(self._iter_raw_records(), desc="Parsing papers"):
            out = parse_paper(rec, coarse_label=self.coarse_label)
            if out is None:
                skipped += 1
                continue
            parsed.append(out)
        if not parsed:
            raise RuntimeError(
                "No usable papers found. Check that your raw JSON has "
                "'field_subfield' (flat) or summarization->summary->field_subfield.")
        print(f"[PapersDataset] parsed={len(parsed)} skipped={skipped}", flush=True)

        # -------------------------------------------------------------------
        # CLASS-IMBALANCE FILTER — drop classes with < min_class_size papers.
        # -------------------------------------------------------------------
        label_counts = Counter(lbl for _, _, lbl in parsed)
        keep_classes = {c for c, n in label_counts.items()
                        if n >= self.min_class_size}
        n_before, c_before = len(parsed), len(label_counts)
        if not keep_classes:
            raise RuntimeError(
                f"No class has >= {self.min_class_size} papers "
                f"(max class count = {max(label_counts.values())}).")
        parsed = [(nt, ty, l) for (nt, ty, l) in parsed if l in keep_classes]
        print(f"[PapersDataset] class filter (min_class_size={self.min_class_size}): "
              f"kept {len(keep_classes)}/{c_before} classes, "
              f"dropped {c_before - len(keep_classes)} rare classes | "
              f"kept {len(parsed)}/{n_before} papers "
              f"(dropped {n_before - len(parsed)})", flush=True)

        # -------------------------------------------------------------------
        # Contiguous label ids 0..K-1 from surviving labels only.
        # -------------------------------------------------------------------
        labels_sorted = sorted(keep_classes)
        self.label_map = {lbl: i for i, lbl in enumerate(labels_sorted)}

        # -------------------------------------------------------------------
        # PASS 2 — embed + build TYPED graphs in CHUNKS (bounded memory).
        # -------------------------------------------------------------------
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = SentenceTransformer(self.embed_model_name, device=device)

        data_list = []
        PAPER_CHUNK = 2000               # tune down if RAM is tight
        for start in tqdm(range(0, len(parsed), PAPER_CHUNK),
                          desc="Embedding + building graphs"):
            chunk = parsed[start:start + PAPER_CHUNK]
            texts, offsets = [], []
            for node_texts, _, _ in chunk:
                offsets.append((len(texts), len(texts) + len(node_texts)))
                texts.extend(node_texts)
            emb = model.encode(texts, batch_size=256, convert_to_numpy=True,
                               show_progress_bar=False)

            for (node_texts, node_types, label), (s, e) in zip(chunk, offsets):
                x = torch.tensor(emb[s:e], dtype=torch.float)   # [num_nodes, emb_dim]
                n = x.size(0)
                # fully-connected edges (no self-loops), both directions
                if n > 1:
                    idx = torch.arange(n)
                    src = idx.repeat_interleave(n)
                    dst = idx.repeat(n)
                    mask = src != dst
                    edge_index = torch.stack([src[mask], dst[mask]], dim=0)
                else:
                    edge_index = torch.empty((2, 0), dtype=torch.long)

                edge_attr = torch.zeros((edge_index.size(1), 1), dtype=torch.long)
                y = torch.tensor([self.label_map[label]], dtype=torch.long)
                nt_ids = torch.tensor([NODE_TYPE_ID[t] for t in node_types],
                                      dtype=torch.long)          # NEW: per-node type

                data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
                data.node_type = nt_ids                          # NEW
                if self.pre_transform is not None:
                    data = self.pre_transform(data)
                data_list.append(data)

            del emb, texts, chunk
            gc.collect()

        emb_dim = data_list[0].x.size(1) if len(data_list) else 0
        print(f"[PapersDataset] built {len(data_list)} graphs | emb_dim={emb_dim} "
              f"| num_classes={len(self.label_map)} | skipped={skipped} "
              f"| coarse={self.coarse_label} | node_types={NODE_TYPE_VOCAB}",
              flush=True)

        # -------------------------------------------------------------------
        # Collate + save (free data_list first to avoid a 2x memory spike).
        # -------------------------------------------------------------------
        data, slices = self.collate(data_list)
        del data_list
        gc.collect()
        torch.save((data, slices), self.processed_paths[0])

        # persist label_map + node-type vocab for inference/evaluation
        json.dump(self.label_map,
                  open(os.path.join(self.processed_dir, "label_map.json"),
                       "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        json.dump(NODE_TYPE_VOCAB,
                  open(os.path.join(self.processed_dir, "node_type_vocab.json"),
                       "w", encoding="utf-8"), ensure_ascii=False, indent=2)