import json
import sys
from pathlib import Path
from evidencerag.backend.main import retriever


def main():
    path = Path(sys.argv[1] if len(sys.argv) > 1 else "evaluation.json")
    if not path.exists():
        print(f"Create {path} with question/relevant_chunk_ids records before evaluating.")
        return
    dataset = json.loads(path.read_text())
    recalls = {5: [], 10: []}; reciprocal = []
    for item in dataset:
        _, _, fused, _, _ = retriever.search(item["question"], 15, 15, 60, 5)
        ids = [r.chunk.chunk_id for r in fused]
        relevant = set(item["relevant_chunk_ids"])
        for k in recalls: recalls[k].append(bool(relevant.intersection(ids[:k])))
        reciprocal.append(1 / (next((i + 1 for i, cid in enumerate(ids) if cid in relevant), len(ids) + 1)))
    print("EvidenceRAG Evaluation")
    print(f"Recall@5:  {sum(recalls[5]) / max(1, len(recalls[5])):.2f}")
    print(f"Recall@10: {sum(recalls[10]) / max(1, len(recalls[10])):.2f}")
    print(f"MRR:       {sum(reciprocal) / max(1, len(reciprocal)):.2f}")


if __name__ == "__main__": main()
