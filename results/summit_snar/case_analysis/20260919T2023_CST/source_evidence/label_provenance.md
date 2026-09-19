# Scientific source excerpt: audit.py

Logical source: `benchmark_extensions/summit_snar_representation_repair_20260919_v2/audit.py`.

SHA256 of the **complete original source**: `8e389992eb0ba00b821b2e8e9b8c18e148e7a995f53a679cee9433dc8ee0e131`. This is a quoted excerpt for review, not a portable executable or the complete source file. Line numbers below refer to the original file; the export manifest separately hashes this excerpt.

## Lines 168–180

```text
168:                     raise ValueError("Original token hash differs")
169:                 if query["parameters"] != g["parsed_action"] or query["no_hvi"] not in (0, 1):
170:                     raise ValueError("Original chosen action/label differs")
171:                 if split == "train": train_hashes.add(g["prefix_hash"])
172:                 elif g["prefix_hash"] in train_hashes:
173:                     raise ValueError("Exact train/dev prefix overlap")
174:                 graph_path = generation_path.parent/"risk_graph.json"
175:                 graph = read(graph_path)
176:                 selected.append({"index": len(selected), "split": split, "episode": name,
177:                     "query_id": query["query_id"], "query": artifact(path), "generation": artifact(generation_path),
178:                     "old_graph": artifact(graph_path), "old_graph_available": graph["available"],
179:                     "label": query["no_hvi"], "original_generation_prefix_hash": g["prefix_hash"],
180:                     "prefix_tokens": ids.shape[1]})
```
