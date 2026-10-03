import json

curated = json.load(open('results/phase3/bench_dense_retrievals.json'))
stress = json.load(open('results/stress_test/stress_test_retrievals.json'))['dense']

cur_map = {}
for item in curated:
    qid = str(item['query_id'])
    gold_ids = set(str(g) for g in item.get('gold_passage_ids', []))
    pids = [r['passage_id'] for r in item.get('results', [])]
    rank = None
    for r_idx, pid in enumerate(pids, start=1):
        if pid in gold_ids:
            rank = r_idx
            break
    cur_map[qid] = {'rank': rank, 'pids': pids, 'gold': gold_ids, 'query': item.get('query')}

improved = []
degraded = []
identical = []

for qid, cur_info in cur_map.items():
    st_pids = stress.get(qid, [])
    st_rank = None
    for r_idx, pid in enumerate(st_pids, start=1):
        if pid in cur_info['gold']:
            st_rank = r_idx
            break
    c_rank = cur_info['rank']
    if c_rank != st_rank:
        if (st_rank is not None and c_rank is None) or (st_rank is not None and c_rank is not None and st_rank < c_rank):
            improved.append((qid, cur_info['query'], c_rank, st_rank, cur_info['pids'][:5], st_pids[:5]))
        else:
            degraded.append((qid, cur_info['query'], c_rank, st_rank, cur_info['pids'][:5], st_pids[:5]))
    else:
        identical.append(qid)

print(f"Total queries: {len(cur_map)}")
print(f"Identical ranks: {len(identical)}")
print(f"Degraded ranks on c100k_hard: {len(degraded)}")
print(f"Improved ranks on c100k_hard: {len(improved)}")

print("\n--- IMPROVED QUERIES ON c100k_hard ---")
for qid, q, cr, sr, cp, sp in improved:
    print(f"qid={qid}: curated_rank={cr} -> stress_rank={sr} | query='{q}'")
    print(f"  curated top-5: {cp}")
    print(f"  stress top-5:  {sp}")
    print(f"  gold: {cur_map[qid]['gold']}")

print("\n--- DEGRADED QUERIES ON c100k_hard ---")
for qid, q, cr, sr, cp, sp in degraded:
    print(f"qid={qid}: curated_rank={cr} -> stress_rank={sr} | query='{q}'")
    print(f"  curated top-5: {cp}")
    print(f"  stress top-5:  {sp}")
    print(f"  gold: {cur_map[qid]['gold']}")
