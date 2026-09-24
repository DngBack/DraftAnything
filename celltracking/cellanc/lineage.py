"""Ancestry / sibling targets from an audited track table (doc §8–9, §11)."""

from collections import defaultdict

Tracks = dict[int, tuple[int, int, int]]  # id -> (start, end, parent)


def trace_to_anchor(track_id: int, a: int, tracks: Tracks, present_at_a: set[int]):
    """Walk the parent chain of `track_id` back to the track alive at frame `a`.

    Returns (anchor_track_id | None, status, generation_depth). A root that starts
    after `a` is NOT concluded to be a new entry (doc §9): status says unresolved.
    """
    seen = set()
    depth = 0
    u = track_id
    while True:
        if u in seen or u not in tracks:
            return None, "invalid_lineage", depth
        seen.add(u)
        start, end, parent = tracks[u]
        if start <= a <= end:
            if u in present_at_a:
                return u, "valid_anchor", depth
            return None, "missing_anchor_annotation", depth
        if end < a:
            return None, "broken_temporal_path", depth
        if parent == 0:
            return None, "unresolved_root_after_anchor", depth
        u = parent
        depth += 1


def hidden_intermediates(track_id: int, anchor: int, tracks: Tracks, kept: set[int]) -> int:
    """Intermediate tracks strictly between target and anchor with no kept frame (doc §11)."""
    n = 0
    u = tracks[track_id][2]
    while u != anchor and u != 0:
        s, e, p = tracks[u]
        if not any(s <= t <= e for t in kept):
            n += 1
        u = p
    return n


def ancestry_targets(a: int, b: int, tracks: Tracks, present: dict[int, set[int]],
                     kept_frames: list[int] | None = None) -> list[dict]:
    """One row per GT track present at frame b (GT IDs: evaluator-side only)."""
    kept = set(kept_frames or [a, b])
    rows = []
    for u in sorted(present.get(b, ())):
        anc, status, depth = trace_to_anchor(u, a, tracks, present.get(a, set()))
        rows.append({
            "target_gt": u,
            "anchor_gt": anc,
            "status": status,
            "generation_depth": depth,
            "hidden_intermediates": hidden_intermediates(u, anc, tracks, kept) if anc else None,
        })
    return rows


def sibling_pairs(b: int, tracks: Tracks, present_b: set[int]) -> list[tuple[int, int]]:
    """Direct sisters both present at b: same non-zero parent (P=0 is never a sibling cue)."""
    by_parent = defaultdict(list)
    for u in present_b:
        p = tracks[u][2]
        if p != 0:
            by_parent[p].append(u)
    return [(x, y) for kids in by_parent.values() for i, x in enumerate(sorted(kids))
            for y in sorted(kids)[i + 1:]]


def schedule(a: int, b: int, stride: int | None) -> list[int]:
    """Matched-horizon schedule (doc §10 exp. A); None = endpoints only. Always keeps b."""
    if stride is None:
        return [a, b]
    frames = list(range(a, b + 1, stride))
    return frames if frames[-1] == b else frames + [b]
