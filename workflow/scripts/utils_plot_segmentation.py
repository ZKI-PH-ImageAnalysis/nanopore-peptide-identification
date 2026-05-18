#!/usr/bin/env python

import logging
import os
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import pod5 as p5
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
from remora import io, refine_signal_map, util


BASE_COLORS = {
    'A': 'lightgreen',
    'C': 'lightblue',
    'G': '#fff176',
    'T': 'lightcoral',
    'N': 'grey',
}
MIN_SIGNAL = 0
MAX_SIGNAL = 200


@dataclass
class DoradoAdapterQuery:
    name: str
    front_seq: str
    rear_seq: str


DORADO_ADAPTERS = [
    DoradoAdapterQuery(
        name="TAdapter",
        front_seq="CCTGTACTTCGTTCAGTTACGTATTGC",
        rear_seq="",
    ),
    DoradoAdapterQuery(
        name="BAdapter",
        front_seq="",
        rear_seq="AGCAATACGT",
    ),
]


logging.getLogger("Remora").setLevel(logging.INFO)
logger = logging.getLogger("nanopore-peptide-classifier")


_global_sig_map_refiner = refine_signal_map.SigMapRefiner(
    kmer_model_filename="9mer_levels_v1.txt",
    do_rough_rescale=False,
    scale_iters=0,
    do_fix_guage=True,
)


def get_signal_from_read(read):
    offset = read.calibration.offset
    scale = read.calibration.scale
    signal = read.signal
    pA_signal = scale * (signal + offset)
    pA_signal = np.array(pA_signal)
    return pA_signal


def _movetable_build_q_to_r(aln, pairs_all, with_seq=True):
    q_to_r = {}
    try:
        pairs = aln.get_aligned_pairs(matches_only=False, with_seq=with_seq)
        for p in pairs:
            if with_seq:
                q, r, _ = p
            else:
                q, r = p
            if q is None or r is None:
                continue
            q = int(q)
            if q not in q_to_r:
                q_to_r[q] = int(r)
    except Exception:
        for q, r in pairs_all:
            q_to_r[int(q)] = int(r)
    return q_to_r


def _movetable_refine_and_extract_levels(io_read):
    try:
        io_read.set_refine_signal_mapping(_global_sig_map_refiner, ref_mapping=False)
        try:
            io_read.set_refine_signal_mapping(_global_sig_map_refiner, ref_mapping=True)
        except Exception:
            pass
    except Exception as e:
        logger.error(f"Error refining mapping for read {io_read.read_id}: {e}")
        return False, False, None

    ref_rebuilt = False
    try:
        io_read.compute_ref_to_signal()
        ref_rebuilt = (
            io_read.ref_to_signal is not None
            and io_read.ref_to_signal.size == (len(io_read.ref_seq) + 1)
        )
    except Exception:
        ref_rebuilt = False

    try:
        move_levels = _global_sig_map_refiner.extract_levels(util.seq_to_int(io_read.seq))
    except Exception:
        move_levels = None

    return True, ref_rebuilt, move_levels


def _movetable_window_bounds(io_read, sig_start, sig_end):
    sig_len = io_read._sig_len
    st = 0 if sig_start is None else sig_start
    en = sig_len if sig_end is None else min(sig_end, sig_len)
    return sig_len, st, en


def _movetable_build_basecall_segments(io_read, st, en):
    q_knots = np.array(io_read.query_to_signal, dtype=float)
    bc_start_idx = int(np.searchsorted(q_knots, st))
    bc_end_idx = int(np.searchsorted(q_knots, en))
    bc_start_idx = max(0, min(bc_start_idx, len(io_read.seq)))
    bc_end_idx = max(0, min(bc_end_idx, len(io_read.seq)))

    if bc_end_idx > bc_start_idx:
        bc_s, bc_e = _knots_to_segments(q_knots, bc_start_idx, bc_end_idx, st, en)
        bc_seq = io_read.seq[bc_start_idx:bc_end_idx]
        n = min(len(bc_seq), len(bc_s), len(bc_e))
        bc_sig_start = bc_s[:n]
        bc_sig_end = bc_e[:n]
        bc_seq = bc_seq[:n]
        bc_q_indices = list(range(bc_start_idx, bc_start_idx + n))
    else:
        bc_sig_start = np.array([], dtype=int)
        bc_sig_end = np.array([], dtype=int)
        bc_seq = ""
        bc_q_indices = []

    return bc_start_idx, bc_end_idx, bc_sig_start, bc_sig_end, bc_seq, bc_q_indices


def _movetable_build_reference_segments(
    io_read,
    aln,
    ref_rebuilt,
    st,
    en,
    ref_seq_str,
    ref_global_start,
    ref_len_local,
    bc_sig_start,
    bc_sig_end,
    bc_q_indices,
    pairs_all,
):
    ref_sig_start = []
    ref_sig_end = []
    ref_bases = []
    ref_positions_global = []

    if ref_rebuilt:
        knots_ref = np.array(io_read.ref_to_signal, dtype=float)
        r0 = int(np.searchsorted(knots_ref[:-1], st - 1))
        r1 = int(np.searchsorted(knots_ref[:-1], en))
        r0 = max(0, min(r0, len(ref_seq_str)))
        r1 = max(0, min(r1, len(ref_seq_str)))
        rs, re = _knots_to_segments(knots_ref, r0, r1, st, en)
        ref_sig_start = rs.tolist()
        ref_sig_end = re.tolist()
        ref_bases = list(ref_seq_str[r0:r1])
        ref_positions_global = list(np.arange(ref_global_start + r0, ref_global_start + r1))
    else:
        q_to_r = _movetable_build_q_to_r(aln, pairs_all, with_seq=True)
        for j, q_idx in enumerate(bc_q_indices):
            r_global = q_to_r.get(q_idx)
            if r_global is None:
                continue
            rl = int(r_global) - ref_global_start
            if rl < 0 or rl >= ref_len_local:
                continue
            ref_sig_start.append(int(bc_sig_start[j]))
            ref_sig_end.append(int(bc_sig_end[j]))
            ref_bases.append(ref_seq_str[rl])
            ref_positions_global.append(int(r_global))

        if len(ref_sig_start) < max(1, int(len(bc_q_indices) * 0.25)):
            if getattr(io_read, "ref_to_signal", None) is not None and io_read.ref_to_signal.size == (len(ref_seq_str) + 1):
                knots_ref = np.array(io_read.ref_to_signal, dtype=float)
                r0 = int(np.searchsorted(knots_ref[:-1], st - 1))
                r1 = int(np.searchsorted(knots_ref[:-1], en))
                r0 = max(0, min(r0, len(ref_seq_str)))
                r1 = max(0, min(r1, len(ref_seq_str)))
                rs, re = _knots_to_segments(knots_ref, r0, r1, st, en)
                for i_local, (rs_i, re_i) in enumerate(zip(rs.tolist(), re.tolist())):
                    gpos = ref_global_start + r0 + i_local
                    if gpos in ref_positions_global:
                        continue
                    assigned = False
                    for bcs, bce in zip(bc_sig_start.tolist(), bc_sig_end.tolist()):
                        center = (rs_i + re_i) / 2.0
                        if center >= bcs and center <= bce:
                            ref_sig_start.append(int(bcs))
                            ref_sig_end.append(int(bce))
                            ref_bases.append(ref_seq_str[r0 + i_local])
                            ref_positions_global.append(int(gpos))
                            assigned = True
                            break
                    if not assigned:
                        ref_sig_start.append(int(rs_i))
                        ref_sig_end.append(int(re_i))
                        ref_bases.append(ref_seq_str[r0 + i_local])
                        ref_positions_global.append(int(gpos))

    if not ref_sig_start:
        return np.array([], dtype=int), np.array([], dtype=int), [], []

    return (
        np.array(ref_sig_start, dtype=int),
        np.array(ref_sig_end, dtype=int),
        ref_bases,
        ref_positions_global,
    )


def _movetable_plot_move_levels(ax_sig, move_levels, bc_sig_start, bc_sig_end, bc_start_idx, signal_type, io_read):
    if move_levels is None or len(bc_sig_start) == 0:
        return

    mv = np.array(move_levels)
    n_bc = len(bc_sig_start)
    mv_start = max(0, bc_start_idx)
    mv_slice = mv[mv_start: mv_start + n_bc]
    if signal_type == "pa":
        mv_slice = mv_slice * io_read.scale_pa_to_norm + io_read.shift_pa_to_norm
    m = min(len(mv_slice), n_bc)
    if m == 0:
        return

    xs = []
    ys = []
    for i in range(m):
        qs_rel = int(bc_sig_start[i])
        qe_rel = int(bc_sig_end[i])
        lvl = float(mv_slice[i])
        if qe_rel <= qs_rel:
            continue

        if not xs:
            xs.extend([qs_rel, qe_rel])
            ys.extend([lvl, lvl])
            continue

        prev_x = xs[-1]
        prev_y = ys[-1]
        if qs_rel != prev_x:
            xs.append(qs_rel)
            ys.append(prev_y)
        if lvl != prev_y:
            xs.append(qs_rel)
            ys.append(lvl)
        xs.append(qe_rel)
        ys.append(lvl)

    if xs:
        ax_sig.plot(
            xs,
            ys,
            color="#b8a8f2",
            linewidth=1.5,
            alpha=0.85,
            zorder=0,
            solid_capstyle="butt",
        )


def _movetable_annotate_basecalls(ax_sig, bc_sig_start, bc_sig_end, bc_seq, ref_sig_start, ref_sig_end, en, st, PLR_start, PLR_end, min_signal, span):
    sig_window_len = en - st
    too_long = sig_window_len > 10000
    no_plr = (PLR_start is None or PLR_end is None)
    if not no_plr and not too_long:
        bc_iter = zip(bc_sig_start, bc_sig_end, bc_seq)
    else:
        window = 50
        ref_min = int(ref_sig_start[0] - window) if len(ref_sig_start) > 0 else 0
        ref_max = int(ref_sig_end[-1] + window) if len(ref_sig_end) > 0 else en - st
        bc_iter = ((xs, xe, b) for xs, xe, b in zip(bc_sig_start, bc_sig_end, bc_seq) if not (xe < ref_min or xs > ref_max))

    for xs, xe, base in bc_iter:
        xc = (xs + xe) / 2
        yc = min_signal - span * 0.02
        ax_sig.text(xc, yc, base, va='top', ha='center', fontsize=6,
                    color=BASE_COLORS.get(base, 'grey'), fontweight='bold')

    ax_sig.text(0, min_signal - span * 0.065, 'Basecalls', va='top', ha='left', fontsize=10, fontweight='bold')


def _movetable_plot_qscore_panel(ax_q, bc_sig_start, bc_sig_end, bc_seq, quals, q_max, st, en):
    midpoints = (bc_sig_start + bc_sig_end) / 2 if len(bc_sig_start) > 0 else np.array([])
    widths = (bc_sig_end - bc_sig_start) if len(bc_sig_start) > 0 else np.array([])
    patches = [Rectangle((xs, 0), max(1, xe - xs), max(1, q_max * 1.05)) for xs, xe in zip(bc_sig_start, bc_sig_end)]
    colors = [BASE_COLORS.get(b, 'gray') for b in bc_seq]
    if patches:
        pc = PatchCollection(patches, facecolor=colors, alpha=0.2, edgecolor='black', linewidth=0.5)
        ax_q.add_collection(pc)
    if len(midpoints) > 0 and len(quals) > 0:
        ax_q.bar(midpoints, quals, width=widths, align='center', color='lightgray', edgecolor='black', linewidth=0.5, alpha=0.5)

    ax_q.set_ylabel('Q-score')
    ax_q.set_xlabel('Signal index')
    ax_q.set_xticks(np.linspace(0, en - st, num=11, dtype=int))
    ax_q.set_xticklabels([str(int(x)) for x in np.linspace(st, en, num=11, dtype=int)])
    ax_q.set_xlim(0, en - st)
    ax_q.set_ylim(0, max(1, q_max * 1.1))


def plot_single_read_movetable(read_record, sam_input, outdir,
                              PLR_start, PLR_end,
                              signal_type='norm', sig_start=None, sig_end=None):

    if sam_input is None:
        signal = get_signal_from_read(read_record)
        plot_single_read(signal, str(read_record.read_id), outdir,
                         PLR_start=PLR_start, PLR_end=PLR_end)
        return True

    max_signal, min_signal = (4, -4) if signal_type == "norm" else (180, 0)
    aln = sam_input.get_first_alignment(str(read_record.read_id))
    alns = list(sam_input.get_alignments(str(read_record.read_id)))
    if len(alns) > 1:
        logger.warning(f"Multiple alignments ({len(alns)}) for read {read_record.read_id}; using first.")
    io_read = io.Read.from_pod5_and_alignment(read_record, aln)

    if io_read.query_to_signal is None:
        logger.error(f"Read {io_read.read_id} missing query_to_signal (move table).")
        return False

    try:
        pairs_raw = aln.get_aligned_pairs(matches_only=False, with_seq=False)
    except Exception:
        pairs_raw = []
    pairs_all = [(int(q), int(r)) for q, r in pairs_raw if q is not None and r is not None]
    if not pairs_all:
        logger.error(f"No matched pairs for read {io_read.read_id}")
        return False

    ok_refine, ref_rebuilt, move_levels = _movetable_refine_and_extract_levels(io_read)
    if not ok_refine:
        return False

    sig_len, st, en = _movetable_window_bounds(io_read, sig_start, sig_end)
    if (PLR_start is None or PLR_end is None) and sig_len > 50_000:
        plot_single_read(get_signal_from_read(read_record), str(read_record.read_id), outdir,
                         PLR_start=PLR_start, PLR_end=PLR_end)
        return True

    ref_global_start = aln.reference_start
    ref_seq_str = io_read.ref_seq or ""
    ref_len_local = len(ref_seq_str)

    bc_start_idx, bc_end_idx, bc_sig_start, bc_sig_end, bc_seq, bc_q_indices = _movetable_build_basecall_segments(io_read, st, en)
    quals_all = aln.query_qualities or []
    quals = np.array(quals_all[bc_start_idx:bc_end_idx]) if len(quals_all) > 0 else np.array([])
    q_max = int(quals.max()) if len(quals) > 0 else 0

    ref_sig_start, ref_sig_end, ref_bases, ref_positions_global = _movetable_build_reference_segments(
        io_read,
        aln,
        ref_rebuilt,
        st,
        en,
        ref_seq_str,
        ref_global_start,
        ref_len_local,
        bc_sig_start,
        bc_sig_end,
        bc_q_indices,
        pairs_all,
    )

    fig, (ax_sig, ax_q) = plt.subplots(2, 1, figsize=(12, 6),
                                       gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
    strand_arrow = "←" if aln.is_reverse else "→"
    strand_label = "(reverse alignment)" if aln.is_reverse else "(forward alignment)"
    title = f"{strand_arrow} {strand_label} | Read {io_read.read_id}  {aln.reference_name}:{aln.reference_start + 1}-{aln.reference_end}"
    ax_sig.set_title(title, pad=20)

    sig = io_read.get_sig_type(signal_type)[st:en]
    ax_sig.plot(np.arange(0, en - st), sig, lw=0.8, color='black')
    ax_sig.set_ylabel('Signal (' + ("pA" if signal_type == "pa" else "Normalized") + ')', fontsize=10)
    ax_sig.set_xticks([])
    fig.subplots_adjust(top=0.85)
    mid = 0.1 * (max_signal - min_signal) if max_signal is not None else 0
    ax_sig.set_ylim(min_signal - mid, max_signal)
    span = max_signal - min_signal if max_signal is not None else 1

    _movetable_plot_move_levels(ax_sig, move_levels, bc_sig_start, bc_sig_end, bc_start_idx, signal_type, io_read)

    if PLR_start is not None and PLR_end is not None:
        mask = (np.arange(st, en) >= PLR_start) & (np.arange(st, en) <= PLR_end)
        ax_sig.plot(np.arange(0, en - st)[mask], sig[mask], lw=1.0, color='orange', zorder=2)

    ref_bars = [(int(xs), int(max(1, xe - xs))) for xs, xe in zip(ref_sig_start, ref_sig_end)]
    ax_sig.broken_barh(ref_bars, (min_signal, span), facecolors=[BASE_COLORS.get(b, 'grey') for b in ref_bases],
                       alpha=0.35, edgecolor='black', linewidth=0.4)
    for xs, xe, base, g in zip(ref_sig_start, ref_sig_end, ref_bases, ref_positions_global):
        xc = xs + (xe - xs) / 2
        ax_sig.text(xc, max_signal - span * 0.05, base, va='bottom', ha='center',
                    fontsize=6, color=BASE_COLORS.get(base, 'grey'), fontweight='bold')
        ax_sig.text(xc, max_signal + span * 0.003, str(int(g + 1)), va='bottom', ha='center', fontsize=4)

    bq_bars = [(int(xs), int(max(1, xe - xs))) for xs, xe in zip(bc_sig_start, bc_sig_end)]
    ax_sig.broken_barh(bq_bars, (min_signal, min_signal - 10),
                       facecolors=[BASE_COLORS.get(b, 'gray') for b in bc_seq], alpha=0.3,
                       edgecolor='black', linewidth=0.5)

    _movetable_annotate_basecalls(
        ax_sig,
        bc_sig_start,
        bc_sig_end,
        bc_seq,
        ref_sig_start,
        ref_sig_end,
        en,
        st,
        PLR_start,
        PLR_end,
        min_signal,
        span,
    )

    _movetable_plot_qscore_panel(ax_q, bc_sig_start, bc_sig_end, bc_seq, quals, q_max, st, en)

    plt.tight_layout()
    out_png = os.path.join(outdir, f"{io_read.read_id}.png")
    fig.savefig(out_png, dpi=120)
    plt.close(fig)
    return True




def fill_small_gaps_ref(mask, max_gap_segments=2):
    mask = mask.copy()
    n = len(mask)
    i = 0
    while i < n:
        if not mask[i]:
            j = i
            while j < n and not mask[j]:
                j += 1
            gap_len = j - i
            if gap_len <= max_gap_segments:
                mask[i:j] = True
            i = j
        else:
            i += 1
    return mask


def _revcomp(seq):
    comp = str.maketrans("ACGTacgt", "TGCAtgca")
    return seq.translate(comp)[::-1]


def semiglobal_edit_distance(query, target, max_dist):
    qlen, tlen = len(query), len(target)
    if qlen == 0 or tlen == 0:
        return None

    # DP rows
    prev = list(range(tlen + 1))

    for i in range(1, qlen + 1):
        cur = [i] + [0] * tlen
        row_min = cur[0]
        qi = query[i - 1]

        for j in range(1, tlen + 1):
            cost = 0 if qi == target[j - 1] else 1
            cur[j] = min(
                cur[j - 1] + 1,
                prev[j] + 1,
                prev[j - 1] + cost
            )
            row_min = min(row_min, cur[j])

        if row_min > max_dist:
            return None
        prev = cur

    # best endpoint anywhere in target
    j_end = min(range(len(prev)), key=lambda j: prev[j])
    dist = prev[j_end]

    if dist > max_dist:
        return None

    return dist, max(0, j_end - qlen), j_end


def find_adapter_blocks_dorado_style(
    io_read,
    q2s,
    seg_qs_local,
    seg_qe_local,
    st,
    en,
    adapters,
    trim_len=200,
    max_edit_frac=0.45,
    max_gap_segments=5,
):
    read_seq = io_read.seq or ""
    read_len = len(read_seq)
    if read_len == 0:
        return []

    front_window = read_seq[:trim_len]
    

    base_hits = [] 

    rear_edit_frac = min(0.8, max_edit_frac * 2.0)

    for ad in adapters:
        if not ad.front_seq:
            continue
        max_ed = max(1, int(len(ad.front_seq) * max_edit_frac))
        res = semiglobal_edit_distance(ad.front_seq, front_window, max_ed)
        if res:
            dist, s, e = res
            base_hits.append(("FRONT", ad.name, s, e, dist))

    for ad in adapters:
        if not ad.rear_seq:
            continue
        rear_start = max(0, read_len - trim_len - len(ad.rear_seq))
        rear_window = read_seq[rear_start:]
        rear_window_rc = _revcomp(rear_window)

        max_ed = max(1, int(len(ad.rear_seq) * rear_edit_frac))

        res_d = semiglobal_edit_distance(ad.rear_seq, rear_window, max_ed)
        res_r = semiglobal_edit_distance(ad.rear_seq, rear_window_rc, max_ed)

        chosen = None
        if res_d and res_r:
            chosen = res_d if res_d[0] <= res_r[0] else res_r
        else:
            chosen = res_d or res_r

        if not chosen:
            continue

        dist, s, e = chosen
        if chosen is res_r:
            L = len(rear_window)
            s, e = L - e, L - s

        base_hits.append(
            ("REAR", ad.name, rear_start + s, rear_start + e, dist)
        )

    if not base_hits:
        return []

    # pick best per side (base space)
    best = {}
    for side, name, s, e, d in base_hits:
        key = (d, -(e - s))
        if side not in best or key < best[side][0]:
            best[side] = (key, (side, name, s, e, d))

    blocks = []

    # project to segment space
    N = len(seg_qs_local)
    for _, (side, name, bs, be, dist) in best.values():
        qb = min(max(0, bs), len(q2s) - 2)
        qe = min(max(0, be), len(q2s) - 1)

        left_sig = q2s[qb] - st
        right_sig = q2s[qe] - st

        idx = [
            i for i, (s, e) in enumerate(zip(seg_qs_local, seg_qe_local))
            if e > left_sig and s < right_sig
        ]
        if not idx:
            continue

        s_idx, e_idx = idx[0], idx[-1]

        dist_left = s_idx
        dist_right = N - e_idx

        if side == "FRONT" and dist_left > dist_right:
            continue
        if side == "REAR" and dist_right > dist_left:
            continue

        blocks.append((s_idx, e_idx, f"{name},d={dist}"))

    return blocks



def intersect_blocks(adapter_blocks_all,
                     seg_qs_all, seg_qe_all,
                     seg_qs_local, seg_qe_local):
    local_blocks = []

    for s_all, e_all, label in adapter_blocks_all:
        left_sig = seg_qs_all[s_all]
        right_sig = seg_qe_all[e_all]

        overlapping = [
            i for i, (qs, qe) in enumerate(zip(seg_qs_local, seg_qe_local))
            if (qe > left_sig) and (qs < right_sig)
        ]

        if overlapping:
            local_blocks.append(
                (min(overlapping), max(overlapping), s_all, e_all, label)
            )

    return local_blocks


def _knots_to_segments(knots, idx0, idx1, st, en):
    if idx1 <= idx0:
        return np.array([], dtype=int), np.array([], dtype=int)
    s = knots[idx0:idx1]
    e = knots[idx0 + 1:idx1 + 1]
    s_clipped = np.clip(s, st, en) - st
    e_clipped = np.clip(e, st, en) - st
    return np.rint(s_clipped).astype(int), np.rint(e_clipped).astype(int)


def _prepare_figure_io_read(read_record, sam_input):
    if sam_input is None:
        raise ValueError("sam_input is required for the Panel plotting function.")

    aln = sam_input.get_first_alignment(str(read_record.read_id))
    if aln is None:
        raise ValueError(f"No alignment found for read {read_record.read_id}")

    alns = list(sam_input.get_alignments(str(read_record.read_id)))
    if len(alns) > 1:
        try:
            logger.warning(f"Multiple alignments ({len(alns)}) found for read {read_record.read_id}. Using the first one.")
        except NameError:
            pass

    io_read = io.Read.from_pod5_and_alignment(read_record, aln)
    return aln, io_read


def _refine_mapping_and_extract_levels(io_read):
    try:
        io_read.set_refine_signal_mapping(_global_sig_map_refiner, ref_mapping=False)
        try:
            io_read.set_refine_signal_mapping(_global_sig_map_refiner, ref_mapping=True)
        except Exception:
            pass
    except Exception as e:
        try:
            logger.error(f"Error refining signal mapping for read {io_read.read_id}: {e}")
        except NameError:
            pass
        return False, False, None

    ref_rebuilt = False
    try:
        io_read.compute_ref_to_signal()
        ref_rebuilt = (
            io_read.ref_to_signal is not None
            and io_read.ref_to_signal.size == (len(io_read.ref_seq) + 1)
        )
    except Exception:
        ref_rebuilt = False

    move_levels = None
    try:
        move_levels = _global_sig_map_refiner.extract_levels(util.seq_to_int(io_read.seq))
    except Exception:
        move_levels = None

    return True, ref_rebuilt, move_levels


def _build_basecall_segments(io_read, st, en):
    q_knots = np.array(io_read.query_to_signal, dtype=float)
    bc_start_idx = int(np.searchsorted(q_knots, st))
    bc_end_idx = int(np.searchsorted(q_knots, en))
    bc_start_idx = max(0, min(bc_start_idx, len(io_read.seq)))
    bc_end_idx = max(0, min(bc_end_idx, len(io_read.seq)))

    if bc_end_idx <= bc_start_idx:
        return bc_start_idx, np.array([], dtype=int), np.array([], dtype=int), "", []

    bc_s, bc_e = _knots_to_segments(q_knots, bc_start_idx, bc_end_idx, st, en)
    bc_seq = io_read.seq[bc_start_idx:bc_end_idx]
    n = min(len(bc_seq), len(bc_s), len(bc_e))
    bc_sig_start = bc_s[:n]
    bc_sig_end = bc_e[:n]
    bc_q_indices = list(range(bc_start_idx, bc_start_idx + n))
    return bc_start_idx, bc_sig_start, bc_sig_end, bc_q_indices


def _build_query_ref_mappings(aln):
    try:
        pairs = aln.get_aligned_pairs(matches_only=True, with_seq=False)
    except Exception:
        pairs = []
    return pairs


def _build_reference_segments(
    io_read,
    aln,
    ref_rebuilt,
    st,
    en,
    bc_sig_start,
    bc_sig_end,
    bc_q_indices,
    pairs,
):
    ref_sig_start = []
    ref_sig_end = []
    ref_bases = []
    ref_positions_global = []

    ref_seq_str = io_read.ref_seq or ""
    ref_global_start = aln.reference_start
    ref_len_local = len(ref_seq_str)

    if ref_rebuilt:
        knots_ref = np.array(io_read.ref_to_signal, dtype=float)
        r0 = int(np.searchsorted(knots_ref[:-1], st - 1))
        r1 = int(np.searchsorted(knots_ref[:-1], en))
        r0 = max(0, min(r0, len(ref_seq_str)))
        r1 = max(0, min(r1, len(ref_seq_str)))
        rs, re = _knots_to_segments(knots_ref, r0, r1, st, en)
        ref_sig_start = rs.tolist()
        ref_sig_end = re.tolist()
        ref_bases = list(ref_seq_str[r0:r1])
        ref_positions_global = list(np.arange(ref_global_start + r0, ref_global_start + r1))
    else:
        q_to_r = {}
        try:
            pairs_full = aln.get_aligned_pairs(matches_only=False, with_seq=True)
            for p in pairs_full:
                if len(p) < 2:
                    continue
                q, r = p[0], p[1]
                if q is None or r is None:
                    continue
                q = int(q)
                if q not in q_to_r:
                    q_to_r[q] = int(r)
        except Exception:
            for q, r in pairs:
                if q is None or r is None:
                    continue
                q_to_r[int(q)] = int(r)

        for j, q_idx in enumerate(bc_q_indices):
            r_global = q_to_r.get(q_idx)
            if r_global is None:
                continue
            rl = int(r_global) - ref_global_start
            if rl < 0 or rl >= ref_len_local:
                continue
            ref_sig_start.append(int(bc_sig_start[j]))
            ref_sig_end.append(int(bc_sig_end[j]))
            ref_bases.append(ref_seq_str[rl])
            ref_positions_global.append(int(r_global))

        if len(ref_sig_start) < max(1, int(len(bc_q_indices) * 0.25)):
            if getattr(io_read, "ref_to_signal", None) is not None and io_read.ref_to_signal.size == (len(ref_seq_str) + 1):
                knots_ref = np.array(io_read.ref_to_signal, dtype=float)
                r0 = int(np.searchsorted(knots_ref[:-1], st - 1))
                r1 = int(np.searchsorted(knots_ref[:-1], en))
                r0 = max(0, min(r0, len(ref_seq_str)))
                r1 = max(0, min(r1, len(ref_seq_str)))
                rs, re = _knots_to_segments(knots_ref, r0, r1, st, en)
                for i_local, (rs_i, re_i) in enumerate(zip(rs.tolist(), re.tolist())):
                    gpos = ref_global_start + r0 + i_local
                    if gpos in ref_positions_global:
                        continue
                    assigned = False
                    for bcs, bce in zip(bc_sig_start.tolist(), bc_sig_end.tolist()):
                        center = (rs_i + re_i) / 2.0
                        if center >= bcs and center <= bce:
                            ref_sig_start.append(int(bcs))
                            ref_sig_end.append(int(bce))
                            ref_bases.append(ref_seq_str[r0 + i_local])
                            ref_positions_global.append(int(gpos))
                            assigned = True
                            break
                    if not assigned:
                        ref_sig_start.append(int(rs_i))
                        ref_sig_end.append(int(re_i))
                        ref_bases.append(ref_seq_str[r0 + i_local])
                        ref_positions_global.append(int(gpos))

    if not ref_sig_start:
        ref_sig_start = np.array([], dtype=int)
        ref_sig_end = np.array([], dtype=int)
        ref_bases = []
        ref_positions_global = []
    else:
        ref_sig_start = np.array(ref_sig_start, dtype=int) - st
        ref_sig_end = np.array(ref_sig_end, dtype=int) - st

    ref_pos_array = (
        np.array([int(p - ref_global_start) for p in ref_positions_global], dtype=int)
        if len(ref_positions_global) > 0
        else np.array([], dtype=int)
    )

    return ref_sig_start, ref_sig_end, ref_pos_array


def _build_query_signal_segments(q2s, st):
    seg_qs_all = []
    seg_qe_all = []
    for base_i in range(len(q2s) - 1):
        qs = int(q2s[base_i]) - st
        qe = int(q2s[base_i + 1]) - st
        if qe <= qs:
            continue
        seg_qs_all.append(qs)
        seg_qe_all.append(qe)

    return (
        np.asarray(seg_qs_all, dtype=int),
        np.asarray(seg_qe_all, dtype=int),
    )


def _resolve_signal_ylim(sig):
    max_signal = sig.max() if len(sig) > 0 else 4
    min_signal = sig.min() if len(sig) > 0 else -4
    span = max_signal - min_signal
    return min_signal, max_signal, span, (min_signal - span * 0.06, max_signal + span * 0.08)


def _plot_expected_levels_on_zoom(ax, move_levels_plot, q2s, st, z0, z1):
    if move_levels_plot is None:
        return

    xs_steps = []
    ys_steps = []
    for base_i in range(len(move_levels_plot)):
        if base_i + 1 >= len(q2s):
            break
        qs = int(q2s[base_i]) - st
        qe = int(q2s[base_i + 1]) - st
        if qe <= z0 or qs >= z1:
            continue
        x_left = max(qs, z0)
        x_right = min(qe, z1)
        lvl = move_levels_plot[base_i]
        xs_steps.extend([x_left, x_right, None])
        ys_steps.extend([lvl, lvl, None])
        if base_i + 1 < len(move_levels_plot):
            next_lvl = move_levels_plot[base_i + 1]
            bnd = int(q2s[base_i + 1]) - st
            if z0 <= bnd <= z1:
                xs_steps.extend([bnd, bnd, None])
                ys_steps.extend([lvl, next_lvl, None])

    if len(xs_steps) > 0:
        ax.plot(xs_steps, ys_steps, lw=1.4, solid_capstyle='butt', zorder=1, label='Expected level')


def _collect_step_panel_segments(
    move_levels_plot,
    q2s,
    st,
    z0,
    z1,
    bc_sig_start,
    bc_sig_end,
    bc_start_idx,
):
    seg_query_idx = []
    seg_qs_local = []
    seg_qe_local = []
    seg_expected = []
    seg_qs_local_global = []
    seg_qe_local_global = []

    if move_levels_plot is not None:
        for base_i in range(len(move_levels_plot)):
            if base_i + 1 >= len(q2s):
                break
            qs_global = int(q2s[base_i])
            qe_global = int(q2s[base_i + 1])
            qs = qs_global - st
            qe = qe_global - st
            if qe <= z0 or qs >= z1:
                continue

            seg_qs_local_global.append(qs)
            seg_qe_local_global.append(qe)

            s_local = max(qs, z0)
            e_local = min(qe, z1)
            seg_query_idx.append(base_i)
            seg_qs_local.append(int(s_local))
            seg_qe_local.append(int(e_local))
            seg_expected.append(float(move_levels_plot[base_i]) if 0 <= base_i < len(move_levels_plot) else np.nan)
    else:
        for i in range(len(bc_sig_start)):
            qs = int(bc_sig_start[i])
            qe = int(bc_sig_end[i])
            if qe <= z0 or qs >= z1:
                continue

            seg_qs_local_global.append(qs)
            seg_qe_local_global.append(qe)

            s_local = max(qs, z0)
            e_local = min(qe, z1)
            seg_query_idx.append(bc_start_idx + i)
            seg_qs_local.append(int(s_local))
            seg_qe_local.append(int(e_local))
            seg_expected.append(np.nan)

    return {
        "seg_query_idx": seg_query_idx,
        "seg_qs_local": seg_qs_local,
        "seg_qe_local": seg_qe_local,
        "seg_expected": np.array(seg_expected),
        "seg_qs_local_global": seg_qs_local_global,
        "seg_qe_local_global": seg_qe_local_global,
    }


def _compute_segment_median_and_std(sig, seg_qs_local, seg_qe_local):
    seg_actual_med = []
    seg_actual_std = []
    for s_local, e_local in zip(seg_qs_local, seg_qe_local):
        s_i = max(0, min(len(sig), s_local))
        e_i = max(0, min(len(sig), e_local))
        if e_i <= s_i:
            e_i = min(len(sig), s_i + 1)
        segment = sig[s_i:e_i]
        if len(segment) == 0:
            seg_actual_med.append(np.nan)
            seg_actual_std.append(0.0)
        else:
            seg_actual_med.append(float(np.median(segment)))
            seg_actual_std.append(float(np.std(segment)))
    return np.array(seg_actual_med), np.array(seg_actual_std)


def _template_threading_masks(ref_name, ref_pos_array, template_color, threading_color):
    if ref_name in ("template", "template_N0_threading", "template_N20_threading", "template_threading_0N"):
        template_ref_mask = ref_pos_array <= 73
        threading_ref_mask = ref_pos_array >= 80
    elif ref_name == "template_cysteinefree":
        template_ref_mask = ref_pos_array <= 99
        threading_ref_mask = np.zeros_like(ref_pos_array, dtype=bool)
    elif ref_name == "template_cysteinefree_revComp":
        template_ref_mask = ref_pos_array <= 99
        threading_ref_mask = ref_pos_array >= 106
        threading_color = template_color
    elif ref_name == "template_N0_revComTemplate":
        template_ref_mask = ref_pos_array <= 73
        threading_ref_mask = ref_pos_array >= 80
        threading_color = template_color
    elif ref_name == "threading_revComp":
        template_ref_mask = ref_pos_array <= 55
        threading_ref_mask = ref_pos_array >= 65
        template_color = threading_color
    else:
        template_ref_mask = np.zeros_like(ref_pos_array, dtype=bool)
        threading_ref_mask = np.zeros_like(ref_pos_array, dtype=bool)

    return template_ref_mask, threading_ref_mask, template_color, threading_color


def _ref_blocks_from_mask(mask):
    blocks = []
    if mask.size == 0 or not np.any(mask):
        return blocks
    true_idx = np.where(mask)[0]
    runs = np.split(true_idx, np.where(np.diff(true_idx) != 1)[0] + 1)
    for r in runs:
        blocks.append((int(r[0]), int(r[-1])))
    return blocks


def _ref_blocks_to_segment_blocks(ref_blocks, seg_qs_local, seg_qe_local, ref_sig_start, ref_sig_end, max_gap_segments):
    seg_mask = np.zeros(len(seg_qs_local), dtype=bool)
    for (r0, r1) in ref_blocks:
        left_sig = float(ref_sig_start[r0])
        right_sig = float(ref_sig_end[r1])
        for si, (s_local, e_local) in enumerate(zip(seg_qs_local, seg_qe_local)):
            if e_local > left_sig and s_local < right_sig:
                seg_mask[si] = True
    seg_mask_filled = fill_small_gaps_ref(seg_mask, max_gap_segments)
    true_idx = np.where(seg_mask_filled)[0]
    if true_idx.size == 0:
        return []
    runs = np.split(true_idx, np.where(np.diff(true_idx) != 1)[0] + 1)
    return [(int(r[0]), int(r[-1])) for r in runs]


def _draw_bottom_panel_overlays(
    ax,
    global_ylim,
    template_blocks,
    threading_blocks,
    adapter_blocks,
    plr_seg_idxs,
    ref_name,
    template_color,
    threading_color,
    adapter_color,
    orange_color,
):
    y0, y1 = global_ylim
    yrange = y1 - y0
    bar_y = y0 + 0.5
    bar_height = max(yrange * 0.02, 1e-6)
    stack_offset = bar_height + max(yrange * 0.002, 1e-8)

    for (s_idx, e_idx) in template_blocks:
        left = s_idx - 0.5
        right = e_idx + 0.5
        rect = Rectangle((left, bar_y), right - left, bar_height,
                         linewidth=0.0, edgecolor='none', facecolor=template_color, alpha=0.26, zorder=5)
        ax.add_patch(rect)
        mid_x = (left + right) / 2.0
        text_y = bar_y + bar_height + 0.02
        title = 'Template'
        if ref_name == "threading_revComp":
            title = 'Threading'
        ax.text(mid_x, text_y, title, ha='center', va='bottom', fontsize=7, zorder=6)

    for (s_idx, e_idx) in threading_blocks:
        left = s_idx - 0.5
        right = e_idx + 0.5
        rect_y = bar_y + stack_offset if len(threading_blocks) > 0 else bar_y
        rect = Rectangle((left, rect_y), right - left, bar_height,
                         linewidth=0.0, edgecolor='none', facecolor=threading_color, alpha=0.26, zorder=5)
        ax.add_patch(rect)
        mid_x = (left + right) / 2.0
        text_y = rect_y + bar_height + 0.02
        title = 'Threading'
        if ref_name == "template_N0_revComTemplate" or ref_name == "template_cysteinefree_revComp":
            title = 'RC Template'
        elif ref_name == "threading_revComp":
            title = 'RC Threading'
        ax.text(mid_x, text_y, title, ha='center', va='bottom', fontsize=7, zorder=6)

    for (s_idx, e_idx, s_all, e_all, label) in adapter_blocks:
        left = s_idx - 0.5
        right = e_idx + 0.5
        rect_y = bar_y + stack_offset if len(threading_blocks) > 0 else bar_y
        rect = Rectangle((left, bar_y), right - left, bar_height + 0.05,
                         linewidth=0.0, edgecolor='none', facecolor=adapter_color, alpha=0.26, zorder=4)
        mid_x = (left + right) / 2.0
        text_y = rect_y + bar_height - 0.06
        ax.text(mid_x, text_y, f'{label}', ha='center', va='bottom', fontsize=7, zorder=6)
        ax.add_patch(rect)

    if plr_seg_idxs:
        left = min(plr_seg_idxs) - 0.5
        right = max(plr_seg_idxs) + 0.5
        rect = Rectangle((left, y0), right - left, yrange,
                         linewidth=0.8, edgecolor=orange_color, facecolor=orange_color, alpha=0.10, zorder=1)
        ax.add_patch(rect)

    for (s_idx, e_idx) in template_blocks:
        left = s_idx - 0.5
        right = e_idx + 0.5
        rect = Rectangle((left, bar_y), right - left, bar_height,
                         linewidth=0.0, edgecolor='none', facecolor=template_color, alpha=0.26, zorder=4)
        ax.add_patch(rect)

    for i, (s_idx, e_idx) in enumerate(threading_blocks):
        left = s_idx - 0.5
        right = e_idx + 0.5
        rect = Rectangle((left, bar_y + bar_height + 0.005 * yrange), right - left, bar_height,
                         linewidth=0.0, edgecolor='none', facecolor=threading_color, alpha=0.26, zorder=4)
        ax.add_patch(rect)

    return bar_y, bar_height, stack_offset


def _draw_top_panel_overlays(
    top_ax,
    ref_name,
    ref_pos_array,
    template_color,
    threading_color,
    max_gap_segments,
    ref_sig_start,
    ref_sig_end,
    en,
    st,
    bar_y,
    bar_height,
    stack_offset,
    global_ylim,
    adapter_blocks_all,
    seg_qs_all,
    seg_qe_all,
    adapter_color,
):
    try:
        if ref_name in ("template", "template_N0_threading", "template_N20_threading", "template_threading_0N"):
            template_mask_ref = ref_pos_array <= 73
            threading_mask_ref = ref_pos_array >= 80
        elif ref_name == "template_cysteinefree":
            template_mask_ref = ref_pos_array <= 99
            threading_mask_ref = np.zeros_like(ref_pos_array, dtype=bool)
        elif ref_name == "template_cysteinefree_revComp":
            template_mask_ref = ref_pos_array <= 99
            threading_mask_ref = ref_pos_array >= 106
            threading_color = template_color
        elif ref_name == "template_N0_revComTemplate":
            template_mask_ref = ref_pos_array <= 73
            threading_mask_ref = ref_pos_array >= 80
            threading_color = template_color
        elif ref_name == "threading_revComp":
            template_mask_ref = ref_pos_array <= 55
            threading_mask_ref = ref_pos_array >= 65
            template_color = threading_color
        else:
            template_mask_ref = np.zeros_like(ref_pos_array, dtype=bool)
            threading_mask_ref = np.zeros_like(ref_pos_array, dtype=bool)

        def fill_gaps_ref(mask, max_gap):
            if mask.size == 0:
                return mask.copy()
            m = mask.astype(bool).copy()
            true_idx = np.where(m)[0]
            if true_idx.size == 0:
                return m
            runs = np.split(true_idx, np.where(np.diff(true_idx) != 1)[0] + 1)
            for i in range(len(runs) - 1):
                end_prev = runs[i][-1]
                start_next = runs[i + 1][0]
                gap = start_next - end_prev - 1
                if 0 < gap <= max_gap:
                    m[end_prev + 1:start_next] = True
            return m

        def blocks_from_ref_mask(mask):
            blocks = []
            if mask.size == 0 or not np.any(mask):
                return blocks
            true_idx = np.where(mask)[0]
            runs = np.split(true_idx, np.where(np.diff(true_idx) != 1)[0] + 1)
            for r in runs:
                blocks.append((int(r[0]), int(r[-1])))
            return blocks

        template_mask_ref_filled = fill_gaps_ref(template_mask_ref, max_gap_segments)
        threading_mask_ref_filled = fill_gaps_ref(threading_mask_ref, max_gap_segments)
        template_ref_blocks = blocks_from_ref_mask(template_mask_ref_filled)
        threading_ref_blocks = blocks_from_ref_mask(threading_mask_ref_filled)

        for (i0, i1) in template_ref_blocks:
            left_sig = float(ref_sig_start[i0])
            right_sig = float(ref_sig_end[i1])
            left_sig = max(left_sig, 0.0)
            right_sig = min(right_sig, en - st)
            if right_sig <= left_sig:
                continue
            rect = Rectangle((left_sig, bar_y), right_sig - left_sig, bar_height,
                             linewidth=0.0, edgecolor='none', facecolor=template_color, alpha=0.26, zorder=6)
            top_ax.add_patch(rect)
            mid_x = (left_sig + right_sig) / 2.0
            label_y = bar_y + bar_height + 0.01 * (global_ylim[1] - global_ylim[0])
            title = 'Template DNA'
            if ref_name == "threading_revComp":
                title = 'Threading DNA'
            top_ax.text(mid_x, label_y, title, ha='center', va='bottom', fontsize=7, color=template_color, zorder=7)

        for (i0, i1) in threading_ref_blocks:
            left_sig = float(ref_sig_start[i0])
            right_sig = float(ref_sig_end[i1])
            left_sig = max(left_sig, 0.0)
            right_sig = min(right_sig, en - st)
            if right_sig <= left_sig:
                continue
            rect_y = bar_y + (stack_offset if len(template_ref_blocks) > 0 else 0.0)
            rect = Rectangle((left_sig, rect_y), right_sig - left_sig, bar_height,
                             linewidth=0.0, edgecolor='none', facecolor=threading_color, alpha=0.26, zorder=6)
            top_ax.add_patch(rect)
            mid_x = (left_sig + right_sig) / 2.0
            label_y = rect_y + bar_height + 0.01 * (global_ylim[1] - global_ylim[0])
            title = 'Threading DNA'
            if ref_name == "template_N0_revComTemplate" or ref_name == "template_cysteinefree_revComp":
                title = 'RC Template'
            if ref_name == "threading_revComp":
                title = 'RC Threading'
            top_ax.text(mid_x, label_y, title, ha='center', va='bottom', fontsize=7, color=threading_color, zorder=7)

        if adapter_blocks_all:
            for (s_all, e_all, label) in adapter_blocks_all:
                left_sig = float(seg_qs_all[s_all])
                right_sig = float(seg_qe_all[e_all])
                left_sig = max(left_sig, 0.0)
                right_sig = min(right_sig, en - st)
                if right_sig <= left_sig:
                    continue

                adapter_rect_y = bar_y + (
                    2 * stack_offset
                    if (len(template_ref_blocks) > 0 or len(threading_ref_blocks) > 0)
                    else stack_offset
                )

                rect = Rectangle(
                    (left_sig, adapter_rect_y),
                    right_sig - left_sig,
                    bar_height,
                    linewidth=0.0,
                    edgecolor='none',
                    facecolor=adapter_color,
                    alpha=0.26,
                    zorder=6,
                )
                top_ax.add_patch(rect)

                mid_x = (left_sig + right_sig) / 2.0
                label_y = adapter_rect_y + bar_height + 0.01 * (global_ylim[1] - global_ylim[0])
                top_ax.text(
                    mid_x, label_y, label,
                    ha='center', va='bottom',
                    fontsize=7, color=adapter_color, zorder=7,
                )
    except Exception:
        pass

def plot_single_read_movetable_panels(
    read_record,
    sam_input,
    outdir,
    PLR_start,
    PLR_end,
    signal_type='norm',
    sig_start=None,
    sig_end=None,
    signal_zoom=300,
):
    orange_color = '#ff7f2a'
    template_color = '#2ca02c'
    threading_color = '#9467bd'
    adapter_color = '#808080'

    aln, io_read = _prepare_figure_io_read(read_record, sam_input)

    ok_refine, ref_rebuilt, move_levels = _refine_mapping_and_extract_levels(io_read)
    if not ok_refine:
        return False

    sig_len = io_read._sig_len
    st = 0 if sig_start is None else sig_start
    en = sig_len if sig_end is None else min(sig_end, sig_len)

    bc_start_idx, bc_sig_start, bc_sig_end, bc_q_indices = _build_basecall_segments(io_read, st, en)
    pairs = _build_query_ref_mappings(aln)
    ref_sig_start, ref_sig_end, ref_pos_array = _build_reference_segments(
        io_read,
        aln,
        ref_rebuilt,
        st,
        en,
        bc_sig_start,
        bc_sig_end,
        bc_q_indices,
        pairs,
    )

    sig = io_read.get_sig_type(signal_type)[st:en]
    t = np.arange(st, en) - st
    q2s = io_read.query_to_signal

    seg_qs_all, seg_qe_all = _build_query_signal_segments(q2s, st)

    plt.rcParams.update({
        "font.size": 9,
        "axes.linewidth": 0.8,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "font.family": "sans-serif"
    })

    fig, (ax_full, ax_zoom, ax_pos) = plt.subplots(
        3, 1,
        figsize=(6.5, 6.0),
        gridspec_kw={'height_ratios': [1.15, 1.0, 1.0]},
        sharex=False
    )

    min_signal, max_signal, span, global_ylim = _resolve_signal_ylim(sig)

    ax = ax_full
    ax.plot(t, sig, lw=0.6, color='black')
    ax.set_ylabel("Current (Normalized)" if signal_type != "pa" else "Current (pA)", fontsize=9)
    ax.set_xlabel("Signal index", fontsize=9)
    ax.set_ylim(*global_ylim)
    ax.set_frame_on(True)
    ax.tick_params(axis='both', which='major', labelsize=8)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.6)

    if PLR_start is not None and PLR_end is not None:
        plr_x0 = max(PLR_start - st, 0)
        plr_x1 = min(PLR_end - st, en - st)
        width = plr_x1 - plr_x0
        rect = Rectangle((plr_x0, min_signal - span * 0.05), width, span * 1.1,
                         linewidth=0.8, edgecolor='none', facecolor=orange_color, zorder=5, alpha=0.12)
        ax.add_patch(rect)

    ax = ax_zoom

    if PLR_start is None or PLR_end is None:
        if len(ref_sig_start) > 0 and len(ref_sig_end) > 0:
            z0 = int(max(0, ref_sig_start[0] - signal_zoom))
            z1 = int(min(en-st, ref_sig_end[-1] + signal_zoom))
        else:
            z0 = 0
            z1 = min(en-st, signal_zoom)
    else:
        pad = max(signal_zoom, int(0.05 * (en-st)))
        z0 = int(max(0, PLR_start - st - pad))
        z1 = int(min(en-st, PLR_end - st + pad))

    zoom_t = np.arange(z0, z1)
    zoom_sig = sig[z0:z1]

    ax.plot(zoom_t, zoom_sig, lw=0.6, color='black', zorder=2)
    ax.set_xlim(z0, z1)
    ax.set_ylim(*global_ylim)
    ax.set_xlabel("Signal index", fontsize=9)
    ax.set_ylabel("Current (normalized)", fontsize=9)

    if PLR_start is not None and PLR_end is not None:
        plr_x0_z = max(PLR_start - st, z0)
        plr_x1_z = min(PLR_end - st, z1)
        rect_fill = Rectangle((plr_x0_z, min_signal - span * 0.06), plr_x1_z - plr_x0_z, (max_signal - min_signal) + span * 0.14,
                              linewidth=0.0, edgecolor='none', facecolor=orange_color, alpha=0.12, zorder=4)
        ax.add_patch(rect_fill)

    move_levels_plot = None
    if move_levels is not None:
        if signal_type == "pa":
            move_levels_plot = move_levels * io_read.scale_pa_to_norm + io_read.shift_pa_to_norm
        else:
            move_levels_plot = move_levels.copy()

    _plot_expected_levels_on_zoom(ax, move_levels_plot, q2s, st, z0, z1)

    ax.tick_params(axis='both', which='major', labelsize=8)
    ax.legend(loc='upper right', fontsize=6, frameon=False)

    ax = ax_pos

    plr_local_start = None
    plr_local_end = None
    if PLR_start is not None and PLR_end is not None:
        plr_local_start = PLR_start - st
        plr_local_end = PLR_end - st

    step_data = _collect_step_panel_segments(
        move_levels_plot,
        q2s,
        st,
        z0,
        z1,
        bc_sig_start,
        bc_sig_end,
        bc_start_idx,
    )

    seg_query_idx = step_data["seg_query_idx"]
    seg_qs_local = step_data["seg_qs_local"]
    seg_qe_local = step_data["seg_qe_local"]
    seg_expected = step_data["seg_expected"]
    seg_qs_local_global = step_data["seg_qs_local_global"]
    seg_qe_local_global = step_data["seg_qe_local_global"]

    if len(seg_query_idx) == 0:
        try:
            logger.warning(f"No segments overlapping zoom window for read {io_read.read_id}.")
        except NameError:
            pass
        return False

    seg_actual_med, seg_actual_std = _compute_segment_median_and_std(sig, seg_qs_local, seg_qe_local)

    N_seg = len(seg_query_idx)
    x_steps = np.arange(N_seg)

    if np.any(~np.isnan(seg_expected)):
        ax.step(x_steps, seg_expected, where='mid', linestyle='-', linewidth=1.4,
                label='Expected level', zorder=2, color='tab:blue')

    lower = seg_actual_med - seg_actual_std
    upper = seg_actual_med + seg_actual_std
    if np.any(~np.isnan(lower)) and np.any(~np.isnan(upper)):
        ax.fill_between(x_steps, lower, upper, where=~np.isnan(seg_actual_med),
                        step='mid', alpha=0.22, linewidth=0.0, zorder=1, color='black')
    ax.step(x_steps, seg_actual_med, where='mid', linestyle='-', linewidth=1.4,
            label='Actual level', zorder=3, color='black')

    if plr_local_start is not None and plr_local_end is not None:
        plr_seg_idxs = [idx for idx, (s_local, e_local) in enumerate(zip(seg_qs_local, seg_qe_local))
                        if (e_local > plr_local_start) and (s_local < plr_local_end)]
    else:
        plr_seg_idxs = []

    max_gap_segments = 2
    ref_name = aln.reference_name

    template_ref_mask, threading_ref_mask, template_color, threading_color = _template_threading_masks(
        ref_name,
        ref_pos_array,
        template_color,
        threading_color,
    )
    template_ref_mask_filled = fill_small_gaps_ref(template_ref_mask, max_gap_segments)
    threading_ref_mask_filled = fill_small_gaps_ref(threading_ref_mask, max_gap_segments)
    template_ref_blocks = _ref_blocks_from_mask(template_ref_mask_filled)
    threading_ref_blocks = _ref_blocks_from_mask(threading_ref_mask_filled)
    template_blocks = _ref_blocks_to_segment_blocks(
        template_ref_blocks,
        seg_qs_local,
        seg_qe_local,
        ref_sig_start,
        ref_sig_end,
        max_gap_segments,
    )
    threading_blocks = _ref_blocks_to_segment_blocks(
        threading_ref_blocks,
        seg_qs_local,
        seg_qe_local,
        ref_sig_start,
        ref_sig_end,
        max_gap_segments,
    )

    adapter_blocks_all = find_adapter_blocks_dorado_style(
        io_read,
        q2s,
        seg_qs_all,
        seg_qe_all,
        0,
        sig_len,
        adapters=DORADO_ADAPTERS,
        trim_len=500,
        max_edit_frac=0.45,
        max_gap_segments=max_gap_segments
    )
    adapter_blocks = intersect_blocks(
        adapter_blocks_all,
        seg_qs_all,
        seg_qe_all,
        seg_qs_local_global,
        seg_qe_local_global
    )

    bar_y, bar_height, stack_offset = _draw_bottom_panel_overlays(
        ax,
        global_ylim,
        template_blocks,
        threading_blocks,
        adapter_blocks,
        plr_seg_idxs,
        ref_name,
        template_color,
        threading_color,
        adapter_color,
        orange_color,
    )

    ax.set_xlim(-0.5, max(N_seg - 0.5, 0.5))
    ax.set_ylim(*global_ylim)

    step = 10
    if N_seg <= 1:
        tick_idxs = np.array([0], dtype=int)
    else:
        tick_idxs = np.arange(0, N_seg, step, dtype=int)
        tick_idxs = tick_idxs[tick_idxs < N_seg]

    ax.set_xticks(tick_idxs)
    ax.set_xticklabels([str(int(i)) for i in tick_idxs], fontsize=8)
    ax.set_xlabel("Move-table segment index", fontsize=9)
    ax.set_ylabel("Current (Normalized)" if signal_type != "pa" else "Current (pA)", fontsize=9)
    ax.tick_params(axis='both', which='major', labelsize=8)
    ax.legend(loc='upper right', fontsize=6, frameon=False)


    _draw_top_panel_overlays(
        ax_full,
        ref_name,
        ref_pos_array,
        template_color,
        threading_color,
        max_gap_segments,
        ref_sig_start,
        ref_sig_end,
        en,
        st,
        bar_y,
        bar_height,
        stack_offset,
        global_ylim,
        adapter_blocks_all,
        seg_qs_all,
        seg_qe_all,
        adapter_color,
    )

    read_id = io_read.read_id
    title = f"Read ID: {read_id}"
    fig.suptitle(title, fontsize=10, y=0.98)
    fig.subplots_adjust(hspace=0.34, top=0.92)

    out_base = os.path.join(outdir, f"{read_id}_panel")
    os.makedirs(outdir, exist_ok=True)
    out_png = out_base + ".png"
    out_svg = out_base + ".svg"

    fig.savefig(out_png, dpi=300, bbox_inches='tight')
    fig.savefig(out_svg, dpi=300, bbox_inches='tight')
    plt.close(fig)

    return True

def plot_single_read_alignment(
    read_record,
    sam_input,
    outdir,
    PLR_start=None,
    PLR_end=None,
    signal_type="pa",
):
    aln     = sam_input.get_first_alignment(str(read_record.read_id))
    io_read = io.Read.from_pod5_and_alignment(read_record, aln)
    sig     = io_read.get_sig_type(signal_type)
    trim   = io_read._trim_tags.get('ts', 0)
    if PLR_start is not None:
        PLR_start -= trim
        PLR_end   -= trim

    pairs = aln.get_aligned_pairs(matches_only=True, with_seq=False)
    ref_to_query = {r: q for q, r in pairs}

    ref_seq     = io_read.ref_seq
    R           = len(ref_seq)
    xs          = np.arange(R)
    signal_cols = []
    bc_letters  = []
    for r in xs:
        if r in ref_to_query:
            q = ref_to_query[r]
            s = io_read.query_to_signal[q]
            e = io_read.query_to_signal[q+1]
            signal_cols.append(sig[s:e])
            bc_letters.append(io_read.seq[q])
        else:
            signal_cols.append([])
            bc_letters.append('-')

    try:
        io_read.set_refine_signal_mapping(_global_sig_map_refiner, ref_mapping=False)
    except:
        pass
    move = _global_sig_map_refiner.extract_levels(util.seq_to_int(io_read.seq))
    if move is not None and signal_type=="pa":
        move = move * io_read.scale_pa_to_norm + io_read.shift_pa_to_norm

    fig = plt.figure(figsize=(max(12, R*0.3), 6))
    gs0 = fig.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.3)

    ax_sig = fig.add_subplot(gs0[0])

    gs1 = gs0[1].subgridspec(2, 1, height_ratios=[0.5, 0.5], hspace=0.0)
    ax_ref = fig.add_subplot(gs1[0], sharex=ax_sig)
    ax_bc  = fig.add_subplot(gs1[1], sharex=ax_sig)



    data = [col if len(col)>0 else [np.nan] for col in signal_cols]
    ax_sig.boxplot(
        data, positions=xs, widths=0.9, showfliers=False,
        patch_artist=True,
        boxprops=dict(facecolor='lightgray', edgecolor='black'),
        medianprops=dict(color='black')
    )
    if move is not None:
        for r, q in ref_to_query.items():
            if 0 <= r < R:
                lvl = move[q]
                x0 = xs[r] - 0.45
                ax_sig.hlines(lvl, x0, x0+0.9,
                              colors='violet', linewidth=1.2)
    ax_sig.set_ylabel("Signal", rotation=0, labelpad=40, va='center')
    ax_sig.set_ylim(MIN_SIGNAL, MAX_SIGNAL)
    ax_sig.grid(axis='y', linestyle='--', alpha=0.3)
    ax_sig.set_xticks([])

    ax_ref.axis('off')
    ax_ref.set_ylim(0,1)
    ax_ref.text(-0.5, 0.5, "Reference",
                ha='right', va='center',
                fontsize=10, fontweight='bold')
    for i, base in enumerate(ref_seq):
        color = BASE_COLORS.get(base, 'lightgray')
        ax_ref.add_patch(Rectangle((i-0.45, 0.2), 0.9, 0.6,
                                  facecolor=color,
                                  edgecolor='black', lw=0.5))
        ax_ref.text(i, 0.5, base,
                    ha='center', va='center', fontsize=8)
        pos_col = 'black'
        if PLR_start is not None and i in ref_to_query:
            q = ref_to_query[i]
            s = io_read.query_to_signal[q]
            e = io_read.query_to_signal[q+1]
            if e > PLR_start and s < PLR_end:
                pos_col = 'orange'
        ax_ref.text(i, 1.02, str(i+1),
                    ha='center', va='bottom',
                    fontsize=10, color=pos_col)

    ax_bc.axis('off')
    ax_bc.set_ylim(0,1)
    ax_bc.text(-0.5, 0.5, "Basecalls",
               ha='right', va='center',
               fontsize=10, fontweight='bold')
    for i, bc in enumerate(bc_letters):
        if bc == '-':
            continue
        color = BASE_COLORS.get(bc, 'lightgray')
        ax_bc.add_patch(Rectangle((i-0.45, 0.2), 0.9, 0.6,
                                  facecolor=color,
                                  edgecolor='black', lw=0.5))
        ax_bc.text(i, 0.5, bc,
                   ha='center', va='center', fontsize=8)

    ax_bc.set_xticks(xs)
    ax_bc.set_xticklabels([str(i+1) for i in xs],
                          fontsize=6, rotation=90)
    ax_bc.set_xlabel("Reference position")

    ref_name  = aln.reference_name
    start_pos = aln.reference_start + 1
    end_pos   = aln.reference_end
    fig.suptitle(
        f"Read {io_read.read_id} → {ref_name}:{start_pos}-{end_pos}",
        y=0.95, fontsize=10
    )

    os.makedirs(outdir, exist_ok=True)
    out_png = os.path.join(outdir, f"{io_read.read_id}_alignment.png")
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    return True


def plot_one_record(args):
    rec, pod5_path, sam, outdir, signal_type = args
    with p5.Reader(pod5_path) as reader:
        rr = next(reader.reads([rec["Read_ID"]], missing_ok=False))
        plot_single_read_movetable(rr, sam, outdir, rec.get("PLR_start"), rec.get("PLR_end"), signal_type=signal_type)
        plot_single_read_movetable_panels(rr, sam, outdir, rec.get("PLR_start"), rec.get("PLR_end"), signal_type=signal_type)


def plot_single_read(raw_signal, selected_read_id, outdir, PLR_start, PLR_end, html_bool=False, max_plots=150):
    os.makedirs(outdir, exist_ok=True)

    existing_images = [f for f in os.listdir(outdir) if f.endswith(('.jpg', '.png'))]
    if len(existing_images) >= max_plots:
        logger.info(f"Skipping plot for {selected_read_id}: more than {max_plots} plots already exist in {outdir}")
        return

    signal = raw_signal
    timepoints = np.arange(len(signal))
    plot_out = os.path.join(outdir, selected_read_id)

    if html_bool:
        fig = go.Figure()

        fig.add_trace(go.Scatter(x=timepoints, y=signal, mode='lines', name='Raw Signal'))

        if PLR_start is not None:
            PLR_indices = np.arange(PLR_start, PLR_end)
            fig.add_trace(go.Scatter(
                x=timepoints[PLR_indices], 
                y=signal[PLR_indices], 
                mode='lines', 
                line=dict(color='red', width=2),
                name='Peptide-Linker Region'
            ))

        fig.update_layout(
            xaxis=dict(rangeslider=dict(visible=True)),
            title=f"Signal for read {selected_read_id}",
            xaxis_title="Time (seconds)",
            yaxis_title="Raw signal (pA)",
            font_family="Times New Roman",
            font_color="blue",
            title_font_family="Times New Roman",
            title_font_color="black",
            legend_title_font_color="green",
        )

    
        fig.write_html(f"{plot_out}.html")

    plt.figure(figsize=(12, 3))
    plt.plot(timepoints, signal, label='Raw Signal')

    if PLR_start is not None:
        # plt.plot(timepoints[peptide_linker_region], signal[peptide_linker_region], color='red', label='Peptide-Linker Region')
        plt.axvspan(timepoints[PLR_start], timepoints[PLR_end], color='red', alpha=0.3, label='Peptide-Linker Region')
    plt.title(f"Signal for read {selected_read_id}")
    plt.xlabel("Time")
    plt.ylabel("Raw signal (pA)")
    plt.tight_layout()

    plt.savefig(f"{plot_out}.jpg")
    if html_bool:
        plt.savefig(f"{plot_out}.svg")
    plt.close() 

