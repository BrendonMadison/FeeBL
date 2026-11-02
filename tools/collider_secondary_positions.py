"""Compact stored-z partitions of native tracked records, with shared All axes.

The selector uses spatial z at the saved tracker state. It does not select by
longitudinal momentum, charge, detector arm, or extrapolated detector position.
"""

import bisect
import math

DEFINITION = dict(
    coordinate="stored native z position",
    unit="nm",
    positive="z > 0",
    negative="z < 0",
    zero="z == 0; included only in All",
    direction_not_used=True,
)
KEYS = ("theta", "theta_folded", "pT")
EXCLUSIONS = (
    "theta_nonpositive",
    "pT_nonpositive",
    "both_nonpositive",
    "nonpositive_records",
    "underflow_records",
    "overflow_records",
)
MODE = "centre of highest count-density histogram bin; approximate"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def count(value):
    return type(value) is int and 0 <= value <= 1_000_000


def definition_valid(value):
    require(
        value == DEFINITION and type(value.get("direction_not_used")) is bool,
        "invalid stored-z position definition",
    )


def bin_index(edges, value):
    if value < edges[0]:
        return -1
    if value > edges[-1]:
        return len(edges) - 1
    return min(len(edges) - 2, bisect.bisect_right(edges, value) - 1)


def compact_marginal(values, edges):
    counts = [0] * (len(edges) - 1)
    under = over = 0
    for value in values:
        i = bin_index(edges, value)
        if i < 0:
            under += 1
        elif i == len(counts):
            over += 1
        else:
            counts[i] += 1
    stats = None
    if values:
        mean = math.fsum(values) / len(values)
        variance = math.fsum((v - mean) ** 2 for v in values) / len(values)
        mode = None
        method = "unavailable: no in-range marginal records"
        if sum(counts):
            i = max(range(len(counts)), key=lambda j: counts[j] / (edges[j + 1] - edges[j]))
            mode = (edges[i] + edges[i + 1]) / 2
            method = MODE
        stats = dict(
            mean=mean,
            variance=variance,
            min=min(values),
            max=max(values),
            mode=mode,
            mode_method=method,
        )
    return dict(
        counts=counts,
        samples=len(values),
        underflow=under,
        overflow=over,
        total_weight=len(values),
        statistics=stats,
    )


def compact_joint(theta, pt, x, y):
    counts = {}
    ex = dict.fromkeys(EXCLUSIONS, 0)
    for angle, momentum in zip(theta, pt):
        a, p = angle <= 0, momentum <= 0
        ex["theta_nonpositive"] += a
        ex["pT_nonpositive"] += p
        ex["both_nonpositive"] += a and p
        if a or p:
            ex["nonpositive_records"] += 1
            continue
        ix, iy = bin_index(x, angle), bin_index(y, momentum)
        if ix < 0 or iy < 0:
            ex["underflow_records"] += 1
        elif ix == len(x) - 1 or iy == len(y) - 1:
            ex["overflow_records"] += 1
        else:
            i = iy * (len(x) - 1) + ix
            counts[i] = counts.get(i, 0) + 1
    return dict(
        counts_sparse=sorted([i, n] for i, n in counts.items()),
        samples=len(theta),
        in_range_records=sum(counts.values()),
        excluded=ex,
    )


def population_axes(pop):
    return dict(
        theta_edges_rad=pop["joint"]["theta_edges_rad"],
        theta_folded_edges_rad=pop["joint_folded"]["theta_edges_rad"],
        pT_edges_GeV=pop["joint"]["pT_edges_GeV"],
        **{
            "marginal_"
            + key
            + "_edges_"
            + ("GeV" if key == "pT" else "rad"): [
                b["low"] for b in pop["distributions"][key]["bins"]
            ]
            + [pop["distributions"][key]["bins"][-1]["high"]]
            for key in KEYS
        },
    )


def compact_population(pop):
    return dict(
        nrecords=pop["nrecords"],
        joint=pop["joint"],
        joint_folded=pop["joint_folded"],
        marginals={
            key: dict(
                counts=[b["weight"] for b in pop["distributions"][key]["bins"]],
                **{
                    k: pop["distributions"][key][k]
                    for k in ("samples", "underflow", "overflow", "total_weight", "statistics")
                },
            )
            for key in KEYS
        },
        selection=pop["selection"],
    )


def build(theta, pt, folded, positions, pop):
    """positions contains (stored z, process key, signed-energy key) per row."""
    axes = population_axes(pop)
    result = dict(zero_records=sum(z == 0 for z, _, _ in positions))
    for name, sign in [("positive", 1), ("negative", -1)]:
        indices = [i for i, (z, _, _) in enumerate(positions) if z * sign > 0]
        angles = [theta[i] for i in indices]
        momenta = [pt[i] for i in indices]
        folds = [folded[i] for i in indices]
        processes, signs = {}, dict(positive_energy=0, negative_energy=0)
        for i in indices:
            _, process, energy_sign = positions[i]
            processes[process] = processes.get(process, 0) + 1
            signs[energy_sign] += 1
        result[name] = dict(
            nrecords=len(indices),
            joint=compact_joint(angles, momenta, axes["theta_edges_rad"], axes["pT_edges_GeV"]),
            joint_folded=compact_joint(
                folds, momenta, axes["theta_folded_edges_rad"], axes["pT_edges_GeV"]
            ),
            marginals={
                key: compact_marginal(
                    values, axes["marginal_" + key + "_edges_" + ("GeV" if key == "pT" else "rad")]
                )
                for key, values in [("theta", angles), ("theta_folded", folds), ("pT", momenta)]
            },
            selection=dict(process_counts=processes, signed_energy_counts=signs),
        )
    validate(result, compact_population(pop), axes)
    return result


def dense_joint(joint, cells):
    pairs = joint.get("counts_sparse")
    require(
        set(joint) == {"counts_sparse", "samples", "in_range_records", "excluded"}
        and isinstance(pairs, list)
        and len(pairs) <= cells,
        "invalid stored-z sparse joint storage",
    )
    counts = [0] * cells
    previous = -1
    for pair in pairs:
        require(
            isinstance(pair, list)
            and len(pair) == 2
            and type(pair[0]) is int
            and previous < pair[0] < cells
            and count(pair[1])
            and pair[1] > 0,
            "invalid stored-z sparse index/count/order",
        )
        previous = pair[0]
        counts[pair[0]] = pair[1]
    return dict(
        counts=counts, **{k: joint.get(k) for k in ("samples", "in_range_records", "excluded")}
    )


def validate(partitions, inclusive, axes):
    from collider_secondary_frames import joint_valid, marginal_valid

    require(
        isinstance(partitions, dict)
        and set(partitions) == {"zero_records", "positive", "negative"}
        and count(partitions["zero_records"]),
        "invalid stored-z partitions",
    )
    zero = partitions["zero_records"]
    n = inclusive["nrecords"]
    groups = [partitions[k] for k in ("positive", "negative")]
    require(
        all(
            isinstance(g, dict)
            and set(g) == {"nrecords", "joint", "joint_folded", "marginals", "selection"}
            and count(g.get("nrecords"))
            for g in groups
        ),
        "invalid stored-z subset",
    )
    require(sum(g["nrecords"] for g in groups) + zero == n, "stored-z count closure")
    for g in groups:
        gn = g["nrecords"]
        require(
            isinstance(g["marginals"], dict) and set(g["marginals"]) == set(KEYS),
            "stored-z marginal keys",
        )
        for key in KEYS:
            edges = axes["marginal_" + key + "_edges_" + ("GeV" if key == "pT" else "rad")]
            require(
                isinstance(g["marginals"][key], dict)
                and set(g["marginals"][key])
                == {"counts", "samples", "underflow", "overflow", "total_weight", "statistics"},
                "stored-z marginal keys",
            )
            marginal_valid(g["marginals"][key], gn, edges, key)
        s = g["selection"]
        require(
            isinstance(s, dict) and set(s) == {"process_counts", "signed_energy_counts"},
            "stored-z subset selection keys",
        )
        for key in ("process_counts", "signed_energy_counts"):
            require(
                isinstance(s[key], dict)
                and all(count(v) for v in s[key].values())
                and sum(s[key].values()) == gn,
                "stored-z process/sign count closure",
            )
        require(
            set(s["signed_energy_counts"]) == {"positive_energy", "negative_energy"}
            and len(s["process_counts"]) <= 128,
            "stored-z sign/process keys",
        )
        require(
            all(str(int(k)) == k and -999 <= int(k) <= 999 for k in s["process_counts"]),
            "stored-z process identifier",
        )
    for key in ("joint", "joint_folded"):
        all_j = inclusive[key]
        dense = [dense_joint(g[key], len(all_j["counts"])) for g in groups]
        for j, g in zip(dense, groups):
            joint_valid(j, g["nrecords"])
        residual = [
            a - b - c for a, b, c in zip(all_j["counts"], dense[0]["counts"], dense[1]["counts"])
        ]
        ex = {k: all_j["excluded"][k] - sum(j["excluded"][k] for j in dense) for k in EXCLUSIONS}
        # The residual is precisely z==0, including its non-disjoint exclusions.
        joint_valid(
            dict(counts=residual, samples=zero, in_range_records=sum(residual), excluded=ex), zero
        )
    for key in KEYS:
        all_h = inclusive["marginals"][key]
        hs = [g["marginals"][key] for g in groups]
        residual = [a - b - c for a, b, c in zip(all_h["counts"], hs[0]["counts"], hs[1]["counts"])]
        under = all_h["underflow"] - sum(h["underflow"] for h in hs)
        over = all_h["overflow"] - sum(h["overflow"] for h in hs)
        require(
            all(v >= 0 for v in residual)
            and under >= 0
            and over >= 0
            and sum(residual) + under + over == zero,
            "stored-z marginal residual closure",
        )
        stats = all_h["statistics"]
        for h in hs:
            s = h["statistics"]
            if s is not None:
                require(
                    stats is not None
                    and s["min"] >= stats["min"] - 1e-12 * max(1, abs(stats["min"]))
                    and s["max"] <= stats["max"] + 1e-12 * max(1, abs(stats["max"])),
                    "stored-z extrema outside All",
                )
        if zero == 0 and n:
            mean = (
                math.fsum(
                    g["nrecords"] * h["statistics"]["mean"]
                    for g, h in zip(groups, hs)
                    if g["nrecords"]
                )
                / n
            )
            variance = (
                math.fsum(
                    g["nrecords"]
                    * (h["statistics"]["variance"] + (h["statistics"]["mean"] - mean) ** 2)
                    for g, h in zip(groups, hs)
                    if g["nrecords"]
                )
                / n
            )
            require(
                math.isclose(mean, stats["mean"], rel_tol=2e-9, abs_tol=1e-12)
                and math.isclose(variance, stats["variance"], rel_tol=2e-9, abs_tol=1e-12),
                "stored-z moment closure",
            )
            require(
                math.isclose(
                    min(h["statistics"]["min"] for h in hs if h["samples"]),
                    stats["min"],
                    rel_tol=2e-9,
                    abs_tol=1e-12,
                )
                and math.isclose(
                    max(h["statistics"]["max"] for h in hs if h["samples"]),
                    stats["max"],
                    rel_tol=2e-9,
                    abs_tol=1e-12,
                ),
                "stored-z extrema union closure",
            )
    for key in ("process_counts", "signed_energy_counts"):
        all_counts = inclusive["selection"][key]
        keys = set(all_counts) | set(groups[0]["selection"][key]) | set(groups[1]["selection"][key])
        residual = [
            all_counts.get(k, 0) - sum(g["selection"][key].get(k, 0) for g in groups) for k in keys
        ]
        require(
            all(v >= 0 for v in residual) and sum(residual) == zero,
            "stored-z selection residual closure",
        )
    return partitions


def validate_populations(data):
    definition = data.get("position_z_definition")
    if definition is not None:
        definition_valid(definition)
    for pop in data["populations"].values():
        partitions = pop.get("position_z")
        if definition is not None and pop["available"]:
            require(
                partitions is not None,
                "available final population lacks advertised stored-z partitions",
            )
        if partitions is None:
            continue
        require(
            pop["available"] and definition is not None,
            "stored-z data without population/definition",
        )
        validate(partitions, compact_population(pop), population_axes(pop))
