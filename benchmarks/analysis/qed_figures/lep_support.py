"""Validate LEP probe identity, canonical admission and stratified estimators.

The two event legs and boundary cohorts remain paired. Initial velocity
records follow the energy/direction mass-shell admission convention.
"""
import math
import numpy as np
ELECTRON_MASS_GEV = 0.51099906e-3
POPULATIONS = [(cohort, role) for cohort in ("inner", "outer") for role in ("electron", "positron")]
INITIAL_COLUMNS = ["label", "source_record", "scheduled_time_counter", "admitted_time_counter", "signed_energy_GeV", "x_nm", "y_nm", "z_nm", "vx", "vy", "vz"]
CURRENT_COLUMNS = ["label", "vx", "vy", "vz", "main_mesh_gathers", "outer_mesh_gathers", "monopole_gathers"]

def admitted_velocity(energy, direction):
    """Predict canonical input admission and its final float beta mirrors."""
    energy = abs(float(np.float32(energy)))
    direction = np.asarray(direction, dtype=np.float32).astype(float)
    if energy <= ELECTRON_MASS_GEV or not np.isfinite(direction).all():
        raise ValueError("Invalid relativistic LEP input")
    length = math.hypot(math.hypot(direction[0], direction[1]), direction[2])
    magnitude = math.sqrt(energy-ELECTRON_MASS_GEV)*math.sqrt(energy+ELECTRON_MASS_GEV)
    return np.asarray([(magnitude*(v/length))/energy for v in direction], dtype=np.float32).astype(float)

def admitted_matches(observed, expected):
    """Allow one float ULP for cross-host double norm evaluation before casting."""
    expected = np.asarray(expected, dtype=np.float32)
    upper = np.nextafter(expected, np.float32(np.inf)).astype(float)
    lower = np.nextafter(expected, np.float32(-np.inf)).astype(float)
    observed = np.asarray(observed, dtype=float)
    return np.isfinite(observed).all() and np.all((observed >= lower) & (observed <= upper))

def integer(value):
    """Reject fractional, nonfinite and boolean identities before indexing."""
    number = float(value)
    if isinstance(value, bool) or not math.isfinite(number) or number != int(number):
        raise ValueError("Invalid integer identity or counter")
    return int(number)

def angle_difference(a, b):
    return math.atan2(math.sin(a - b), math.cos(a - b))

def native_rows(rows, columns):
    """Check complete finite rows and unique labels before building a map."""
    result = {}
    for values in rows:
        if len(values) != len(columns) or not np.isfinite(values).all():
            raise ValueError("Malformed or nonfinite native trajectory row")
        row = dict(zip(columns, values))
        label = integer(row["label"])
        if label <= 0 or label in result:
            raise ValueError("Duplicate or invalid native label")
        if not any(row[k] != 0 for k in ("vx", "vy", "vz")):
            raise ValueError("Zero native direction")
        for key in columns:
            if key.endswith("_gathers") and integer(row[key]) < 0:
                raise ValueError("Negative native field-gather count")
        result[label] = row
    return result

def validate_admitted_azimuth(source_velocity, native_velocity, planned_phi, role, energy):
    """Bound the azimuth error by the actual float component-rounding error.

    A perturbation of transverse length delta to a vector of length r changes
    its angle by at most asin(delta/r). The 32 double-epsilon allowance covers
    the few atan2/trigonometric operations, not additional float/physics error.
    Initial native components must match the canonical on-shell input conversion.
    """
    source = np.asarray(source_velocity, dtype=float)
    admitted = np.asarray(native_velocity, dtype=float)
    rounded = admitted_velocity(energy, source)
    if not admitted_matches(admitted, rounded):
        raise ValueError("Admitted direction differs from canonical input normalization")
    radius = math.hypot(*source[:2])
    if radius == 0 or not np.isfinite(rounded).all():
        raise ValueError("Undefined planned azimuth")
    expected = planned_phi + (math.pi if role == "positron" else 0)
    numerical = 32 * np.finfo(float).eps
    if abs(angle_difference(math.atan2(source[1], source[0]), expected)) > numerical:
        raise ValueError("Input direction disagrees with planned azimuth")
    relative = float(np.linalg.norm(admitted[:2] - source[:2])) / radius
    if relative >= 1:
        raise ValueError("Float input does not resolve transverse direction")
    bound = math.asin(relative) + numerical
    actual = math.atan2(admitted[1], admitted[0])
    if abs(angle_difference(actual, expected)) > bound:
        raise ValueError("Admitted azimuth exceeds float-rounding bound")
    return actual, bound

def validate_probe_records(raw, identity, sampling, events, probe_rows, case):
    """Join trusted frozen metadata to checked native states and derived rows.

    This is restricted to the prepared pure-imported LEP schema. It does not
    infer event identity, species or pairing from output order or azimuth.
    """
    count = integer(case["metadata"]["vertices_per_boundary"])
    generation = identity["generation"]
    nz, nt = integer(generation["n_z"]), integer(generation["n_t"])
    if nz < 1 or nt < 1 or 4 * count > 8192:
        raise ValueError("Invalid prepared record/grid bounds")
    if (raw.get("schema_version") != "feebl.collider.imported-trajectories/1"
            or raw.get("available") is not True
            or raw.get("phase") != "post-diagonal-tracker-state"
            or raw.get("coordinate_frame") != "native_pair_tracking"
            or raw.get("source_file") != "event.ini"
            or raw.get("initial_columns") != INITIAL_COLUMNS
            or raw.get("current_columns") != CURRENT_COLUMNS):
        raise ValueError("Unexpected imported trajectory schema/columns/frame")
    if (identity.get("schema_version") != "feebl.collider.manufactured-events/1"
            or identity.get("coordinate_frame") != "native_pair_tracking"):
        raise ValueError("Unexpected manufactured identity schema/frame")
    initial = native_rows(raw["initial_records"], INITIAL_COLUMNS)
    final = native_rows(raw["final"]["records"], CURRENT_COLUMNS)
    if len(initial) != 4 * count or initial.keys() != final.keys() or len(events) != 4 * count:
        raise ValueError("Incomplete native/input identity population")
    event_map, source_map, sample_map = {}, {}, {}
    for event in identity["events"]:
        event_id = integer(event["event_id"])
        if event_id in event_map:
            raise ValueError("Duplicate manufactured event identity")
        roles = {}
        for leg in event["particles"]:
            role, source = leg["role"], integer(leg["source_record"])
            if (role not in ("electron", "positron") or role in roles
                    or integer(leg["pdg_id"]) != (11 if role == "electron" else -11)
                    or source in source_map or source < 1 or source > len(events)):
                raise ValueError("Duplicate or inconsistent source/role/PDG identity")
            roles[role] = source
            source_map[source] = (event_id, role)
        if set(roles) != {"electron", "positron"}:
            raise ValueError("Incomplete event species")
        event_map[event_id] = roles
    for row in sampling:
        event_id = integer(row["event_id"])
        if event_id in sample_map:
            raise ValueError("Duplicate sampling event identity")
        sample, cohort = integer(row["sample"]), row["cohort"]
        if (not 0 <= sample < count or cohort not in ("inner", "outer")
                or event_id != sample + (count if cohort == "outer" else 0)):
            raise ValueError("Sampling event/cohort identity mismatch")
        if not np.isfinite([float(row[k]) for k in ("phi_rad", "t_quantized_s")]).all():
            raise ValueError("Nonfinite sampling coordinates")
        if event_id not in event_map or any(
                integer(row[role + "_source_record"]) != source
                for role, source in event_map[event_id].items()):
            raise ValueError("Sampling source identity differs from event identity")
        sample_map[event_id] = row
    if set(sample_map) != set(range(2 * count)) or len(source_map) != 4 * count:
        raise ValueError("Incomplete sampling/source identities")
    derived = {}
    for row in probe_rows:
        key = (integer(row["event_id"]), row["role"])
        if key in derived:
            raise ValueError("Duplicate derived probe identity")
        derived[key] = row
    if set(derived) != {(eid, role) for eid in event_map for role in event_map[eid]}:
        raise ValueError("Incomplete derived probe identities")
    overlap = [min(d + 1, 2 * nz - 1 - d, nz) for d in range(2 * nz - 1)]
    starts = np.r_[0, np.cumsum(overlap)[:-1] * nt]
    admission, seen_sources, checked = {}, set(), []
    for label, first in initial.items():
        source = integer(first["source_record"])
        if source in seen_sources or source not in source_map:
            raise ValueError("Duplicate or unknown admitted source record")
        seen_sources.add(source)
        event_id, role = source_map[source]
        planned = sample_map[event_id]
        diagonal = integer(planned["diagonal"])
        if not 0 <= diagonal < len(starts):
            raise ValueError("Invalid admission diagonal")
        counter = integer(planned["native_time_counter"])
        original = np.asarray(events[source - 1], dtype=float)
        if original.shape != (8,) or not np.isfinite(original).all():
            raise ValueError("Malformed input event row")
        if (counter != starts[diagonal] or counter != integer(original[0])
                or counter != integer(first["scheduled_time_counter"])
                or counter != integer(first["admitted_time_counter"])):
            raise ValueError("Probe admission counter differs from schedule")
        if (original[1] > 0) != (role == "electron"):
            raise ValueError("Input signed energy disagrees with species")
        if not np.array_equal([first[k] for k in INITIAL_COLUMNS[4:8]],
                              original[1:5].astype(np.float32).astype(float)):
            raise ValueError("Initial native state differs from float-converted input")
        phi, phi_bound = validate_admitted_azimuth(
            original[5:], [first[k] for k in ("vx", "vy", "vz")],
            float(planned["phi_rad"]), role, original[1])
        last = final[label]
        before = math.atan2(math.hypot(first["vx"], first["vy"]), abs(first["vz"]))
        after = math.atan2(math.hypot(last["vx"], last["vy"]), abs(last["vz"]))
        expected = dict(field_index=case["metadata"]["field_index"],
                        chunk_index=case["metadata"]["chunk_index"],
                        event_id=event_id, sample=integer(planned["sample"]),
                        role=role, cohort=planned["cohort"], phi_rad=phi,
                        t_s=float(planned["t_quantized_s"]), focus_urad=(before-after)*1e6,
                        **{key: last[key] for key in CURRENT_COLUMNS[4:]})
        supplied = derived[(event_id, role)]
        for key, value in expected.items():
            if isinstance(value, str):
                matches = supplied[key] == value
            else:
                observed = float(supplied[key])
                # Allow double libm rounding when reducing on a different host.
                # The angle bound is propagated before cancellation in focus.
                tolerance = (32 * np.finfo(float).eps if key == "phi_rad" else
                             32 * np.finfo(float).eps * (abs(before) + abs(after)) * 1e6
                             if key == "focus_urad" else 0)
                matches = math.isfinite(observed) and abs(observed - value) <= tolerance
            if not matches:
                raise ValueError("Derived probe differs from native/metadata: " + key)
        checked.append(dict(expected, source_record=source,
                            planned_phi_rad=float(planned["phi_rad"]),
                            native_phi_rounding_bound_rad=phi_bound))
        admission[label] = diagonal
    last_diagonal = 2 * nz - 2
    stride = max(1, (last_diagonal + 62) // 63)
    diagonals = sorted(set(range(0, last_diagonal + 1, stride)) | {last_diagonal})
    if len(raw["frames"]) != len(diagonals) or len(diagonals) > 64:
        raise ValueError("Missing or excess native frames")
    for index, (frame, diagonal) in enumerate(zip(raw["frames"], diagonals)):
        if integer(frame["id"]) != index or integer(frame["diagonal"]) != diagonal:
            raise ValueError("Native frame ID/diagonal mismatch")
        current = native_rows(frame["records"], CURRENT_COLUMNS)
        if set(current) != {label for label, d in admission.items() if d <= diagonal}:
            raise ValueError("Native frame identity/admission population mismatch")
    paired_rows(checked, count)
    return checked

def stratum(phi, count=48):
    """Periodic initial-azimuth stratum, with pi identified with minus pi."""
    phi = np.asarray(phi, dtype=float)
    if not np.isfinite(phi).all():
        raise ValueError("Nonfinite initial azimuth")
    wrapped = (phi + np.pi) % (2 * np.pi)
    return np.minimum((wrapped * count / (2 * np.pi)).astype(int), count - 1)

def uniform_azimuth_weights(phi, count=48):
    """Give each populated stratum weight 1/count, irrespective of occupancy."""
    index = stratum(phi, count)
    occupancy = np.bincount(index, minlength=count)
    if np.any(occupancy == 0):
        raise ValueError("A prescribed azimuth stratum is empty")
    return 1 / (count * occupancy[index]), index, occupancy

def boundary_bias(inner, outer):
    """Leading luminosity boundary estimator in percent, angles in microrad."""
    a, b = 0.0313, 0.0516
    return (
        -200
        / (a ** -2 - b ** -2)
        * (np.asarray(inner) * 1e-6 / a ** 3 - np.asarray(outer) * 1e-6 / b ** 3)
    )

def paired_rows(rows, samples_per_chunk=None):
    """Require one inner/outer electron/positron record for each chunk/sample."""
    joined = {}
    for row in rows:
        key = (int(row["chunk_index"]), int(row["sample"]))
        population = (row["cohort"], row["role"])
        if population not in POPULATIONS or population in joined.setdefault(key, {}):
            raise ValueError("Duplicate or invalid paired probe record")
        if not np.isfinite(
            [float(row[k]) for k in ("phi_rad", "planned_phi_rad", "t_s", "focus_urad")]
        ).all():
            raise ValueError("Nonfinite joined probe")
        if samples_per_chunk is not None:
            expected = key[1] + (samples_per_chunk if row["cohort"] == "outer" else 0)
            if int(row["event_id"]) != expected:
                raise ValueError("Event identity is inconsistent with angular cohort")
        joined[key][population] = row
    for group in joined.values():
        if set(group) != set(POPULATIONS):
            raise ValueError("Incomplete inner/outer or electron/positron pairing")
        reference = group[("inner", "electron")]
        for row in group.values():
            if float(row["t_s"]) != float(reference["t_s"]):
                raise ValueError("Paired probes have different planned encounter times")
            if float(row["planned_phi_rad"]) != float(reference["planned_phi_rad"]):
                raise ValueError("Paired probes have different planned azimuths")
    return joined

def estimate_field(rows, low, high, strata=48):
    """Return matched scalar, azimuthal and conditional-histogram estimates."""
    pairs = paired_rows(rows)
    ordered = [pairs[key] for key in sorted(pairs)]
    phi = np.array([float(pair[("inner", "electron")]["planned_phi_rad"]) for pair in ordered])
    weight, index, occupancy = uniform_azimuth_weights(phi, strata)
    focus = np.array(
        [[float(pair[p]["focus_urad"]) for p in POPULATIONS] for pair in ordered]
    )
    mean = weight @ focus
    raw_mean = focus.mean(axis=0)
    paired_bias = boundary_bias(focus[:, :2], focus[:, 2:])
    scalars = np.r_[mean, weight @ paired_bias]
    raw_scalars = np.r_[raw_mean, paired_bias.mean(axis=0)]
    angular = np.array([focus[index == k, 0].mean() for k in range(strata)])
    low, high = np.asarray(low), np.asarray(high)
    if np.any(high <= low) or np.any(low[1:] < high[:-1] - 1e-10):
        raise ValueError("Invalid or overlapping histogram supports")
    selected = (focus[:, 0, None] >= low) & (focus[:, 0, None] < high)
    mass = weight @ selected
    retained = mass.sum()
    if retained <= 0:
        raise ValueError("No probes in the fixed histogram support")
    return {
        "vector": np.r_[angular, mass / retained, scalars],
        "raw_scalars": raw_scalars,
        "histogram_counts": selected.sum(axis=0),
        "stratum_counts": occupancy,
        "retained_probability": float(retained),
        "excluded_records": int(np.count_nonzero(~selected.any(axis=1))),
        "paired_vertices": len(ordered),
    }

def pair_zero_control(on_rows, zero_rows):
    """Join only the matching field-zero controls; do not subtract from others."""
    on, zero = paired_rows(on_rows), paired_rows(zero_rows)
    if on.keys() != zero.keys():
        raise ValueError("Zero-field control has different paired vertices")
    difference = []
    for key in sorted(on):
        for population in POPULATIONS:
            a, b = on[key][population], zero[key][population]
            if any(float(a[key]) != float(b[key]) for key in
                   ("planned_phi_rad", "phi_rad", "t_s")):
                raise ValueError("Zero-field control is not the same probe input")
            difference.append(
                dict(a, focus_urad=float(a["focus_urad"]) - float(b["focus_urad"]))
            )
    return difference
