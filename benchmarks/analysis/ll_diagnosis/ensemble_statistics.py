"""Whole-collision ratios and paired changes, with no independent-row assumption."""
import numpy as np

from definitions import ARMS, CHANNELS, CONTRASTS


def moments(vectors):
    """Return the ensemble mean and its covariance, rather than run scatter."""
    values = np.asarray(vectors, dtype=float)
    if values.ndim != 2 or len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("Finite, complete collision vectors are required")
    mean = values.mean(axis=0)
    centered = values-mean
    return mean, centered.T@centered/(len(values)*(len(values)-1))


def ratio_statistics(vectors, arm_names):
    """Convert counts and inverse-microbarn luminosities to mb cross sections.

    Each arm contributes BW, BH, LL counts followed by ee, eg, ge, gg
    luminosities. Cross-arm covariance is measured from matched seeds.
    """
    values = np.asarray(vectors, dtype=float)
    if values.shape[1] != 7*len(arm_names):
        raise ValueError("Unexpected count/luminosity column layout")
    mean, covariance = moments(values)
    jacobian = np.zeros((3*len(arm_names), len(mean)))
    estimate = []
    for arm in range(len(arm_names)):
        luminosity_index = 7*arm+3
        luminosity = mean[luminosity_index]
        if np.any(values[:, luminosity_index] <= 0):
            raise ValueError("Every collision requires positive ee luminosity")
        for channel in range(3):
            count_index, output_index = 7*arm+channel, 3*arm+channel
            estimate.append(mean[count_index]/luminosity/1000)
            jacobian[output_index, count_index] = 1/(1000*luminosity)
            jacobian[output_index, luminosity_index] = -mean[count_index]/(1000*luminosity**2)
    # Form covariance through the centered observations. This is equivalent
    # to J C J^T while preserving positive semidefiniteness under cancellation.
    transferred = (values-mean)@jacobian.T
    ratio_covariance = transferred.T@transferred/(len(values)*(len(values)-1))
    return np.asarray(estimate), ratio_covariance


def summarize(vectors, arm_names):
    """Keep full delta-method and delete-one-seed covariance for all channels."""
    values = np.asarray(vectors, dtype=float)
    if len(values) != 4 or len(arm_names) != 9:
        raise ValueError("The prescribed study has nine arms and four seeds")
    estimate, covariance = ratio_statistics(values, arm_names)
    omitted, omitted_sem = [], []
    for index in range(len(values)):
        current, current_covariance = ratio_statistics(np.delete(values, index, axis=0), arm_names)
        omitted.append(current)
        omitted_sem.append(np.sqrt(np.diag(current_covariance)))
    omitted = np.asarray(omitted)
    centered = omitted-omitted.mean(axis=0)
    jackknife = (len(values)-1)/len(values)*(centered.T@centered)
    records = []
    labels = {arm: label for arm, label, *_ in ARMS}
    for index, arm in enumerate(arm_names):
        for channel, name in enumerate(CHANNELS):
            position = 3*index+channel
            records.append(dict(arm=arm, label=labels[arm], channel=name,
                cross_section_mb=float(estimate[position]), sem_mb=float(np.sqrt(covariance[position, position])),
                jackknife_sem_mb=float(np.sqrt(jackknife[position, position])),
                leave_one_out_sem_min_mb=float(np.min(omitted_sem, axis=0)[position]),
                leave_one_out_sem_max_mb=float(np.max(omitted_sem, axis=0)[position])))
    differences = []
    for label, arm, baseline, definition in CONTRASTS:
        selection = np.zeros((3, len(estimate)))
        for channel in range(3):
            selection[channel, 3*arm_names.index(arm)+channel] = 1
            selection[channel, 3*arm_names.index(baseline)+channel] = -1
        delta = selection@estimate
        paired_covariance = selection@covariance@selection.T
        paired_jackknife = selection@jackknife@selection.T
        differences.append(dict(label=label, arm=arm, baseline=baseline, definition=definition,
            change_mb=delta.tolist(), sem_mb=np.sqrt(np.maximum(0, np.diag(paired_covariance))).tolist(),
            covariance_mb2=paired_covariance.tolist(), jackknife_covariance_mb2=paired_jackknife.tolist()))
    return dict(arms=arm_names, channels=list(CHANNELS), records=records, changes=differences,
        mean_mb=estimate.tolist(), covariance_of_mean_mb2=covariance.tolist(),
        jackknife_covariance_mb2=jackknife.tolist(), leave_one_seed_out_mb=omitted.tolist(),
        leave_one_seed_out_sem_mb=np.asarray(omitted_sem).tolist(),
        independent_collisions_per_arm=4, variance_degrees_of_freedom=3, covariance_rank_upper_bound=3,
        estimator="Ratio of summed admitted daughter counts to summed inclusive electron-positron luminosity",
        qualification="Four-seed sampling SEMs only. Cross-arm covariance is measured; no independent-row Poisson term or calibrated global significance is inferred.")
