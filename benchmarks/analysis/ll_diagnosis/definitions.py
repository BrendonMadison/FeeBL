"""Declared controls and comparison scopes for the nine-arm LL study."""

ARMS = (
    ("legacy_epa_ring", "Legacy EPA, ring", 0, 1, 2),
    ("finite_mass_epa_ring", "Finite-mass EPA, ring", 1, 1, 2),
    ("finite_mass_epa_profile", "Finite-mass EPA, profile", 1, 3, 2),
    ("exact_ring", "Full QED, ring", 2, 1, 2),
    ("exact_profile", "Full QED, profile", 2, 3, 2),
    ("finite_mass_epa_cain_qmax", "EPA, electron-mass cutoff", 1, 1, 0),
    ("finite_mass_epa_no_beam_size", "EPA, no beam-size suppression", 1, 0, 2),
    ("exact_no_beam_size", "Full QED, no beam-size suppression", 2, 0, 2),
    ("exact_profile_expanded_domain", "Full QED, expanded domain", 2, 3, 2),
)
CHANNELS = ("BW", "BH", "LL")
SEEDS = (73111001, 73111002, 73111003, 73111004)

# Each difference is changed arm minus baseline arm. These are defined before
# examining results; no control or seed is selected for its apparent agreement.
CONTRASTS = (
    ("EPA refinements", "finite_mass_epa_ring", "legacy_epa_ring",
     "Finite-mass flux, alpha-constant adjustment and stable angular sampler together; the Born normalization is already corrected in both arms."),
    ("EPA spatial profile", "finite_mass_epa_profile", "finite_mass_epa_ring",
     "Change the transverse placement prescription at fixed finite-mass EPA treatment."),
    ("Full treatment, ring", "exact_ring", "finite_mass_epa_ring",
     "Change hard amplitudes, proposal support, exact kinematics, virtuality and identical-lepton treatment together; the common placement rule does not imply an identical sampled spatial distribution."),
    ("Full treatment, profile", "exact_profile", "finite_mass_epa_profile",
     "Change the full hard-process and phase-space treatment while retaining the profile placement rule."),
    ("Full-QED spatial profile", "exact_profile", "exact_ring",
     "Change the placement prescription within the full-QED calculation."),
    ("Electron-mass cutoff", "finite_mass_epa_cain_qmax", "finite_mass_epa_ring",
     "Apply Q squared no greater than electron mass squared within EPA; this does not implement the complete CAIN spatial model."),
    ("Remove suppression, EPA", "finite_mass_epa_no_beam_size", "finite_mass_epa_ring",
     "Disable finite transverse beam-size suppression within EPA."),
    ("Remove suppression, full", "exact_no_beam_size", "exact_ring",
     "Disable finite transverse beam-size suppression within full QED."),
    ("Expand domain", "exact_profile_expanded_domain", "exact_profile",
     "Double both transverse extents and cell counts while keeping cell spacing fixed."),
)

# Values are the printed effective daughter cross sections, not divisions of
# separately rounded published counts. Targets belong to their named controls.
REFERENCE_TARGETS = (
    ("legacy_epa_ring", "GUINEA-PIG", "II", (1.05, 37.7, 19.2)),
    ("finite_mass_epa_ring", "GUINEA-PIG", "II", (1.05, 37.7, 19.2)),
    ("finite_mass_epa_cain_qmax", "GUINEA-PIG", "V", (1.09, 35.2, 15.6)),
    ("finite_mass_epa_no_beam_size", "GUINEA-PIG", "III", (1.01, 66.3, 33.9)),
    ("exact_no_beam_size", "BDK", "III", (None, None, 31.8)),
)
