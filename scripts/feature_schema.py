"""
Shared Gaia feature schema definitions.

Canonical naming:
- 8D: base compact feature set
- 10D: 8D + phot_g_mean_mag + bp_rp
- 16D: 8D + 8 additional Gaia quality/contamination metrics

Backward compatibility:
- "17D" is accepted as an alias for "16D".
"""

from __future__ import annotations

from typing import List


Y_FEATURE_COLS_10D: List[str] = [
    "feat_log10_snr",
    "feat_ruwe",
    "feat_astrometric_excess_noise",
    "feat_parallax_over_error",
    "feat_visibility_periods_used",
    "feat_ipd_frac_multi_peak",
    "feat_c_star",
    "feat_pm_significance",
    "feat_phot_g_mean_mag",
    "feat_bp_rp",
]

Y_FEATURE_COLS_8D: List[str] = Y_FEATURE_COLS_10D[:8]

# 8 net-new features on top of 8D -> canonical "16D".
Y_FEATURE_COLS_16D: List[str] = Y_FEATURE_COLS_8D + [
    "feat_astrometric_excess_noise_sig",
    "feat_ipd_gof_harmonic_amplitude",
    "feat_ipd_gof_harmonic_phase",
    "feat_ipd_frac_odd_win",
    "feat_phot_bp_n_contaminated_transits",
    "feat_phot_bp_n_blended_transits",
    "feat_phot_rp_n_contaminated_transits",
    "feat_phot_rp_n_blended_transits",
]

META_FEATURE_COLS_16D: List[str] = Y_FEATURE_COLS_10D + [
    c for c in Y_FEATURE_COLS_16D if c not in Y_FEATURE_COLS_10D
]

GAIA_COLS_OPTIONAL_EXTENDED: List[str] = [
    "astrometric_excess_noise_sig",
    "ipd_gof_harmonic_amplitude",
    "ipd_gof_harmonic_phase",
    "ipd_frac_odd_win",
    "phot_bp_n_contaminated_transits",
    "phot_bp_n_blended_transits",
    "phot_rp_n_contaminated_transits",
    "phot_rp_n_blended_transits",
]

# Backfill list includes one already-common base metric to repair legacy files.
GAIA_BACKFILL_TARGET_COLS: List[str] = [
    "astrometric_excess_noise_sig",
    "ipd_gof_harmonic_amplitude",
    "ipd_gof_harmonic_phase",
    "ipd_frac_multi_peak",
    "ipd_frac_odd_win",
    "phot_bp_n_contaminated_transits",
    "phot_bp_n_blended_transits",
    "phot_rp_n_contaminated_transits",
    "phot_rp_n_blended_transits",
]


def normalize_feature_set(feature_set: str) -> str:
    fs = str(feature_set).upper()
    if fs == "17D":
        return "16D"
    if fs in ("8D", "10D", "16D"):
        return fs
    raise ValueError("FEATURE_SET must be '8D', '10D', or '16D' (legacy alias: '17D').")


def get_feature_cols(feature_set: str) -> List[str]:
    fs = normalize_feature_set(feature_set)
    if fs == "8D":
        return Y_FEATURE_COLS_8D
    if fs == "10D":
        return Y_FEATURE_COLS_10D
    return Y_FEATURE_COLS_16D


def scaler_stem(feature_set: str) -> str:
    fs = normalize_feature_set(feature_set)
    if fs == "8D":
        return "y_feature_iqr_8d"
    if fs == "10D":
        return "y_feature_iqr"
    return "y_feature_iqr_16d"
