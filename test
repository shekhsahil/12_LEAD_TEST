import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.signal import (
    butter,
    filtfilt,
    find_peaks,
    iirnotch
)


# ============================================================
#                    CONFIGURATION
# ============================================================

LEAD1_FILE = "lead1_data.csv"
LEAD2_FILE = "lead2_data.csv"

SAMPLE_RATE = 250.0

ROUND_DIGITS = 2


# ============================================================
# ECG FILTER SETTINGS
# ============================================================

# Baseline wander removal
HIGH_PASS_CUTOFF = 0.5

# ECG low-pass cutoff
LOW_PASS_CUTOFF = 40.0

# Power-line interference
NOTCH_FREQUENCY = 50.0
NOTCH_Q = 30.0


# ============================================================
# R-PEAK DETECTION FILTER
# This filter is ONLY used temporarily for detecting R peaks.
# It is NOT used for the actual lead calculations.
# ============================================================

RPEAK_LOW = 5.0
RPEAK_HIGH = 25.0


# ============================================================
# R-PEAK DETECTION
# ============================================================

MIN_RR_SECONDS = 0.35

MIN_RR_SAMPLES = int(
    MIN_RR_SECONDS * SAMPLE_RATE
)


# Maximum synchronization difference
MAX_SYNC_SHIFT_SECONDS = 0.50

MAX_SYNC_SHIFT_SAMPLES = int(
    MAX_SYNC_SHIFT_SECONDS * SAMPLE_RATE
)


# ============================================================
# aVL DISPLAY OPTION
# ============================================================

# Standard mathematical aVL is:
#
# aVL = I - II/2
#
# Set True ONLY if you want the displayed aVL
# polarity reversed so the main spike points upward.

INVERT_AVL_DISPLAY = True


# ============================================================
# READ RAW COLUMN FROM CSV
# ============================================================

def read_raw_csv(filename):

    if not os.path.exists(filename):

        raise FileNotFoundError(
            f"\nERROR: File not found:\n{filename}"
        )

    df = pd.read_csv(
        filename
    )

    # --------------------------------------------------------
    # New CSV format:
    #
    # Raw,Filtered
    #
    # We intentionally use ONLY Raw.
    # --------------------------------------------------------

    if "Raw" in df.columns:

        values = pd.to_numeric(
            df["Raw"],
            errors="coerce"
        )

    else:

        # Fallback for old one-column CSV
        values = pd.to_numeric(
            df.iloc[:, 0],
            errors="coerce"
        )

    values = values.dropna()

    if len(values) == 0:

        raise ValueError(
            f"No valid Raw ECG values found in:\n{filename}"
        )

    return values.to_numpy(
        dtype=float
    )


# ============================================================
# BANDPASS FILTER
# ============================================================

def bandpass_filter(
    signal,
    lowcut,
    highcut,
    fs,
    order=3
):

    nyquist = fs / 2.0

    low = lowcut / nyquist
    high = highcut / nyquist

    b, a = butter(
        order,
        [low, high],
        btype="bandpass"
    )

    return filtfilt(
        b,
        a,
        signal
    )


# ============================================================
# HIGH PASS FILTER
# ============================================================

def highpass_filter(
    signal,
    cutoff,
    fs,
    order=3
):

    nyquist = fs / 2.0

    normalized = cutoff / nyquist

    b, a = butter(
        order,
        normalized,
        btype="highpass"
    )

    return filtfilt(
        b,
        a,
        signal
    )


# ============================================================
# LOW PASS FILTER
# ============================================================

def lowpass_filter(
    signal,
    cutoff,
    fs,
    order=4
):

    nyquist = fs / 2.0

    normalized = cutoff / nyquist

    b, a = butter(
        order,
        normalized,
        btype="lowpass"
    )

    return filtfilt(
        b,
        a,
        signal
    )


# ============================================================
# 50 Hz NOTCH FILTER
# ============================================================

def notch_filter(
    signal,
    frequency,
    fs,
    Q
):

    w0 = frequency / (fs / 2.0)

    b, a = iirnotch(
        w0,
        Q
    )

    return filtfilt(
        b,
        a,
        signal
    )


# ============================================================
# FINAL ECG FILTER
#
# This is applied AFTER raw lead calculations.
# ============================================================

def filter_ecg(signal):

    signal = np.asarray(
        signal,
        dtype=float
    )

    # Remove DC offset
    signal = signal - np.median(
        signal
    )

    # Remove baseline wander
    signal = highpass_filter(
        signal,
        HIGH_PASS_CUTOFF,
        SAMPLE_RATE,
        order=3
    )

    # Remove 50 Hz power-line interference
    signal = notch_filter(
        signal,
        NOTCH_FREQUENCY,
        SAMPLE_RATE,
        NOTCH_Q
    )

    # Remove high-frequency noise
    signal = lowpass_filter(
        signal,
        LOW_PASS_CUTOFF,
        SAMPLE_RATE,
        order=4
    )

    return signal


# ============================================================
# TEMPORARY FILTER FOR R-PEAK DETECTION
#
# This does NOT replace the raw data.
# ============================================================

def prepare_rpeak_signal(signal):

    qrs = bandpass_filter(
        signal,
        RPEAK_LOW,
        RPEAK_HIGH,
        SAMPLE_RATE,
        order=3
    )

    return qrs


# ============================================================
# R-PEAK DETECTION
# ============================================================

def detect_r_peaks(signal):

    qrs = prepare_rpeak_signal(
        signal
    )

    # QRS can be positive or negative.
    # Absolute value allows detection of either polarity.
    envelope = np.abs(
        qrs
    )

    # Smooth QRS envelope
    window = int(
        0.08 * SAMPLE_RATE
    )

    if window < 3:
        window = 3

    kernel = np.ones(
        window
    ) / window

    smooth = np.convolve(
        envelope,
        kernel,
        mode="same"
    )

    median_value = np.median(
        smooth
    )

    std_value = np.std(
        smooth
    )

    # Adaptive threshold
    threshold = (
        median_value +
        1.2 * std_value
    )

    peaks, properties = find_peaks(
        smooth,
        distance=MIN_RR_SAMPLES,
        prominence=max(
            std_value * 0.35,
            1e-6
        ),
        height=threshold
    )

    # If too few peaks, lower threshold
    if len(peaks) < 3:

        threshold = (
            median_value +
            0.7 * std_value
        )

        peaks, properties = find_peaks(
            smooth,
            distance=MIN_RR_SAMPLES,
            prominence=max(
                std_value * 0.20,
                1e-6
            ),
            height=threshold
        )

    return peaks


# ============================================================
# MATCH PEAKS
# ============================================================

def find_peak_matches(
    peaks1,
    peaks2,
    max_difference
):

    if (
        len(peaks1) == 0 or
        len(peaks2) == 0
    ):

        return []

    matches = []

    used = set()

    for p1 in peaks1:

        differences = np.abs(
            peaks2 - p1
        )

        for index in np.argsort(
            differences
        ):

            if index in used:
                continue

            p2 = peaks2[index]

            if abs(
                p2 - p1
            ) <= max_difference:

                matches.append(
                    (
                        p1,
                        p2
                    )
                )

                used.add(
                    index
                )

                break

    return matches


# ============================================================
# CALCULATE SYNCHRONIZATION SHIFT
# ============================================================

def calculate_sync_shift(
    peaks1,
    peaks2
):

    matches = find_peak_matches(
        peaks1,
        peaks2,
        MAX_SYNC_SHIFT_SAMPLES
    )

    if len(matches) == 0:

        return 0, []

    differences = np.array(
        [
            p1 - p2
            for p1, p2 in matches
        ]
    )

    median_difference = np.median(
        differences
    )

    # Remove bad/outlier matches
    good = (
        np.abs(
            differences -
            median_difference
        )
        <= int(
            0.15 *
            SAMPLE_RATE
        )
    )

    if np.sum(good) > 0:

        shift = int(
            np.round(
                np.median(
                    differences[good]
                )
            )
        )

        good_matches = [
            matches[i]
            for i in range(
                len(matches)
            )
            if good[i]
        ]

    else:

        shift = int(
            np.round(
                median_difference
            )
        )

        good_matches = matches

    return (
        shift,
        good_matches
    )


# ============================================================
# ALIGN RAW SIGNALS
#
# IMPORTANT:
# We align the RAW readings.
# We do NOT align the already-filtered column.
# ============================================================

def align_raw_signals(
    lead1,
    lead2,
    shift
):

    if shift > 0:

        # Lead I R peak occurs later than Lead II.
        #
        # Move Lead II to the right.
        #
        # Use common overlapping region.

        lead1_aligned = lead1[
            shift:
        ]

        lead2_aligned = lead2[
            :-shift
        ]

    elif shift < 0:

        amount = abs(
            shift
        )

        # Lead II R peak occurs later than Lead I.
        #
        # Move Lead II to the left.

        lead1_aligned = lead1[
            :-amount
        ]

        lead2_aligned = lead2[
            amount:
        ]

    else:

        lead1_aligned = lead1.copy()

        lead2_aligned = lead2.copy()

    n = min(
        len(lead1_aligned),
        len(lead2_aligned)
    )

    return (
        lead1_aligned[:n],
        lead2_aligned[:n]
    )


# ============================================================
# CALCULATE DERIVED LEADS FROM RAW DATA
# ============================================================

def calculate_derived_leads(
    lead1_raw,
    lead2_raw
):

    # --------------------------------------------------------
    # Lead III
    # --------------------------------------------------------

    lead3_raw = (
        lead2_raw -
        lead1_raw
    )

    # --------------------------------------------------------
    # aVR
    # --------------------------------------------------------

    avr_raw = -(
        lead1_raw +
        lead2_raw
    ) / 2.0

    # --------------------------------------------------------
    # Standard aVL
    # --------------------------------------------------------

    avl_raw = (
        lead1_raw -
        lead2_raw / 2.0
    )

    # --------------------------------------------------------
    # aVF
    # --------------------------------------------------------

    avf_raw = (
        lead2_raw -
        lead1_raw / 2.0
    )

    return (
        lead3_raw,
        avr_raw,
        avl_raw,
        avf_raw
    )


# ============================================================
# START
# ============================================================

print()
print("==========================================================")
print("        RAW ECG CALCULATION + FILTERED GRAPH")
print("==========================================================")

print()
print("[INFO] CSV format expected:")
print()
print("Raw,Filtered")
print("2093.0,2095.62")
print("2104.0,2095.38")
print("2092.0,2091.11")
print()


# ============================================================
# READ RAW VALUES
# ============================================================

print(
    "[INFO] Reading Lead I RAW values..."
)

lead1_raw = read_raw_csv(
    LEAD1_FILE
)

print(
    f"[OK] Lead I RAW samples: "
    f"{len(lead1_raw)}"
)


print(
    "[INFO] Reading Lead II RAW values..."
)

lead2_raw = read_raw_csv(
    LEAD2_FILE
)

print(
    f"[OK] Lead II RAW samples: "
    f"{len(lead2_raw)}"
)


# ============================================================
# SAME INITIAL LENGTH
# ============================================================

initial_n = min(
    len(lead1_raw),
    len(lead2_raw)
)

lead1_raw = lead1_raw[
    :initial_n
]

lead2_raw = lead2_raw[
    :initial_n
]


print()
print(
    f"[INFO] Initial synchronized length: "
    f"{initial_n}"
)

print(
    f"[INFO] Initial duration: "
    f"{initial_n / SAMPLE_RATE:.2f} seconds"
)


# ============================================================
# R-PEAK DETECTION
#
# Detection uses temporary filtered copies.
# Calculation still uses RAW values.
# ============================================================

print()
print(
    "[INFO] Detecting R peaks..."
)

r_peaks_1 = detect_r_peaks(
    lead1_raw
)

r_peaks_2 = detect_r_peaks(
    lead2_raw
)

print(
    f"[OK] Lead I R peaks : "
    f"{len(r_peaks_1)}"
)

print(
    f"[OK] Lead II R peaks: "
    f"{len(r_peaks_2)}"
)


# ============================================================
# SYNCHRONIZATION
# ============================================================

if (
    len(r_peaks_1) < 2 or
    len(r_peaks_2) < 2
):

    print()
    print(
        "[WARNING] Not enough R peaks for "
        "reliable synchronization."
    )

    sync_shift = 0

    matched_peaks = []

else:

    sync_shift, matched_peaks = (
        calculate_sync_shift(
            r_peaks_1,
            r_peaks_2
        )
    )


print()
print("==========================================================")
print("                 R-PEAK SYNCHRONIZATION")
print("==========================================================")

print(
    f"Matched R peaks: "
    f"{len(matched_peaks)}"
)

print(
    f"Calculated shift: "
    f"{sync_shift} samples"
)

print(
    f"Time shift: "
    f"{sync_shift / SAMPLE_RATE * 1000:.2f} ms"
)


# ============================================================
# ALIGN RAW LEAD I AND RAW LEAD II
# ============================================================

lead1_raw_sync, lead2_raw_sync = (
    align_raw_signals(
        lead1_raw,
        lead2_raw,
        sync_shift
    )
)


n = min(
    len(lead1_raw_sync),
    len(lead2_raw_sync)
)

lead1_raw_sync = lead1_raw_sync[
    :n
]

lead2_raw_sync = lead2_raw_sync[
    :n
]


print()
print(
    f"[OK] Final common samples: {n}"
)

print(
    f"[OK] Final duration: "
    f"{n / SAMPLE_RATE:.2f} seconds"
)


# ============================================================
# CALCULATE FROM RAW VALUES
# ============================================================

print()
print(
    "=========================================================="
)

print(
    "        CALCULATING DERIVED LEADS FROM RAW DATA"
)

print(
    "=========================================================="
)


(
    lead3_raw,
    avr_raw,
    avl_raw,
    avf_raw
) = calculate_derived_leads(
    lead1_raw_sync,
    lead2_raw_sync
)


# ============================================================
# STANDARD aVL VALUE
# ============================================================

avl_standard_raw = avl_raw.copy()


# ============================================================
# DISPLAY aVL
# ============================================================

if INVERT_AVL_DISPLAY:

    avl_display_raw = (
        -avl_standard_raw
    )

else:

    avl_display_raw = (
        avl_standard_raw.copy()
    )


# ============================================================
# EINTHOVEN CHECK ON RAW CALCULATION
# ============================================================

einthoven_error = (
    lead1_raw_sync +
    lead3_raw -
    lead2_raw_sync
)

max_einthoven_error = np.max(
    np.abs(
        einthoven_error
    )
)


print()
print(
    "Einthoven relationship:"
)

print(
    "Lead I + Lead III = Lead II"
)

print(
    f"Maximum raw calculation error: "
    f"{max_einthoven_error:.6f}"
)


# ============================================================
# FILTER AFTER CALCULATION
# ============================================================

print()
print(
    "=========================================================="
)

print(
    "              FILTERING AFTER CALCULATION"
)

print(
    "=========================================================="
)


# Lead I
lead1_filtered = filter_ecg(
    lead1_raw_sync
)

# Lead II
lead2_filtered = filter_ecg(
    lead2_raw_sync
)

# Lead III
lead3_filtered = filter_ecg(
    lead3_raw
)

# aVR
avr_filtered = filter_ecg(
    avr_raw
)

# Standard aVL
avl_standard_filtered = filter_ecg(
    avl_standard_raw
)

# Display aVL
if INVERT_AVL_DISPLAY:

    avl_display_filtered = (
        -avl_standard_filtered
    )

else:

    avl_display_filtered = (
        avl_standard_filtered.copy()
    )

# aVF
avf_filtered = filter_ecg(
    avf_raw
)


print(
    "[OK] Baseline wander removed."
)

print(
    "[OK] 50 Hz notch applied."
)

print(
    "[OK] ECG low-pass filtering applied."
)

print(
    "[OK] Filtering complete."
)


# ============================================================
# ROUND VALUES
# ============================================================

lead1_raw_sync = np.round(
    lead1_raw_sync,
    ROUND_DIGITS
)

lead2_raw_sync = np.round(
    lead2_raw_sync,
    ROUND_DIGITS
)

lead3_raw = np.round(
    lead3_raw,
    ROUND_DIGITS
)

avr_raw = np.round(
    avr_raw,
    ROUND_DIGITS
)

avl_standard_raw = np.round(
    avl_standard_raw,
    ROUND_DIGITS
)

avl_display_raw = np.round(
    avl_display_raw,
    ROUND_DIGITS
)

avf_raw = np.round(
    avf_raw,
    ROUND_DIGITS
)


lead1_filtered = np.round(
    lead1_filtered,
    ROUND_DIGITS
)

lead2_filtered = np.round(
    lead2_filtered,
    ROUND_DIGITS
)

lead3_filtered = np.round(
    lead3_filtered,
    ROUND_DIGITS
)

avr_filtered = np.round(
    avr_filtered,
    ROUND_DIGITS
)

avl_standard_filtered = np.round(
    avl_standard_filtered,
    ROUND_DIGITS
)

avl_display_filtered = np.round(
    avl_display_filtered,
    ROUND_DIGITS
)

avf_filtered = np.round(
    avf_filtered,
    ROUND_DIGITS
)


# ============================================================
# TIME AXIS
# ============================================================

time = (
    np.arange(n)
    / SAMPLE_RATE
)


# ============================================================
# PRINT CALCULATION CHECK
# ============================================================

print()
print(
    "=========================================================="
)

print(
    "                  CALCULATION CHECK"
)

print(
    "=========================================================="
)

print()

print(
    "Index       I       II       III       aVR       aVL       aVF"
)

print(
    "-" * 80
)

for i in range(
    min(10, n)
):

    print(
        f"{i:5d} "
        f"{lead1_raw_sync[i]:9.2f} "
        f"{lead2_raw_sync[i]:9.2f} "
        f"{lead3_raw[i]:9.2f} "
        f"{avr_raw[i]:9.2f} "
        f"{avl_standard_raw[i]:9.2f} "
        f"{avf_raw[i]:9.2f}"
    )


# ============================================================
# CREATE OUTPUT CSV
# ============================================================

output_df = pd.DataFrame(
    {
        "Time": np.round(
            time,
            4
        ),

        # ----------------------------------------------------
        # SOURCE RAW LEADS
        # ----------------------------------------------------

        "Lead_I_Raw": lead1_raw_sync,

        "Lead_II_Raw": lead2_raw_sync,

        # ----------------------------------------------------
        # DERIVED RAW LEADS
        # ----------------------------------------------------

        "Lead_III_Raw": lead3_raw,

        "aVR_Raw": avr_raw,

        "aVL_Raw_Standard": avl_standard_raw,

        "aVF_Raw": avf_raw,

        # ----------------------------------------------------
        # FILTERED LEADS
        # ----------------------------------------------------

        "Lead_I_Filtered": lead1_filtered,

        "Lead_II_Filtered": lead2_filtered,

        "Lead_III_Filtered": lead3_filtered,

        "aVR_Filtered": avr_filtered,

        "aVL_Filtered_Standard": avl_standard_filtered,

        "aVF_Filtered": avf_filtered,

        # ----------------------------------------------------
        # DISPLAY aVL
        # ----------------------------------------------------

        "aVL_Filtered_Display": avl_display_filtered
    }
)


output_file = (
    "ECG_raw_calculated_filtered.csv"
)


output_df.to_csv(
    output_file,
    index=False
)


print()
print(
    f"[OK] Saved:"
)

print(
    f"     {output_file}"
)


# ============================================================
# GRAPH
# ============================================================

print()
print(
    "[INFO] Creating ECG graph..."
)


signals = [

    lead1_filtered,

    lead2_filtered,

    lead3_filtered,

    avr_filtered,

    avl_display_filtered,

    avf_filtered
]


titles = [

    "Lead I",

    "Lead II - R ",

    "Lead III = II - I",

    "aVR = -(I + II) / 2",

    "aVL = I - II/2",

    "aVF = II - I/2"
]


# ============================================================
# FIGURE
# ============================================================

fig, axes = plt.subplots(
    6,
    1,
    figsize=(16, 14),
    sharex=True
)


# ============================================================
# PLOT EACH LEAD
# ============================================================

for ax, signal, title in zip(
    axes,
    signals,
    titles
):

    # Robust graph scale
    low = np.percentile(
        signal,
        1
    )

    high = np.percentile(
        signal,
        99
    )

    amplitude = max(
        abs(low),
        abs(high)
    )

    if amplitude <= 0:

        amplitude = 1

    amplitude *= 1.20

    ax.plot(
        time,
        signal,
        linewidth=1.0
    )

    # Zero reference
    ax.axhline(
        0,
        linewidth=0.8,
        alpha=0.5
    )

    ax.set_ylim(
        -amplitude,
        amplitude
    )

    ax.set_title(
        title,
        loc="left",
        fontweight="bold"
    )

    ax.set_ylabel(
        "ADC"
    )

    ax.grid(
        True,
        alpha=0.30
    )


# ============================================================
# MARK R PEAKS
# ============================================================

# Recalculate R peaks on final Lead I filtered signal
# only for visualization.

final_r_peaks = detect_r_peaks(
    lead1_filtered
)


if len(final_r_peaks) > 0:

    valid = (
        final_r_peaks >= 0
    ) & (
        final_r_peaks < n
    )

    axes[0].plot(
        time[final_r_peaks[valid]],
        lead1_filtered[
            final_r_peaks[valid]
        ],
        "o",
        markersize=4
    )


# ============================================================
# X AXIS
# ============================================================

axes[-1].set_xlabel(
    "Time (seconds)"
)


# ============================================================
# MAIN TITLE
# ============================================================

fig.suptitle(
    "6-Lead ECG - Raw Calculation → Filtering → Graph",
    fontsize=16,
    fontweight="bold"
)


# ============================================================
# LAYOUT
# ============================================================

plt.tight_layout(
    rect=[
        0,
        0,
        1,
        0.97
    ]
)


# ============================================================
# SHOW
# ============================================================

plt.show()


# ============================================================
# FINAL MESSAGE
# ============================================================

print()
print(
    "=========================================================="
)

print(
    "                    COMPLETE"
)

print(
    "=========================================================="
)

print()
print(
    "RAW values were used for lead calculations."
)

print(
    "The existing CSV 'Filtered' column was NOT used."
)

print(
    "R-peak detection used a temporary filter only."
)

print(
    "Raw Lead I and Lead II were synchronized before"
)

print(
    "calculating III, aVR, aVL and aVF."
)

print(
    "Filtering was applied AFTER the calculations."
)

print(
    f"Output file: {output_file}"
)

print()
