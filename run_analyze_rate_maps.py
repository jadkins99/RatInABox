from utils import load_rate_maps
from representation_analysis import build_peak_dataset
from plotting import plot_peaks_per_environment

data = load_rate_maps("data")
peaks = build_peak_dataset(data)
plot_peaks_per_environment(peaks)