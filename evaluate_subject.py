import numpy as np
from scipy.signal import detrend, butter, sosfiltfilt
from pulse_extractor import extract_raw_signal

def parse_ground_truth(gt_path):
    """
    Parses UBFC Dataset 2 ground_truth.txt:
    Line 0: Raw contact PPG wave
    Line 1: Hardware-calculated HR (ignored)
    Line 2: Timestamps in seconds
    """
    with open(gt_path, 'r') as f:
        raw_lines = [line.strip() for line in f.readlines() if line.strip()]
        
    ppg_wave = np.array([float(x) for x in raw_lines[0].split()])
    timestamps = np.array([float(x) for x in raw_lines[2].split()])
    
    # Calculate true contact PPG sampling rate (Hz) from timestamps
    fs_contact = len(timestamps) / (timestamps[-1] - timestamps[0])
    return ppg_wave, timestamps, fs_contact

def calculate_window_bpm(signal_chunk, fs):
    """Filters a slice of data and extracts the dominant frequency in BPM."""
    if len(signal_chunk) < int(fs * 2):
        return 0.0
        
    detrended = detrend(signal_chunk)
    nyquist = 0.5 * fs
    sos = butter(4, [0.7 / nyquist, 3.0 / nyquist], btype='bandpass', output='sos')
    filtered = sosfiltfilt(sos, detrended)
    
    fft_out = np.abs(np.fft.rfft(filtered))
    freqs = np.fft.rfftfreq(len(filtered), d=1.0/fs)
    
    # Restrict to human cardiac range (0.7 Hz - 3.0 Hz / 42 - 180 BPM)
    valid_idx = np.where((freqs >= 0.7) & (freqs <= 3.0))[0]
    if len(valid_idx) == 0:
        return 0.0
        
    dominant_freq = freqs[valid_idx[np.argmax(fft_out[valid_idx])]]
    return dominant_freq * 60.0

def benchmark_subject(video_path, gt_path, window_sec=10, step_sec=1):
    print(f"Reading video: {video_path}")
    raw_video_signal, video_fps = extract_raw_signal(video_path)
    
    print(f"Reading ground truth: {gt_path}")
    contact_ppg, timestamps, contact_fs = parse_ground_truth(gt_path)
    
    # Align the shorter signal to the longer signal
    video_duration = len(raw_video_signal) / video_fps
    contact_duration = timestamps[-1] - timestamps[0]
    total_sec = min(video_duration, contact_duration)
    
    video_bpms = []
    contact_bpms = []
    
    # Slide a window across the aligned recording
    current_time = 0.0
    while current_time + window_sec <= total_sec:
        # 1. Video window bounds
        v_start = int(current_time * video_fps)
        v_end = int((current_time + window_sec) * video_fps)
        v_chunk = raw_video_signal[v_start:v_end]
        
        # 2. Contact PPG window bounds
        c_start = int(current_time * contact_fs)
        c_end = int((current_time + window_sec) * contact_fs)
        c_chunk = contact_ppg[c_start:c_end]
        
        # Compute BPMs for this exact time window
        bpm_pred = calculate_window_bpm(v_chunk, video_fps)
        bpm_true = calculate_window_bpm(c_chunk, contact_fs)
        
        if bpm_pred > 0 and bpm_true > 0:
            video_bpms.append(bpm_pred)
            contact_bpms.append(bpm_true)
            
        current_time += step_sec
        
    video_bpms = np.array(video_bpms)
    contact_bpms = np.array(contact_bpms)
    
    # Step 4 metric: Mean Absolute Error (MAE)
    absolute_errors = np.abs(video_bpms - contact_bpms)
    mae = np.mean(absolute_errors)
    
    print("\n--- BENCHMARK RESULTS ---")
    print(f"Total Windows Evaluated: {len(absolute_errors)}")
    print(f"Mean Absolute Error (MAE): {mae:.2f} BPM")
    print(f"Max Error in a Window:   {np.max(absolute_errors):.2f} BPM")
    
    return mae
if __name__ == "__main__":
    # Update these exact paths to where your UBFC subject files live
    SUBJECT_VIDEO = r"C:\Users\PC\Downloads\subject1-20261017T074140Z-1-001\subject1\vid.avi"
    SUBJECT_GT = r"C:\Users\PC\Downloads\subject1-20261017T074140Z-1-001\subject1\ground_truth.txt"

    benchmark_subject(SUBJECT_VIDEO, SUBJECT_GT)