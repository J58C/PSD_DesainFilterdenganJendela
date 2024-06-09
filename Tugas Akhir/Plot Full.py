import numpy as np
import scipy.signal as signal
import librosa
import matplotlib.pyplot as plt
import soundfile as sf
import os

def read_audio(file_path):
    try:
        y, sr = librosa.load(file_path, sr=None)
        print(f"Audio file loaded successfully: {file_path}")
        return y, sr
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        raise
    except Exception as e:
        print(f"Error loading audio file: {file_path}")
        print(e)
        raise

def apply_fir_filter(data, numtaps, cutoff, pass_type, window, fs):
    try:
        if pass_type == 'low':
            fir_coeff = signal.firwin(numtaps, cutoff, window=window, fs=fs)
        elif pass_type == 'band':
            fir_coeff = signal.firwin(numtaps, cutoff, pass_zero=False, window=window, fs=fs)
        elif pass_type == 'high':
            fir_coeff = signal.firwin(numtaps, cutoff, pass_zero=False, window=window, fs=fs)
        else:
            raise ValueError(f"Filter type {pass_type} not supported.")
        filtered_signal = signal.lfilter(fir_coeff, 1.0, data)
        return filtered_signal, fir_coeff
    except Exception as e:
        print(f"Error applying FIR filter: {e}")
        raise

def apply_iir_filter(data, cutoff, pass_type, filter_type, order, fs, rp=None, rs=None):
    try:
        nyquist = 0.5 * fs
        normalized_cutoff = [c / nyquist for c in cutoff] if isinstance(cutoff, list) else cutoff / nyquist

        if filter_type == 'chebyshev':
            b, a = signal.cheby1(order, rp, normalized_cutoff, btype=pass_type)
        elif filter_type == 'butterworth':
            b, a = signal.butter(order, normalized_cutoff, btype=pass_type)
        elif filter_type == 'elliptic':
            b, a = signal.ellip(order, rp, rs, normalized_cutoff, btype=pass_type)
        elif filter_type == 'bessel':
            b, a = signal.bessel(order, normalized_cutoff, btype=pass_type)
        else:
            raise ValueError(f"Filter type {filter_type} not supported.")

        filtered_signal = signal.lfilter(b, a, data)
        return filtered_signal, (b, a)
    except Exception as e:
        print(f"Error applying IIR filter: {e}")
        raise

def save_audio(file_path, signal, sr):
    try:
        output_dir = os.path.dirname(file_path)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        sf.write(file_path, signal, sr)
        print(f"Audio file saved successfully: {file_path}")
    except Exception as e:
        print(f"Error saving audio file: {file_path}")
        print(e)
        raise

def plot_signals(original, filtered, sr, filter_type, window, filter_coeff):
    try:
        plt.figure(figsize=(15, 6))
        plt.subplot(4, 1, 1)
        total_samples = len(original)
        sample_rate = sr
        freq_range = np.linspace(0, 200000, total_samples//2) # Mengatur rentang sumbu x dari 0 Hz hingga 200000 Hz
        plt.plot(freq_range, np.abs(np.fft.fft(original)[:total_samples//2]))
        plt.title('Original Signal')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Amplitude')

        plt.subplot(4, 1, 2)
        plt.plot(filtered)
        plt.title(f'Filtered Signal ({filter_type} Filter, {window.capitalize()} Window)')
        plt.xlabel('Sample')
        plt.ylabel('Amplitude')

        plt.subplot(4, 2, 5)
        if isinstance(filter_coeff, tuple):
            b, a = filter_coeff
            impulse = np.zeros(50)
            impulse[0] = 1
            response = signal.lfilter(b, a, impulse)
            plt.stem(response)
        else:
            plt.stem(filter_coeff)
        plt.title('Filter Impulse Response')
        plt.xlabel('Sample')
        plt.ylabel('Amplitude')

        plt.subplot(4, 2, 6)
        if isinstance(filter_coeff, tuple):
            w, h = signal.freqz(*filter_coeff, worN=8000)
        else:
            w, h = signal.freqz(filter_coeff, worN=8000)
        plt.plot(0.5 * sr * w / np.pi, np.abs(h), 'b')
        plt.title('Filter Frequency Response')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Gain')

        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"Error plotting signals: {e}")
        raise

def main():
    file_path = r'D:\Kampus\Project\Python\Tugas\Tapis\PSD_DesainFilterdenganJendela\Tugas Akhir\Song Full.wav'
    output_folder = r'D:\Kampus\Project\Python\Tugas\Tapis\PSD_DesainFilterdenganJendela\Tugas Akhir\FilteredFull'

    try:
        y, sr = read_audio(file_path)

        numtaps = 101
        low = 1200
        high = 3600
        cutoffs = [low, [low, high], high]
        windows = ['hamming', 'hann', 'boxcar', 'blackman']
        iir_filters = ['chebyshev', 'butterworth', 'elliptic', 'bessel']
        pass_types = ['low', 'band', 'high']

        for pass_type, cutoff in zip(pass_types, cutoffs):
            for window in windows:
                filtered_signal, fir_coeff = apply_fir_filter(y, numtaps, cutoff, pass_type, window, sr)
                output_file = os.path.join(output_folder, f'filtered_{window}_{pass_type}_fir.wav')
                save_audio(output_file, filtered_signal, sr)
                plot_signals(y, filtered_signal, sr, pass_type, window, fir_coeff)
                print(f"Filtered FIR signal saved to: {output_file}")

            for filter_type in iir_filters:
                order = 4
                rp, rs = 1, 40
                filtered_signal, iir_coeff = apply_iir_filter(y, cutoff, pass_type, filter_type, order, sr, rp, rs)
                output_file = os.path.join(output_folder, f'filtered_{filter_type}_{pass_type}_iir.wav')
                save_audio(output_file, filtered_signal, sr)
                plot_signals(y, filtered_signal, sr, pass_type, filter_type, iir_coeff)
                print(f"Filtered IIR signal saved to: {output_file}")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == '__main__':
    main()