import numpy as np
import pywt
import matplotlib.pyplot as plt
from scipy.signal import argrelmax
from scipy.signal import find_peaks
import neurokit2 as nk
import pandas as pd

def combine_12_leads(weights, signal):
    """
    Combine 12-lead signals into a single combined signal based on specified weights for each lead.

    :param weights: Dictionary containing weights for each lead.
    :param signal: 3D array representing the 12-lead signal.
    :return: Combined weighted signal.
    """
    # Calculate the total weight
    total_weight = sum(weights.values())

    # Initialize combined weighted signal with proper shape
    combined_weighted_signal = np.zeros((signal.shape[0], signal.shape[1], 1))

    # Calculate the weighted sum of signals across leads using mapping
    for i in range(signal.shape[0]):
        for lead, weight in weights.items():
            # Perform element-wise multiplication and addition to calculate combined signal
            combined_weighted_signal[i] += (signal[i, :, lead] * weight).reshape(-1, 1)

    # Normalize by dividing by total weight
    combined_weighted_signal /= total_weight

    print("New shape:", combined_weighted_signal.shape)
    return combined_weighted_signal


"""
####################################################################################
"""


def plot_wavelet_analysis(ECG, wavelet, level=3):
    """
    Takes the ECG signal and decomposes it into its CD and CA compositions
    in the case level=3, we have CA3,CD3,CD2,CD1. change them if you chose a different level of decomposition
    :param ECG: 1D array
    :param wavelet: for ECG sym4 is usually chosen.
    :param level: usually 3 or 4 is chosen.
    :return: plots CD and CA compositions.
    """
    coefs = pywt.swt(ECG, wavelet=wavelet, level=level, axis=0, trim_approx=True)
    CA3, CD3, CD2, CD1 = coefs
    coef_list = [CA3, CD3, CD2, CD1]
    n = 0
    for coef in coef_list:
        plt.figure(figsize=(30, 5))
        plt.plot(coef)
        plt.xlabel('Amplitude')
        plt.ylabel('Time/Samples')
        plt.title(f'Coefficient{n}')
        plt.show()
        n += 1


"""
####################################################################################
"""


def wavelet_analysis(ECG, wavelet, level=3):
    """
    Based on the ECG signal, wavelet form and level of decompostion, reconstrucs the signal
    :param ECG: 1D array
    :param wavelet: Usually sym4 is chosen.
    :param level: change this according to your needs
    :return: reconstructed signal
    """
    coefs = pywt.swt(ECG, wavelet=wavelet, level=level, axis=0, trim_approx=True)

    # I chose not to consider the coefficients below. you can change this according to your needs

    coefs[-1] = np.zeros_like(coefs[-1])
    coefs[0] = np.zeros_like(coefs[0])

    signal_rec = pywt.iswt(coefs, wavelet=wavelet, axis=0)

    return signal_rec


"""
####################################################################################
"""


def R_finder(signals):
    """
    Based on the reconstructed signa, finds the R peaks of the ECG signal
    :param signals: reconstructed signal/ The original sigan could also be used.
    :return: R peak indexes and amplitudes
    """

    signals_1d = np.squeeze(signals)
    y_prime = np.abs(signals_1d) ** 2
    average = y_prime.mean()
    R_peaks, properties = find_peaks(signals_1d, height=9 * average, distance=75)
    peak_indices = argrelmax(properties['peak_heights'])
    R_peaks = R_peaks[peak_indices]
    peak_values = properties['peak_heights'][peak_indices]

    return R_peaks, peak_values


"""
####################################################################################
"""


def R_plot(signals):
    """
    Plots the R peaks.
    :param signals: 1D array of a signal
    :return: plots the signal and its identified R peaks
    """
    signals_1d = np.squeeze(signals)
    y_prime = np.abs(signals_1d) ** 2
    average = y_prime.mean()

    # Height and distance values can be changed based on your needs
    R_peaks, properties = find_peaks(signals_1d, height=9 * average, distance=75)
    peak_indices = argrelmax(properties['peak_heights'])
    peak_values = properties['peak_heights'][peak_indices]
    plt.figure(figsize=(20, 10))
    plt.plot(R_peaks, properties['peak_heights'], 'r', label='Detected QRS Complexes')
    plt.scatter(R_peaks[peak_indices], peak_values, label='Peaks')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.title('ECG Signal with Detected QRS Complexes')
    plt.legend()


"""
####################################################################################
"""


def PQRS_extraction(V3,aVF,sampling_rate=500):
    """
    works better on V3 Lead for QRS extraction.
    P wave should be extracted from aVF
    :param V3: Signal array from V3 Lead
    :param aVF: signal array from aVF lead
    :param sampling_rate: based your own data
    :return: PQRS indexes and amplitudes based on the ECG provided
    """
    import neurokit2 as nk
    try:
        # Assuming combined_weighted_signal[2000] contains your ECG signal data
        ecg=wavelet_analysis(V3,'sym4')
        avf_ecg=wavelet_analysis(aVF,'sym4')
        # Ensure the ECG signal data is in the correct format (1-dimensional array)
        ecg_signal = np.squeeze(ecg)
        evf_signal= np.squeeze(avf_ecg)

        ecg_info = nk.ecg_process(ecg_signal, sampling_rate=sampling_rate)
        evf_info= nk.ecg_process(evf_signal, sampling_rate=sampling_rate)
        # Get the R-peaks indices
        r_peaks = ecg_info[1]["ECG_R_Peaks"]

        r_amplitude=ecg[r_peaks]

        p_peaks = evf_info[1]["ECG_P_Peaks"]
        q_peaks = ecg_info[1]["ECG_Q_Peaks"]
        s_peaks = ecg_info[1]["ECG_S_Peaks"]

        valid_p_peaks = [peak for peak in p_peaks if not np.isnan(peak)]
        valid_q_peaks = [peak for peak in q_peaks if not np.isnan(peak)]
        valid_s_peaks = [peak for peak in s_peaks if not np.isnan(peak)]

        # You can also get the amplitudes of the peaks
        p_amplitudes = avf_ecg[valid_p_peaks]
        q_amplitudes = ecg[valid_q_peaks]
        s_amplitudes = ecg[valid_s_peaks]

        return r_peaks, r_amplitude,valid_p_peaks,p_amplitudes,valid_q_peaks,q_amplitudes,valid_s_peaks,s_amplitudes
    except Exception as e:
        print("Error:", e)
        # Return NaN or 0 for all durations if an error occurs
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

"""
####################################################################################
"""


def HR_counter(ECG,sample_rate=500):
    """
    Counts the heart rate based on the data provided. works better on V3 Lead
    :param ECG: 1D signal array
    :param sample_rate: based on your own data
    :return: rounded Heart Rate
    """
    import neurokit2 as nk
    try:
        # Assuming that youre using sym4 wavelet
        ecg=wavelet_analysis(ECG,'sym4')

        # Ensure the ECG signal data is in the correct format (1-dimensional array)
        ecg_signal = np.squeeze(ecg)

        ecg_info = nk.ecg_process(ecg_signal, sampling_rate=sample_rate)

        # Get the R-peaks indices
        r_peaks = ecg_info[1]["ECG_R_Peaks"]
        RR_intervals = np.mean(np.diff(r_peaks)/sample_rate)
        heart_rate = 60 / RR_intervals  # Convert to beats per minute (BPM)
        return round(heart_rate,0)
    except Exception as e:
            print("Error:", e)
            # Return NaN or 0 for all durations if an error occurs
            return np.nan


"""
####################################################################################
"""


def mean_PQRS_amplitude(V3,aVF,sampling_rate=500):
    """
    based on the signal, extracts and calculates the mean of PQRS amplitudes,works better on V3 Lead and aVF
    :param V3: Signal array from V3 lead
    :param aVF: Signal array from aVF lead
    :param sampling_rate: based on your own data
    :return: mean values of PQRS peaks
    """
    # s=wavelet_analysis(ECG,'sym4')
    try:
        r_peaks, r_amplitude,valid_p_peaks,p_amplitudes,valid_q_peaks,q_amplitudes,valid_s_peaks,s_amplitudes =PQRS_extraction(V3,aVF,sampling_rate=sampling_rate)
        mean_r_peaks=r_amplitude.mean()
        mean_p_peaks=p_amplitudes.mean()
        mean_q_peaks=q_amplitudes.mean()
        mean_s_peaks=s_amplitudes.mean()

        return mean_r_peaks,mean_p_peaks,mean_q_peaks,mean_s_peaks
    except Exception as e:
        print("Error:", e)
        # Return NaN or 0 for all durations if an error occurs
        return np.nan, np.nan, np.nan, np.nan


"""
####################################################################################
"""


def wave_duration(avf,sampling_rate=500):
    """
    calculates the PR segment and Rwave and Pwave durations,aVF lead seems to work best
    :param aVF: 1D signal array
    :param sampling_rate: based on your own data
    :return: wave and important intervals duration
    """
    import neurokit2 as nk
        # ecg_cleaned = np.nan_to_num(avf, nan=0)
    try:
        # Assuming combined_weighted_signal[2000] contains your ECG signal data
        ecg=wavelet_analysis(avf,'sym4')

        # Ensure the ECG signal data is in the correct format (1-dimensional array)
        ecg_signal = np.squeeze(ecg)

        ecg_info = nk.ecg_process(ecg_signal, sampling_rate=sampling_rate)
        # Get the wave onset and offsets
        r_onset = ecg_info[1]["ECG_Q_Peaks"]
        r_offset=ecg_info[1]["ECG_S_Peaks"]
        p_onset = ecg_info[1]["ECG_P_Onsets"]
        p_offset=ecg_info[1]["ECG_P_Offsets"]
        t_offset=ecg_info[1]["ECG_T_Offsets"]
        t_onset=ecg_info[1]["ECG_T_Onsets"]

        VALID_r_onset=np.array(r_onset)
        VALID_r_offset=np.array(r_offset)

        # valid_r_onsets = [onset for onset in r_onset if not np.isnan(onset)]
        # valid_r_offsets = [offset for offset in r_offset if not np.isnan(offset)]

        VALID_p_onset=np.array(p_onset)
        VALID_p_offset=np.array(p_offset)

        VALID_t_offset=np.array(t_offset)
        VALID_t_onset=np.array(t_onset)



        pwaves=VALID_p_offset-VALID_p_onset
        P_waves=[wave for wave in pwaves if not np.isnan(wave)]

        twave=VALID_t_offset-VALID_t_onset
        T_wave=[wave for wave in twave if not np.isnan(wave)]

        rwaves=VALID_r_offset-VALID_r_onset
        R_waves=[wave for wave in rwaves if not np.isnan(wave)]


        mean_rwave_duration=np.mean(R_waves)/sampling_rate # in seconds
        mean_pwave_duration=np.mean(P_waves)/sampling_rate # in seconds
        mean_twave_duration=np.mean(T_wave)/sampling_rate # in seconds

        # Convert lists to numpy arrays to handle nan values
        ECG_P_onset_arr = np.array(p_onset)
        ECG_Q_Peaks_arr = np.array(r_onset)
        Difference = ECG_Q_Peaks_arr - ECG_P_onset_arr

        ECG_S_offset=np.array(r_offset)
        ECG_T_onset =np.array(t_onset)
        Diff=ECG_T_onset-ECG_S_offset

        valid_PR_segment=[segment for segment in Difference if not np.isnan(segment)]
        mean_PR_segment=np.mean(valid_PR_segment)/sampling_rate

        valid_ST_segment=[segment for segment in Diff if not np.isnan(segment)]
        mean_ST_segment=np.mean(valid_ST_segment)/sampling_rate

        return round(mean_rwave_duration,2), round(mean_pwave_duration,2),round(mean_twave_duration,2),round(mean_PR_segment,3),round(mean_ST_segment,3)
    except Exception as e:
        print("Error:", e)
        # Return NaN or 0 for all durations if an error occurs
        return np.nan, np.nan, np.nan, np.nan, np.nan


"""
####################################################################################
"""


def RR_ratio(ECG,sampling_rate=500):
    """
    calculates the RR ratio which is: RR(i)/RR(i+1). use V3 lead
    :param ECG:
    :param sampling_rate:
    :return: rounded RR_ratio to 2 decimals
    """
    import neurokit2 as nk
    try:
        # Assuming combined_weighted_signal[2000] contains your ECG signal data
        ecg=wavelet_analysis(ECG,'sym4')

        # Ensure the ECG signal data is in the correct format (1-dimensional array)
        ecg_signal = np.squeeze(ecg)

        ecg_info = nk.ecg_process(ecg_signal, sampling_rate=sampling_rate)

        # Get the R-peaks indices
        r_peaks = ecg_info[1]["ECG_R_Peaks"]
        RR_intervals = np.diff(r_peaks)/sampling_rate
        rr_ratios = []
        for i in range(len(RR_intervals) - 1):
            rr_ratio = RR_intervals[i] / RR_intervals[i + 1]
            rr_ratios.append(rr_ratio)

        RR_R = np.mean(rr_ratios)

        return round(RR_R,2)
    except Exception as e:
        print("Error:", e)
        # Return NaN or 0 for all durations if an error occurs
        return np.nan


"""
####################################################################################
"""

def calculate_axis_deviation(ecg_lead_III, ecg_lead_aVL, ecg_lead_aVF):
    """
    Calculate the electrical axis deviation from an ECG.

    Args:
    ecg_lead_III (numpy array): ECG signal from lead III
    ecg_lead_aVL (numpy array): ECG signal from lead aVL
    ecg_lead_aVF (numpy array): ECG signal from lead aVF

    Returns:
    axis_deviation (str): Description of the axis deviation (e.g., "Normal", "Left Axis Deviation", "Right Axis Deviation")
    """
    try:
        # Calculate mean QRS amplitudes in leads I, II, and III
        mean_amplitude_III = np.mean(ecg_lead_III)
        mean_amplitude_aVL = np.mean(ecg_lead_aVL)
        mean_amplitude_aVF = np.mean(ecg_lead_aVF)

        # Determine the axis deviation based on net deflection in leads III and aVF
        if mean_amplitude_III > 0 and mean_amplitude_aVF > 0:
            axis_deviation = "Normal or Left Axis Deviation"
        elif mean_amplitude_III < 0 and mean_amplitude_aVF < 0:
            axis_deviation = "Right Axis Deviation"
        else:
            # Look at lead I to differentiate between left and right axis deviation
            if mean_amplitude_aVL > 0:
                axis_deviation = "Left Axis Deviation"
            else:
                axis_deviation = "Right Axis Deviation"

        return axis_deviation
    except Exception as e:
        print("Error:", e)
        # Return NaN or 0 for all durations if an error occurs
        return np.nan
"""
#############################################################################
"""

def extraction(Lead_III,aVL,aVF,V3):
    """

    :param Lead_III, aVL, aVF: The 1D array of 3 important leads
    :return: A data fram of extracted features
    """
    HR_list = []
    RR_list = []
    R_wave_duration_list = []
    P_wave_duration_list = []
    PR_interval_list = []
    R_peak_list = []
    P_peak_list = []
    Q_peak_list = []
    S_peak_list = []
    T_wave_duration_list = []
    ST_interval_list = []
    axis_list = []

    for item in range(20000):
        print(item)

        Lead3=Lead_III[item]
        avl=aVL[item]
        avf=aVF[item]
        v3=V3[item]

        HR=HR_counter(v3)
        RR=RR_ratio(v3)
        R_wave_duration,P_wave_duration,T_wave_duration,PR_interval,ST_interval=wave_duration(avf)
        R_peak,P_peak,Q_peak,S_peak=mean_PQRS_amplitude(v3,avf)
        axis=calculate_axis_deviation(Lead3,avl,avf)

        # Append extracted features to lists
        HR_list.append(HR)
        RR_list.append(RR)
        R_wave_duration_list.append(R_wave_duration)
        P_wave_duration_list.append(P_wave_duration)
        PR_interval_list.append(PR_interval)
        R_peak_list.append(R_peak)
        P_peak_list.append(P_peak)
        Q_peak_list.append(Q_peak)
        S_peak_list.append(S_peak)
        T_wave_duration_list.append(T_wave_duration)
        ST_interval_list.append(ST_interval)
        axis_list.append(axis)
    # Create a DataFrame from the lists
    data = {
        'HR': HR_list,
        'RR': RR_list,
        'R_wave_duration': R_wave_duration_list,
        'P_wave_duration': P_wave_duration_list,
        'T_wave_duration': T_wave_duration_list,
        'PR_interval': PR_interval_list,
        'ST_interval': ST_interval_list,
        'R_peak': R_peak_list,
        'P_peak': P_peak_list,
        'Q_peak': Q_peak_list,
        'S_peak': S_peak_list,
        'Axis': axis_list


    }
    df = pd.DataFrame(data)
    return df


if __name__ == '__main__':
    pass
