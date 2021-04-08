import os
import sys 
import glob
import json
import librosa
import argparse
import numpy as np


def min_max_scaling(x):
    """
    Args:
        S: np.array, Spectrogram. Shape=(f, t)
    Returns:
        S: np.array, scalied Spectrogram. Shape=(f, t)
    """
    _max = np.max(x)
    _min = np.min(x)
    
    x = (x - _min + 1e-7)  / (_max - _min)
    
    return x
        
        
def time_average(S):
    """ Summation or Average by time
    Args:
        S : np.array, Spectrogram. Shape=(n_mfcc, time) or (frame_bins, time)
        
    Returns:
        spect : np.array, spectrum. Shape=(n_mfcc) or (frame_bins)
    """
    spectrum = np.mean(S, axis=-1)

    return spectrum


def preprocess(data_list, args):
    example = data_list[0]
    item_name = example['item']
    item_type = example['type']
    print(f">> target : {item_name}_{item_type}")
    
    save_dir_path =  os.path.join(args.main_dir, f'seqlen_{args.seq_len}_mels_{args.n_mels}', args.dataset_name)
    save_path = os.path.join(save_dir_path, item_name)
    os.makedirs(save_path, exist_ok=True)

    for data in data_list:
        # Original
        wav_name = os.path.basename(data['wav'])
        wav_path = os.path.join(args.dataset_path, data['wav']+'.wav')
        sr = data['sr']
        
        wav = librosa.load(wav_path, sr=sr)[0]
        
        n_fft = args.n_fft
        hop_length = args.hop_length
        S = librosa.feature.melspectrogram(y=wav, sr=sr,
                                           n_fft=n_fft,hop_length=hop_length,
                                           n_mels=args.n_mels)
        S_len = S.shape[1]

        """
        Temporal Adaptive Average pooling
        """
        q = int(S_len / args.seq_len)
        r = S_len % args.seq_len
        
        if q != 0:
            margin = (q + 1) * args.seq_len - S_len
            padded_S = np.zeros((S.shape[0], S.shape[1]+margin)).astype(np.float32)
            padded_S[:,:S_len] = S
            S = padded_S
            S_len += margin
        
        kernel_size = int(S_len / args.seq_len)

        spectrum_list = []
        for i in range(args.seq_len):
            kernel_start = i * kernel_size
            kernel_end = kernel_start + kernel_size

            local_S = S[:,kernel_start:kernel_end]
            spectrum = time_average(local_S)
            spectrum_list.append(spectrum)

        sequence = np.stack(spectrum_list, 0) # Shape = (sequence_length, n_dims)
        sequence = min_max_scaling(sequence)

        np.save(f"{save_path}/{wav_name}.npy", sequence)
    
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--main-dir', type=str, default='', help='-')
    parser.add_argument('--dataset-name', type=str, default='', help='-')
    parser.add_argument('--dataset-path', type=str, default='', help='-')
    parser.add_argument('--target-manifest', type=str, default='', help='-')
    parser.add_argument('--seq-len', type=int, default=32, help='-')
    parser.add_argument('--n-mels', type=int, default=80, help='-')
    parser.add_argument('--n-fft', type=int, default=2048, help='-')
    args, unknown = parser.parse_known_args()
    args.win_length = args.n_fft
    args.hop_length = int(args.n_fft / 4)

    with open(args.target_manifest, 'r') as f:
        data_list = json.load(f)
    
    print(">> Preproecssing..")
    preprocess(data_list, args)
    print(">> Done")   
