# -*- coding: utf-8 -*-
__author__ = "Lukas Pfeifenberger"


import time
import numpy as np
import argparse
import json
import os
import sys
import pandas as pd
from pystoi import stoi
import matplotlib.pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
sys.path.append(os.path.abspath('../'))
sys.path.append("/gpfs0/bgu-br/users/yahelso/model-based-speech-dereverberation/model-based-speech-dereverberation/BSD-main")

from keras.models import Sequential, Model
from keras.layers import Layer, Dense, Activation, LSTM, Input, Lambda, BatchNormalization, LayerNormalization, Conv1D, Bidirectional
from keras import activations
import keras.backend as K
import tensorflow as tf
from loaders.feature_generator import feature_generator
from utils.mat_helpers import *
from algorithms.audio_processing import *
from utils.keras_helpers import *
from ops.complex_ops import *
from utils.matplotlib_helpers import *

from modules.beamforming_td import beamforming
from modules.identification_td import identification

from classic.wpe_wrapper import ClassicWPE

np.set_printoptions(precision=3, threshold=3, edgeitems=3)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)




#-------------------------------------------------------------------------
#-------------------------------------------------------------------------

class bssd(object):

    def __init__(self, config, set='train'):

        self.config = config
        self.fgen = feature_generator(config, set)
        self.nsrc = config['nsrc']                      # number of concurrent speakers

        self.speakers_configed = config.get("speakers", 20)
        self.speakers = min(self.speakers_configed, self.fgen.nspk)         # self.fgen.nspk - number of speakers in date set
        self.batch_size = config.get('batch_size', 12)
        self.validate_batch_size = config.get("validate_batch_size", self.batch_size)
        self.is_load_weights = config.get("is_load_weights", True)
        self.is_save_weights = config.get("is_save_weights", True)

        os.makedirs(self.config['log_path'], exist_ok=True)

        if self.speakers_configed > self.fgen.nspk:
            print(f"[warn] --speakers clipped from {self.speakers_configed} to {self.fgen.nspk}")
        if self.validate_batch_size < self.batch_size:
            self.validate_batch_size = self.batch_size


        self.filename = os.path.basename(__file__)
        self.name = self.filename[:-3] + '_' + config['rir_type']
        self.creation_date = os.path.getmtime(self.filename)
        self.weights_file = self.config['weights_path'] + self.name + '.h5'
        self.predictions_file = self.config['predictions_path'] + self.name + '.mat'
        self.predictions_file_for_compare = self.config['predictions_for_compare_path'] + self.name + '.mat'

        self.logger = Logger(self.name)

        self.samples = self.fgen.samples                # number of samples per utterance
        self.nmic = self.fgen.nmic                      # number of microphones
        self.ndoa = self.fgen.ndoa                      # number of DOA vectors on the sphere

        self.nbin = 500                                 # latent space H
        self.wlen = 200                                 # convolution kernel filter length
        self.shift = self.wlen//4                       # convolution stride
        self.ndim = 100                                 # embedding dimension E

        self.beamforming = beamforming(self.fgen)
        self.identification = identification(self.fgen)
        self.create_model()

        # for benchmark
        self.wpe_model = ClassicWPE()

        self.si_sdr = []
        self.eer = []
        self.epoch = 0
        self.hist_si_sdr, self.hist_stoi = [], []      # training curves
        self.best_metric = -1e9                        # higher is better
        self.chkpt_path   = os.path.join(self.logdir, "weights.h5")

        data = load_numpy_from_mat(self.predictions_file)
        if data is not None:
            if 'epoch' in data.keys():
                self.epoch = data['epoch']
                self.si_sdr = data['si_sdr']
                self.eer = data['eer']



    #---------------------------------------------------------
    def create_model(self):

        print('*** creating model: %s' % self.name)

        Z = Input(shape=(self.samples, self.nmic), dtype=tf.float32)                # shape = (nbatch, nsamples, nmic)
        R = Input(shape=(self.samples,), dtype=tf.float32)                          # shape = (nbatch, nsamples)
        pid = Input(shape=(1,), dtype=tf.int32)                                     # shape = (nbatch,)
        sid = Input(shape=(1,), dtype=tf.int32)                                     # shape = (nbatch, 1)

        [Py, Y, cost_bf] = self.beamforming.model([Z, R, pid])
        [E, cost_id] = self.identification.model([Py, sid])

        # compile model
        self.model = Model(inputs=[Z, R, pid, sid], outputs=[Y, E])
        self.model.add_loss(cost_bf + 0.01*cost_id)
        self.model.compile(loss=None, optimizer='adam')

        print(self.model.summary())
        try:
            self.model.load_weights(self.weights_file)
        except:
            print('error loading weights file: %s' % self.weights_file)



    #---------------------------------------------------------
    def save_weights(self):
        print(f"need to save weights here from {self.weights_file}")
        self.model.save_weights(self.weights_file)

        return



    #---------------------------------------------------------
    def train(self):

        print('train the model')
        while (self.epoch<self.config['epochs']) and self.check_date():

            sid0 = self.fgen.generate_triplet_indices(speakers=self.speakers, utterances_per_speaker=3)
            z, r, sid, pid = self.fgen.generate_multichannel_mixtures(nsrc=self.nsrc, sid=sid0)
            self.model.fit([z, r, pid[:,0], sid[:,0]], None, batch_size=len(sid0), epochs=1, verbose=1, shuffle=False, callbacks=[self.logger])

            self.epoch += 1
            if self.epoch <= 200:
                save_every = 5  # first epochs
            else:
                save_every = 10  # advanced epochs
            
            if (self.epoch % save_every)==0:
                self.save_weights()
                self.validate()

    #---------------------------------------------------------
    def save_rev_files(self):
        count =  0
        while (count < 3):
            print(f"epcount number {count}")
            sid0 = self.fgen.generate_triplet_indices(speakers=self.speakers, utterances_per_speaker=3)
            z, r, sid, pid = self.fgen.generate_multichannel_mixtures(nsrc=self.nsrc, sid=sid0)  # len(sid) = 1 for test

            count += 1
            



    #---------------------------------------------------------
    def validate(self):

        sid = self.fgen.generate_triplet_indices(speakers=self.fgen.nspk, utterances_per_speaker=3)
        z, r, sid, pid = self.fgen.generate_multichannel_mixtures(nsrc=self.nsrc, sid=sid)
        y, E = self.model.predict([z, r, pid[:,0], sid[:,0]], batch_size=50)

        si_sdr = self.beamforming.si_sdr(r, y)
        far, frr, eer = self.identification.calc_eer(E, sid[:,0])
        print('SI-SDR:', si_sdr)
        print('EER:', eer)
        self.si_sdr = np.append(self.si_sdr, si_sdr)
        self.eer = np.append(self.eer, eer)

        data = {
            'z': z[0,:,0],
            'r': r[0,:],
            'y': y[0,:],
            'E': E,
            'pid': pid,
            'sid': sid,
            'far': far,
            'frr': frr,
            'si_sdr': self.si_sdr,
            'eer': self.eer,
            'epoch': self.epoch,
        }
        save_numpy_to_mat(self.predictions_file, data)


#---------------------------------------------------------
    def validate_with_wpe(self, number_of_estimations=1):

        results_dir = self.config['validation_comparison_results']
        os.makedirs(results_dir, exist_ok=True)

        # Create spectrogram subfolder if needed
        spectrogram_dir = os.path.join(results_dir, "spectrograms")
        os.makedirs(spectrogram_dir, exist_ok=True)

        results = {'method': [], 'si_sdr': [], 'stoi': []}

        for i in range(number_of_estimations):
            print(f"\nEvaluation {i+1}/{number_of_estimations}")

            SINGLE_UTTERANCE = 1
            sid = self.fgen.generate_triplet_indices(speakers=self.fgen.nspk, utterances_per_speaker=SINGLE_UTTERANCE)
            z, r, sid, pid = self.fgen.generate_multichannel_mixtures(nsrc=self.nsrc, sid=sid)

            y_bsd, _ = self.model.predict([z, r, pid[:,0], sid[:,0]], batch_size=1)
            si_sdr_bsd = self.beamforming.si_sdr(r, y_bsd)
            stoi_bsd = stoi(r[0], y_bsd[0], self.config['fs'], extended=False)

            results['method'].append('BSD')
            results['si_sdr'].append(si_sdr_bsd)
            results['stoi'].append(stoi_bsd)

            y_wpe = self.wpe_model.dereverb_batch(z)
            si_sdr_wpe = self.beamforming.si_sdr(r, y_wpe)
            stoi_wpe = stoi(r[0], y_wpe[0], self.config['fs'], extended=False)

            results['method'].append('WPE')
            results['si_sdr'].append(si_sdr_wpe)
            results['stoi'].append(stoi_wpe)

            print(f"BSD - SI-SDR: {si_sdr_bsd:.2f} dB | STOI: {stoi_bsd:.3f}")
            print(f"WPE - SI-SDR: {si_sdr_wpe:.2f} dB | STOI: {stoi_wpe:.3f}")

            # Save spectrogram comparison for this estimation
            spectrogram_path = os.path.join(
                spectrogram_dir,
                f"{self.name}_dereverb_comparison_estimation_{i+1}.png"
            )

            self.plot_spectograms_comparison_bsd_wpe(
                z=z[0,:,0],
                r=r[0,:],
                y_bsd=y_bsd[0,:],
                y_wpe=y_wpe[0,:],
                save_path=spectrogram_path
            )

        df = pd.DataFrame(results)
        csv_path = os.path.join(results_dir, self.name + '_bsd_vs_wpe_metrics.csv')
        df.to_csv(csv_path, index=False)
        print(f"\nMetrics saved to {csv_path}")

        self.plot_bsd_vs_wpe_metrics(csv_path, results_dir)

    
    #---------------------------------------------------------
    def plot_bsd_vs_wpe_metrics(self, csv_path, results_dir):

        df = pd.read_csv(csv_path)

        estimations = list(range(1, (len(df)//2) + 1))
        bsd_si_sdr = df[df['method'] == 'BSD']['si_sdr'].values
        wpe_si_sdr = df[df['method'] == 'WPE']['si_sdr'].values
        bsd_stoi = df[df['method'] == 'BSD']['stoi'].values
        wpe_stoi = df[df['method'] == 'WPE']['stoi'].values

        # Colors
        color_bsd = 'blue'
        color_wpe = 'orange'

        # Plot SI-SDR
        plt.figure(figsize=(10, 6))
        plt.plot(estimations, bsd_si_sdr, marker='o', color=color_bsd, label='BSD')
        plt.plot(estimations, wpe_si_sdr, marker='s', color=color_wpe, label='WPE')
        plt.title('SI-SDR Comparison: BSD vs WPE')
        plt.xlabel('Estimation #')
        plt.ylabel('SI-SDR (dB)')
        plt.grid(True)
        plt.legend()
        si_sdr_plot_path = os.path.join(results_dir, self.name + '_si_sdr_comparison.png')
        plt.savefig(si_sdr_plot_path)
        plt.close()
        print(f"SI-SDR plot saved to {si_sdr_plot_path}")

        # Plot STOI
        plt.figure(figsize=(10, 6))
        plt.plot(estimations, bsd_stoi, marker='o', color=color_bsd, label='BSD')
        plt.plot(estimations, wpe_stoi, marker='s', color=color_wpe, label='WPE')
        plt.title('STOI Comparison: BSD vs WPE')
        plt.xlabel('Estimation #')
        plt.ylabel('STOI (unitless, 0-1)')
        plt.ylim(0, 1)
        plt.grid(True)
        plt.legend()
        stoi_plot_path = os.path.join(results_dir, self.name + '_stoi_comparison.png')
        plt.savefig(stoi_plot_path)
        plt.close()
        print(f"STOI plot saved to {stoi_plot_path}")

    #---------------------------------------------------------
    def plot_spectograms_comparison_bsd_wpe(self, z, r, y_bsd, y_wpe, save_path):
        """
        Generates and saves a spectrogram comparison plot for a single estimation.
        """

        # Normalize for visualization
        z = z / np.max(np.abs(z))
        r = r / np.max(np.abs(r))
        y_bsd = y_bsd / np.max(np.abs(y_bsd))
        y_wpe = y_wpe / np.max(np.abs(y_wpe))

        # Compute log magnitude STFTs
        def log_spec(signal):
            return 20 * np.log10(np.abs(mstft(signal)) + 1e-6)

        specs = [
            log_spec(z),
            log_spec(r),
            log_spec(y_bsd),
            log_spec(y_wpe)
        ]

        legend = ['Mixture $z(t)$', 'Clean $r(t)$', 'BSD output', 'WPE output']
        draw_subpcolor(specs, legend, save_path)

        print(f"Spectrogram saved to {save_path}")

    #---------------------------------------------------------
    def plot(self):

        z, r, sid, pid = self.fgen.generate_multichannel_mixtures(nsrc=self.nsrc)
        
        data = []
        z0 = z[0,:,0]/np.amax(np.abs(z[0,:,0]))
        data.append( 20*np.log10(np.abs(mstft(z0))) )

        for c in range(self.nsrc):
            y, E = self.model.predict([z, r, pid[:,c], sid[:,c]])
            y0 = y[0,:]/np.amax(np.abs(y[0,:]))
            data.append( 20*np.log10(np.abs(mstft(y0))) )

        legend = ['mixture z(t)', 'extracted speaker y1(t)', 'extracted speaker y2(t)', 'extracted speaker y3(t)', 'extracted speaker y4(t)']
        filename = self.config['predictions_path'] + self.name + '_spectrogram.png'
        draw_subpcolor(data, legend, filename)


    def plot_dereverb_comparison(self):

        # Load saved .mat file
        from scipy.io import loadmat
        data = loadmat(self.predictions_file_for_compare)

        z = data['z'].squeeze()
        r = data['r'].squeeze()
        y_bsd = data['y_bsd'].squeeze()
        y_wpe = data['y_wpe'].squeeze()

        # Normalize for visualization
        z = z / np.max(np.abs(z))
        r = r / np.max(np.abs(r))
        y_bsd = y_bsd / np.max(np.abs(y_bsd))
        y_wpe = y_wpe / np.max(np.abs(y_wpe))

        # Compute log magnitude STFTs
        def log_spec(signal):
            return 20 * np.log10(np.abs(mstft(signal)) + 1e-6)

        specs = [
            log_spec(z),
            log_spec(r),
            log_spec(y_bsd),
            log_spec(y_wpe)
        ]

        legend = ['Mixture $z(t)$', 'Clean $r(t)$', 'BSD output', 'WPE output']
        filename = self.config['predictions_for_compare_path'] + self.name + '_dereverb_comparison.png'
        draw_subpcolor(specs, legend, filename)


    #---------------------------------------------------------
    def check_date(self):

        if (self.creation_date == os.path.getmtime(self.filename)):
            return True
        else:
            return False





#---------------------------------------------------------
#---------------------------------------------------------
if __name__ == "__main__":


    # parse command line args
    parser = argparse.ArgumentParser(description='speaker separation')
    parser.add_argument('--config_file', help='name of json configuration file', default='shoebox_c2.json')
    #parser.add_argument('--mode', help='mode: [train, valid, plot]', nargs='?', choices=('train', 'valid', 'plot'), default='train')
    parser.add_argument('--mode', help='mode: [train, valid, plot,save_rev_files]', nargs='?', choices=('train', 'valid','valid_with_wpe', 'plot','plot_dereverb_comparison','save_rev_files'), default='train')
    args = parser.parse_args()


    # load config file
    try:
        print('*** loading config file: %s' % args.config_file )
        with open(args.config_file, 'r') as f:
            config = json.load(f)
    except:
        print('*** could not load config file: %s' % args.config_file)
        quit(0)



    if args.mode == 'train':
        bssd = bssd(config)
        bssd.train()

    if args.mode == 'valid':
        bssd = bssd(config)
        bssd.validate()

    if args.mode == 'valid_with_wpe':
        bssd = bssd(config)
        # number of estimation should be number of times evaluating the si-sdr and stoi metrics
        bssd.validate_with_wpe(number_of_estimations = 1)  

    if args.mode == 'plot_dereverb_comparison':
        bssd = bssd(config)
        bssd.plot_dereverb_comparison()

    if args.mode == 'plot':
        bssd = bssd(config)
        bssd.plot()

    if args.mode == 'save_rev_files':
        bssd = bssd(config)
        bssd.save_rev_files()


