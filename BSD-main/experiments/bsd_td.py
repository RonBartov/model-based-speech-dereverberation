# -*- coding: utf-8 -*-
import time
import numpy as np
import argparse
import json
import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
sys.path.append(os.path.abspath('../'))
sys.path.append("/gpfs0/bgu-br/users/yahelso/model-based-speech-dereverberation/BSD-main")

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
import pandas as pd
import matplotlib.pyplot as plt
import ipdb

np.set_printoptions(precision=3, threshold=3, edgeitems=3)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
# tf.debugging.set_log_device_placement(True)


# if not tf.config.list_physical_devices('GPU'):
#     print("No GPU detected. Exiting to avoid idle job.")
#     exit(1)

#-------------------------------------------------------------------------
#-------------------------------------------------------------------------

class bsd(object):

    def __init__(self, config, set='train',verbose=1,speakers=20):

        self.config = config
        self.fgen = feature_generator(config, set)
        self.nsrc = config['nsrc']                      # number of concurrent speakers
        self.speakers_per_batch = min(speakers, self.fgen.nspk)         # self.fgen.nspk - number of speakers in date set
        self.batch_size = config.get('batch_size', 24)
        self.val_batch_size = config.get("val_batch_size", self.batch_size)
        self.is_load_weights = config.get("is_load_weights", True)
        self.is_save_weights = config.get("is_save_weights", True)
        
        os.makedirs(self.config['log_path'], exist_ok=True)

        if speakers > self.fgen.nspk:
            print(f"[warn] --speakers clipped from {speakers} to {self.fgen.nspk}")
        if self.val_batch_size < self.batch_size:
            self.val_batch_size = self.batch_size


        

        self.filename = os.path.basename(__file__)
        self.name = self.filename[:-3] + '_' + config['rir_type']
        self.creation_date = os.path.getmtime(self.filename)
        self.weights_file = self.config['weights_path'] + self.name + '.h5'
        self.predictions_file = self.config['predictions_path'] + self.name + '.mat'
        self.predictions_file_for_compare = self.config['predictions_for_compare_path'] + self.name + '.mat'
        self.timestamp = time.strftime("%Y%m%d-%H%M%S")

        self.logger = Logger(self.name)
        self.verbose = verbose

        self.samples = self.fgen.samples                # number of samples per utterance
        self.nmic = self.fgen.nmic                      # number of microphones
        self.ndoa = self.fgen.ndoa                      # number of DOA vectors on the sphere

        self.nbin = 500                                 # latent space H
        self.wlen = 200                                 # convolution kernel filter length
        self.shift = self.wlen//4                       # convolution stride
        self.ndim = 100                                 # embedding dimension E

        self.beamforming = beamforming(self.fgen)
        # self.identification = identification(self.fgen) # ------------------------------ identification -----------------------------------
        self.create_model()

        # for benchmark
        self.wpe_model = ClassicWPE()

        self.si_sdr_bsd = []
        self.si_sdr_wpe = []
        self.epoch = 0

        if self.is_load_weights:
            # data = load_numpy_from_mat(self.predictions_file)
            data = load_numpy_from_mat(self.predictions_file_for_compare)
            print(data.keys())
            # ipdb.set_trace()

            if data is not None:
                if 'epoch' in data.keys():
                    self.epoch = data['epoch']
                    self.si_sdr_bsd = data['si_sdr_bsd']
                    self.si_sdr_wpe = data['si_sdr_wpe']
                    # self.eer = data['eer']
        else:
            print("skipping load_state (flag=False)", flush=True)




    #---------------------------------------------------------
    def create_model(self):

        print('*** creating model: %s' % self.name)

        Z = Input(shape=(self.samples, self.nmic), dtype=tf.float32)                # shape = (nbatch, nsamples, nmic)
        R = Input(shape=(self.samples,), dtype=tf.float32)                          # shape = (nbatch, nsamples)
        pid = Input(shape=(1,), dtype=tf.int32)                                     # shape = (nbatch,)
        sid = Input(shape=(1,), dtype=tf.int32)                                     # shape = (nbatch, 1)

        [Py, Y, cost_bf] = self.beamforming.model([Z, R, pid])
        # [E, cost_id] = self.identification.model([Py, sid])                       # ------------------------------ identification -----------------------------------

        # ------------------------------ identification -----------------------------------
        # compile model
        # self.model = Model(inputs=[Z, R, pid, sid], outputs=[Y, E])
        # self.model.add_loss(cost_bf + 0.01*cost_id)
        # self.model.compile(loss=None, optimizer='adam')

        # compile model
        self.model = Model(inputs=[Z, R, pid, sid], outputs=[Y])
        self.model.add_loss(cost_bf)
        self.model.compile(loss=None, optimizer='adam')

        print(self.model.summary())
        # try:
        #     self.model.load_weights(self.weights_file)
        # except:
        #     print('error loading weights file: %s' % self.weights_file)
        if self.is_load_weights and os.path.exists(self.weights_file):
            try:
                print(f"loading weights from {self.weights_file}", flush=True)
                self.model.load_weights(self.weights_file)
            except Exception as e:
                print(f"error loading weights file {self.weights_file}: {e}", flush=True)
        else:
            print("skipping load_weights (flag=False or file missing)", flush=True)



    #---------------------------------------------------------
    def save_weights(self):
        print(f"need to save weights here from {self.weights_file}")
        # self.model.save_weights(self.weights_file)
        if self.is_save_weights:
            self.model.save_weights(self.weights_file)
        return



    #---------------------------------------------------------
    def train(self):

        print('train the model')
        self.history = []
        # print(f"--> self.epoch = {self.epoch}", flush=True)
        # print(f"--> self.config['epochs'] = {self.config['epochs']}", flush=True)

        # while (self.epoch<self.config['epochs']) and self.check_date():
        while (self.epoch < self.config['epochs']):
            
            print(f"[epoch {self.epoch+1}] nspk seen by model = {self.fgen.nspk}")

            sid0 = self.fgen.generate_triplet_indices(speakers=self.speakers_per_batch, utterances_per_speaker=3)
            z, r, sid, pid = self.fgen.generate_multichannel_mixtures(nsrc=self.nsrc, sid=sid0)
            self.model.fit([z, r, pid[:,0], sid[:,0]], 
                            None, 
                            batch_size=self.batch_size, 
                            epochs=1, 
                            verbose=self.verbose, 
                            shuffle=False, 
                            callbacks=[self.logger])
            print(f"finished epoch {self.epoch + 1}/{self.config['epochs']}")
            


            self.epoch += 1
            if self.epoch <= 200:                 
                save_every = 5          # first ephocs
            else:
                save_every = 10         # advansed ephocs
            # if (self.epoch%save_every)==0:
            #     self.save_weights()
            #     self.validate()

            if self.epoch % save_every == 0:
                self.save_weights()

                # ---------- run validate_with_wpe() and log metrics ---------------
                val_dict = self.validate_with_wpe()          # make sure validate_with_wpe() returns a dict
                val_dict["epoch"] = self.epoch
                self.history.append(val_dict)

                # Save current validation curve
                df = pd.DataFrame(self.history)
                df.to_csv(os.path.join(self.config['log_path'], "val_curve.csv"), index=False)
                df.to_csv(os.path.join(self.config['log_path'], f"val_curve_{self.timestamp}.csv"), index=False)

                # Print results
                msg = "  ".join(f"{k}={v:.3f}" for k, v in val_dict.items() if k != "epoch")
                print(f"[val] epoch {self.epoch}  {msg}", flush=True)

        # --------------- write history once at the end -------------------
        if self.history:
            pd.DataFrame(self.history).to_csv("val_curve.csv", index=False)
            print("validation curve saved to val_curve.csv")
            df = pd.DataFrame(self.history)
            
            # plt.plot(df["epoch"], df["val_wpe"])
            # plt.xlabel("epoch"); plt.ylabel("Validation WPE")
            # plt.title("Validation curve"); plt.tight_layout()
            # plt.savefig(os.path.join(self.config['log_path'], "val_curve.png"))
            # plt.savefig(os.path.join(self.config['log_path'], f"val_curve_{self.timestamp}.png"))
            # plt.close()
            fig, axes = plt.subplots(2, 1, figsize=(6, 8))

            # subplot 1 – Validation WPE
            axes[0].plot(df["epoch"], df["val_wpe"])
            axes[0].set_xlabel("epoch")
            axes[0].set_ylabel("Validation WPE")
            axes[0].set_title("Validation WPE curve")

            # subplot 2 – Validation SI-SDR
            axes[1].plot(df["epoch"], df["val_si_sdr"])
            axes[1].set_xlabel("epoch")
            axes[1].set_ylabel("Validation SI-SDR (dB)")
            axes[1].set_title("Validation SI-SDR curve")

            plt.tight_layout()
            plt.savefig(os.path.join(self.config['log_path'], "val_curve.png"))
            plt.savefig(os.path.join(self.config['log_path'], f"val_curve_{self.timestamp}.png"))

            plt.close()

    #---------------------------------------------------------
    def save_rev_files(self):
        count =  0
        while (count < 3):
            print(f"epcount number {count}")
            sid0 = self.fgen.generate_triplet_indices(speakers=self.speakers_per_batch, utterances_per_speaker=3)
            z, r, sid, pid = self.fgen.generate_multichannel_mixtures(nsrc=self.nsrc, sid=sid0)  # len(sid) = 1 for test

            count += 1
            



    #---------------------------------------------------------
    def validate(self):

        sid = self.fgen.generate_triplet_indices(speakers=self.fgen.nspk, utterances_per_speaker=3)
        z, r, sid, pid = self.fgen.generate_multichannel_mixtures(nsrc=self.nsrc, sid=sid)
        y, E = self.model.predict([z, r, pid[:,0], sid[:,0]], batch_size=self.val_batch_size)

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
        if self.is_save_weights:
            save_numpy_to_mat(self.predictions_file, data)


#---------------------------------------------------------
    def validate_with_wpe(self):
        # Generate validation data
        SINGLE_UTERANCE = 1
        sid = self.fgen.generate_triplet_indices(speakers=self.fgen.nspk, utterances_per_speaker=SINGLE_UTERANCE)
        z, r, sid, pid = self.fgen.generate_multichannel_mixtures(nsrc=self.nsrc, sid=sid)

        # BSD prediction
        # y_bsd, _ = self.model.predict([z, r, pid[:,0], sid[:,0]], batch_size=self.val_batch_size) #--------------IDENTIFICATION------------------------------

        y_bsd = self.model.predict([z, r, pid[:,0], sid[:,0]], batch_size=self.val_batch_size)
        # print(f"Number of outputs: {len(y_bsd) if isinstance(y_bsd, list) else 1}")

        si_sdr_bsd = self.beamforming.si_sdr(r, y_bsd)

        # WPE dereverberation
        y_wpe = self.wpe_model.dereverb_batch(z)

        print(f"z shape {z.shape}, r shape {r.shape}, y_bsd shape {y_bsd.shape}, y_wpe shape {y_wpe.shape}")
        si_sdr_wpe = self.beamforming.si_sdr(r, y_wpe)

        # Save only relevant data
        data = {
            'z': z[0, :, 0],
            'r': r[0, :],
            'y_bsd': y_bsd[0, :],
            'y_wpe': y_wpe[0, :],
            'si_sdr_bsd': si_sdr_bsd,
            'si_sdr_wpe': si_sdr_wpe,
            'epoch': self.epoch,
        }
        if self.is_save_weights:
            save_numpy_to_mat(self.predictions_file_for_compare, data)
            file_ts = os.path.join(self.config['predictions_for_compare_path'], f"{self.name}_{self.timestamp}.mat")
            save_numpy_to_mat(file_ts , data)

        # Print for logging
        print(f"SI-SDR BSD: {si_sdr_bsd:.2f} dB | SI-SDR WPE: {si_sdr_wpe:.2f} dB")

        return {'val_si_sdr': float(si_sdr_bsd), 'val_wpe': float(si_sdr_wpe)}

    

    #---------------------------------------------------------
    def plot(self):

        z, r, sid, pid = self.fgen.generate_multichannel_mixtures(nsrc=self.nsrc)
        
        data = []
        z0 = z[0,:,0]/np.amax(np.abs(z[0,:,0]))
        data.append( 20*np.log10(np.abs(mstft(z0))) )

        for c in range(self.nsrc):
            y, E = self.model.predict([z, r, pid[:,c], sid[:,c]])                       # E is from identifiction model--> need to fix
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
    parser.add_argument('--config_file', 
                    help='name of json configuration file', 
                    default='shoebox_c2.json')
    #parser.add_argument('--mode', help='mode: [train, valid, plot]', nargs='?', choices=('train', 'valid', 'plot'), default='train')
    parser.add_argument('--mode', 
                    help='mode: [train, valid, plot,save_rev_files]', 
                    nargs='?', 
                    choices=('train', 'valid','valid_with_wpe', 'plot','plot_dereverb_comparison','save_rev_files'), 
                    default='train')
    parser.add_argument('--verbose', type=int, 
                    choices=(0,1,2), 
                    default=1,
                    help='Keras verbosity: 0 = silent, 1 = progress-bar, 2 = one-line/epoch')
    parser.add_argument('--speakers', type=int,
                    default=20,                  
                    help='How many distinct anchor speakers to sample per training iteration'
)
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
        bsd = bsd(config, verbose=args.verbose, speakers=args.speakers)
        try:
            bsd.train()
        except Exception as e:
            print(f" Training crashed: {e}")
            if hasattr(bsd, "history") and bsd.history:
                df = pd.DataFrame(bsd.history)
                df.to_csv(os.path.join(self.config['log_path'],"val_curve_crash_backup.csv"), index=False)
                df.to_csv(os.path.join(self.config['log_path'],f"val_curve_crash_backup_{self.timestamp}.csv"), index=False)
                print("Saved backup val_curve_crash_backup.csv after crash")
            raise

    if args.mode == 'valid':
        bsd = bsd(config)
        bsd.validate()

    if args.mode == 'valid_with_wpe':
        bsd = bsd(config)
        bsd.validate_with_wpe()

    if args.mode == 'plot_dereverb_comparison':
        bsd = bsd(config)
        bsd.plot_dereverb_comparison()

    if args.mode == 'plot':
        bsd = bsd(config)
        bsd.plot()

    if args.mode == 'save_rev_files':
        bsd = bsd(config)
        bsd.save_rev_files()


