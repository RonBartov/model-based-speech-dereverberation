# -*- coding: utf-8 -*-
__author__ = "Lukas Pfeifenberger"


import numpy as np
import glob
import sys
import os

sys.path.append(os.path.abspath('../'))
from algorithms.audio_processing import *



# loader class for mono wav files, i.e. wsj0

class audio_loader(object):

    # --------------------------------------------------------------------------
    def __init__(self, config, set):

        self.fs = config['fs']
        self.wlen = config['wlen']
        self.shift = config['shift']
        self.samples = int(self.fs*config['duration'])
        self.nfram = int(np.ceil( (self.samples-self.wlen+self.shift)/self.shift ))
        self.nbin = int(self.wlen/2+1)
        self.set = set

        if set == 'train':
            path = config['train_path'] 
        elif set == 'valid':
            path = config['valid_path']
        elif set == 'test':
            path = config['test_path']
        else:
            print('unknown set name: ', set)
            quit(0)

        # self.file_list = glob.glob(path + '*/' + '*.wav')  #glob.glob(path+'*.wav')
        self.file_list = glob.glob(path + '*.wav')
        self.numof_files = len(self.file_list)
        self.n_subset = config.get("n_subset_speakers",50)     # e.g. 50
        self.seed  = config.get("split_seed", 1234)      # deterministic

        self.get_speakers()
        self.n_speakers = len(self.speaker_keys)

        print('*** audio_loader found %d unique speakers in %d files in: %s' % (self.n_speakers, self.numof_files, path))



    #-------------------------------------------------------------------------
    def get_speakers(self):

        speaker_keys = []
        for f in self.file_list:
            speaker_keys.append(f.split('/')[-2])

        self.speaker_keys = sorted(list(set(speaker_keys)))

        if self.set == "train" and self.n_subset is not None:
            n = min(self.n_subset, len(self.speaker_keys))
            rng = np.random.default_rng(self.seed)
            rng.shuffle(self.speaker_keys)                     # one deterministic shuffle
            self.speaker_keys = self.speaker_keys[: n] # keep first N after shuffle
            print(f"[info] train: using {len(self.speaker_keys)}/{n} speakers "
            f"(seed={self.seed})")


        self.nsid = len(self.speaker_keys)

        self.files_list_per_speaker = {}
        for speaker_key in self.speaker_keys:

            flist = [f for f in self.file_list if '/'+speaker_key+'/' in f]
            self.files_list_per_speaker[speaker_key] = flist



    #-------------------------------------------------------------------------
    def get_speaker_id_by_key(self, speaker_key):

        for sid, key in enumerate(self.speaker_keys):
            if speaker_key==key:
                return sid

        print('speaker key ', speaker_key, ' was not found.')
        return None



    #-------------------------------------------------------------------------
    def load_random_file_from_speaker_id(self, speaker_id=None):

        if speaker_id is None:
            speaker_id = np.random.choice(self.nsid)

        f = np.random.choice(self.files_list_per_speaker[self.speaker_keys[speaker_id]])
        x, fs = audioread(f)
        assert (fs == self.fs)
        x = self.pad_file(x, self.samples)

        return x



    #-------------------------------------------------------------------------
    def load_random_files_from_speaker_id(self, speaker_id):
        
        num_of_rand_files = 0
        chosen_files = []
        speaker_name = self.speaker_keys[speaker_id]
        x = np.zeros((self.samples,), dtype=np.float32)
        n = 0
        while n<self.samples:
            num_of_rand_files += 1
            f = np.random.choice(self.files_list_per_speaker[self.speaker_keys[speaker_id]])
            chosen_files.append(f)
            s, fs = audioread(f)
            length = s.shape[0]
            n1 = min(n+length, self.samples)
            x[n:n1] = s[0:n1-n]
            n = n1

        #commented out prints about speakers and details on wav file used to generate
        # print(f"in 'load_random_files_from_speaker_id': \n"
        #   f" number of files loaded: {num_of_rand_files}, speaker id: {speaker_id}, speaker name: {speaker_name}\n"
        #   f"Files used:\n" + "\n".join(chosen_files) + "\n\n")
        return x



    #-------------------------------------------------------------------------
    def load_random_files_from_speaker_ids(self, speaker_id_list):

        wav_list = []

        for speaker_id in speaker_id_list:
            wav_list.append( self.load_random_files_from_speaker_id(speaker_id) )

        s0 = np.stack(wav_list, axis=0)                 # shape = (nbatch, samples)
        Fs0 = rfft(s0, n=self.samples, axis=1)          # shape = (nbatch, samples/2+1)

        return Fs0

 

    #-------------------------------------------------------------------------
    def load_random_files_from_random_speakers(self,  nsrc=2):

        wav_list = []

        # randomize speaker indices
        if nsrc <= self.nsid:
            sid_list = np.random.permutation(self.nsid)[:nsrc]
        else:
            sid_list = np.arange(nsrc)%self.nsid

        # randomize file indices
        for sid in sid_list:
            f = np.random.choice(self.files_list_per_speaker[self.speaker_keys[sid]])
            x, fs = audioread(f)
            assert (fs == self.fs)
            wav_list.append( self.pad_file(x, self.samples) )
            
        sid_list = sid_list.astype(np.int32)
        return wav_list, sid_list



    #-------------------------------------------------------------------------
    def load_file(self, utterance=None, samples=None):

        if utterance is None:
            # load a random file
            x, fs = audioread(np.random.choice(self.file_list))
            assert (fs == self.fs)
        else:
            # load a specific file
            if utterance in self.file_list:
                x, fs = audioread(utterance)
                assert (fs == self.fs)
            else:
                print('requested file: %x was not found.' % utterance)
                quit(0)

        if samples is None:
            return x
        else:
            return self.pad_file(x, samples)



    #-------------------------------------------------------------------------
    # repeat the file until it has <samples> length
    def pad_file(self, x, samples, random_offset=True):

        #return file if no sample count was given
        if samples is None:
            return x

        #start the file from a random offset of up to half the length of the file
        if random_offset is True:
            x = np.roll(x, int(np.random.uniform()*x.shape[0]/2), axis=0)

        #repeat the file until <samples> have been read
        y = np.zeros((samples,), dtype=np.float32)
        i0 = 0
        while i0 < samples:
            
            if i0==0:
                x0 = x
                i1 = np.minimum(x0.shape[0], samples)
                y[0:i1] = x0[0:i1]
            else:
                i1 = np.minimum(i0+x.shape[0], samples)
                y[i0:i1] = x[0:i1-i0]

            i0 = i1

        return y






#---------------------------------------------------------
#---------------------------------------------------------
if __name__ == "__main__":


    config = {
        'fs': 16000,
        'wlen': 1024,
        'shift': 256,
        'duration': 10.0,
        'wsj0_path': '/clusterFS/project/beamforming/data/wsj0/*_dt_*/*/',
    }
    wsj0_loader = audio_loader(config, 'wsj0_path')




