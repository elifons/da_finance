import os
import pandas as pd
import json
import numpy as np
from collections import deque
from tensorboardX import SummaryWriter
import torch


class SummaryLogger(object):
    """Writes entries directly to event files in the logdir

    """
    def __init__(self, logdir=None, comment=''):

        if not logdir:
            import socket
            from datetime import datetime
            current_time = datetime.now().strftime('%b%d_%H-%M-%S')
            logdir = os.path.join(
                'runs', current_time + '_' + socket.gethostname() + comment)
            os.makedirs(logdir)
        else:
            try:
                os.makedirs(logdir)
            except FileExistsError:
                pass

        self.logdir = logdir
        self.writer = SummaryWriter(self.logdir)
        self.scalar_dict = {}
        self.vector_dict = {}


    def __append_to_scalar_dict(self, tag, scalar_value, global_step, timestep):
        from tensorboardX.x2num import make_np
        if tag not in self.scalar_dict.keys():
            self.scalar_dict[tag] = []
        self.scalar_dict[tag].append(
            [global_step, timestep, float(make_np(scalar_value))])


    def __append_to_vector_dict(self, tag, vector_value, global_step, timestep):
        import numpy as np
        if tag not in self.vector_dict.keys():
            self.vector_dict[tag] = []
        self.vector_dict[tag].append(
            [global_step, timestep, np.array(vector_value)])

    def add_scalar(self, tag, scalar_value, global_step=None, timestep=None, tb=False):
        self.__append_to_scalar_dict(tag, scalar_value, global_step, timestep)
        if tb is True:
            self.writer.add_scalar(tag, scalar_value, global_step)

    def add_scalars(self,tag, dict_scalars, global_step=None, timestep=None, tb=False):
    	if tb is True:
    		self.writer.add_scalars(tag, dict_scalars, global_step)


    def add_custom_scalar(self, tag, scalar_value, global_step=None, timestep=None, tb=False):
        if tb is True:
            self.writer.add_scalar(tag, scalar_value, global_step)


    def add_vector(self, tag, vector_value, global_step=None, timestep=None):
        self.__append_to_vector_dict(tag, vector_value, global_step, timestep)


    def save_model(self, network, tag='checkpoint_net.pth'):
        print('saving model')
        torch.save(network.state_dict(), self.logdir+'/'+tag)


    def __create_target_filename(self, key):
        filename = key + '.csv'
        return os.path.join(self.logdir,filename)


    def dict_to_files(self):
        if self.scalar_dict is not None:
            for key, value in self.scalar_dict.items():
                df = pd.DataFrame(value, columns = ['epi', 'timestep', key])
                df.to_csv(self.__create_target_filename(key))
        if self.vector_dict is not None:
            for key, value in self.vector_dict.items():
                df = pd.DataFrame(value, columns = ['epi', 'timestep', key])
                tags = df[key].apply(pd.Series)
                tags = tags.rename(columns = lambda x : key + str(x))
                full_df = pd.concat([df[df.columns[:2]], tags], axis=1)
                full_df.to_csv(self.__create_target_filename(key))

    def close(self):
        self.writer.close()   
        self.dict_to_files()
