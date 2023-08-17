from base import BaseDataLoader
import pandas as pd
import torch
import math
from sklearn import preprocessing

class TimeseriesDataLoader(BaseDataLoader):
    """
    Timeseries data loader for temporal causal discovery
    """
    def __init__(self, data_dir, batch_size, time_step, output_window, feature_dim, output_dim, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        self.data_dir = data_dir
        self.df_data = pd.read_csv(self.data_dir)
        self.data_len = len(self.df_data.index)
        self.data = self.df_data.values.astype('float32')

        self.batch_size = batch_size
        self.time_step = time_step
        self.output_window = output_window
        self.series_num = self.data.shape[1]
        self.feature_dim = feature_dim
        self.output_dim = output_dim
        
        # zoom data for regression relevance propogation
        scaler = preprocessing.MinMaxScaler(feature_range=[0.5,1])
        self.data = scaler.fit_transform(self.data)
        
        # construct input samples
        self.dataset = []
        assert self.time_step<len(self.data)+1, "the length of input window must be shorter than whole data"
        assert self.output_window<self.time_step, \
            "the length of output window must be shorter than input window. \
            Practically, we ignore the prediction of the first time slot for \
            the sake of fairness, because the observations of each time series \
            do not contribute to their own predictions in the first time slot \
            due to the right shifting of self-convolution result, which is \
            different from other time slots."
        for i in range(self.time_step,len(self.data)+1):
            self.dataset.append((self.data[i-self.time_step:i].reshape(self.time_step,self.series_num,self.feature_dim) ,self.data[i-self.output_window:i].reshape(self.output_window,self.series_num,self.output_dim)))
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)