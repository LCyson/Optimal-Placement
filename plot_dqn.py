#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 19 14:31:44 2021

@author: Logan
"""


#from keras.layers import Dense
import keras
from keras.utils.vis_utils import plot_model
#model = Sequential()
#model.add(Dense(2, input_dim=1, activation='relu'))
#model.add(Dense(1, activation='sigmoid'))

model =keras.models.load_model('dqn-model')

plot_model(model, to_file='dqn-model_plot.png', show_shapes=True, show_layer_names=True)