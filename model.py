import os
import glob
import time
import math
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import LSTM, CuDNNLSTM, Bidirectional
from tensorflow.keras import backend as K
from modules import multihead_attention

class DataLoader:
    def __init__(self, inputs_list, batch_size=None, is_shuffle=False, is_prefetch=False):
        def load_np_from_path(input_path):
            sequence = np.load(input_path.decode("utf-8"))

            return sequence.astype(np.float32)
        
        n_data = len(inputs_list)
        if batch_size is None : batch_size = n_data
            
        dataset = tf.data.Dataset.from_tensor_slices(inputs_list)
        dataset = dataset.map(map_func= lambda input: tuple(tf.py_func(load_np_from_path, inp=[input], Tout=[tf.float32])),
                              num_parallel_calls = 12)
        
        dataset = dataset.shuffle(buffer_size=10*batch_size) if is_shuffle != False else dataset
        dataset = dataset.prefetch(buffer_size=5*batch_size) if is_prefetch != False else dataset
        dataset = dataset.batch(batch_size)
        
        self.batch_steps = int(np.ceil(n_data / batch_size))
        self.iterator = dataset.make_initializable_iterator()
        self.inputs = self.iterator.get_next()

        
class SSSAE:
    def __init__(self, config):
        self.n_layers         = config.n_layers
        self.input_dims       = config.dims
        self.is_bidirectional = config.bidirectional
        self.learning_rate    = config.learning_rate
        self.batch_size       = config.batch_size
        self.beta_1           = config.beta_1
        self.beta_2           = config.beta_2
        self.epsilon          = config.epsilon
        self.epochs           = config.epochs
        self.save_path        = config.save_path

        self.inputs = tf.placeholder(tf.float32, shape=[None, None, self.input_dims], name='inputs')

        self._build()
        
        tf_config = tf.ConfigProto()
        tf_config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=tf_config)
        self.saver = tf.train.Saver()
        
    
    def _build(self):
        x = self.inputs

        n_layers = self.n_layers
        units = self.input_dims
        
        encoder_states = []
        last_h_states = [] # last states of each Encoder outputs
        self.attentions = []

        
        """ Encoder """
        for i in range(n_layers//2):
            if i % 2 == 0:
                units = int(units / 2) if self.is_bidirectional else units
            else:
                units = int(units / 2)

            x_forward = None
            x_backward = None

            forward_layer = CuDNNLSTM(units=units, kernel_initializer='he_normal', 
                                     return_sequences=True, return_state=True, name=f'layer_{i}_forward')

            x_forward, forward_h, forward_c = forward_layer(x)

            if bidirectional == True:
                backward_layer = CuDNNLSTM(units=units, kernel_initializer='he_normal', 
                                           return_sequences=True, return_state=True, go_backwards=True,
                                           name=f'layer_{i}_backward')

                x_backward, backward_h, backward_c = backward_layer(x)

            if x_backward != None:
                x = tf.concat([x_forward, x_backward], axis=-1)
                last_h_states += [forward_h, backward_h]
                encoder_states.append([[forward_h, forward_c], [backward_h, backward_c]])

            else:
                x = x_forward
                last_h_states += [forward_h]
                encoder_states.append([[forward_h, forward_c]])
                
            x, attention = multihead_attention(x, x, x, key_masks=None, num_heads=self.num_heads, normalize=True, scope=f"multihead_attention_{i}")
            self.attentions.append(attention)

        """ Decoder """
        for j in range(n_layers//2):
            if j % 2 != 0:
                units *= 2
                
            x_forward = None
            x_backward = None
            forward_initial_state = None
            backward_initial_state = None
            
            forward_initial_state = encoder_states[-(j+1)][0]
            if self.is_bidirectional:
                backward_initial_state = encoder_states[-(j+1)][1]

            forward_layer = CuDNNLSTM(units=units, kernel_initializer='he_normal', 
                                     return_sequences=True, name=f'layer_{j}_forward')

            x_forward = forward_layer(x, initial_state=forward_initial_state)

            if self.is_bidirectional == True:
                backward_layer = CuDNNLSTM(units=units, kernel_initializer='he_normal', 
                                           return_sequences=True, go_backwards=True, name=f'layer_{j}_backward')

                x_backward = backward_layer(x, initial_state=backward_initial_state)

            if x_backward != None:
                x = tf.concat([x_forward, x_backward], axis=-1)
            else:
                x = x_forward
                
            x, attention = multihead_attention(x, x, x, key_masks=None, num_heads=self.num_heads, normalize=True, scope=f"multihead_attention_{i}")
            #self.attentions.append(attention)

        outputs = x
        self.laten_vector = tf.concat(last_h_states, axis=-1) # laten vectors
        self.loss = tf.reduce_mean(tf.pow(outputs - self.inputs, 2))


    def train(self, inputs_list):
        """
        param: inputs_list: list, a list of data path.
        """
        def calc_time(n_iteration, elasped_time):
            total_time = n_iteration * elasped_time

            hour = str(int(total_time // 3600)).zfill(2)
            minute = str(int((total_time % 3600) // 60)).zfill(2)
            second = str(int(total_time % 60)).zfill(2)


            log = f"{hour}:{minute}:{second}"

            return log 
        
        
        # Initialize Optimizer
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, 
                                                beta1=self.beta_1, beta2=self.beta_2, epsilon=self.epsilon).minimize(self.loss)
        self.sess.run(tf.global_variables_initializer()) # Initialize variables
        
        
        """ Training """
        trainData_loader = DataLoader(inputs_list, self.batch_size, is_shuffle=True, is_prefetch=False) 
        n_iter = trainData_loader.batch_steps * self.epochs
        
        switch = 20
        self.loss_list = []
        check_time = lambda step: None if step != switch else time.time()
        best_recons_loss = 10000
        is_save = False
        
        # Make save folder
        if not os.path.exists(self.save_path): 
            os.makedirs(self.save_path)
            
        for epoch in range(self.epochs):
            self.sess.run(trainData_loader.iterator.initializer)
            start = time.time()
            total_loss = 0
            
            for step  in range(trainData_loader.batch_steps):
                _start = check_time(step)

                batch_inputs = self.sess.run(trainData_loader.inputs)[0]
                
                loss, _ = self.sess.run([self.loss, self.optimizer], feed_dict={self.inputs:batch_inputs})
                total_loss += loss / trainData_loader.batch_steps
                
                _end = check_time(step)
                if epoch == 0 and step == switch: print("Predict training time :", calc_time(n_iter-switch, _end-_start))
                    
            end = time.time()
            self.loss_list.append(total_loss)
            
            log = None
            epoch_log = f"Epoch : {epoch}"
            time_log = f"Elasped time : {calc_time(1, end-start)}"
            
            epoch_log = f"Epoch : {epoch}"
            loss_log = f"Recons loss : {total_loss:.4f}"

            log = f"{epoch_log}\t{loss_log}\t{time_log}"
            print(log)
            
            
            # Save best reconstruct loss
            if epoch >= self.epochs - 100:
                if epoch == self.epochs - 100:
                    self.saver.save(self.sess, '{}/model.ckpt'.format(self.save_path))
                    best_recons_loss = total_loss
                else:
                    if best_recons_loss > total_loss:
                        best_recons_loss = total_loss
                        self.saver.save(self.sess, '{}/model.ckpt'.format(self.save_path))
            
        print("Training completed")
        self.loss_list = np.array(self.loss_list)
        
        
    def test(self, inputs_list, name, threshold=None):
        """
        param:
            inputs_list : list, a list of data path.
        
        return:
            loss_list : numpy, shape=(n_data).
            laten_vectors : numpy, shape=(n_data, D), D is dimension.
        """
        testData_loader = DataLoader(inputs_list, batch_size=1, is_shuffle=False, is_prefetch=False)
        self.sess.run(testData_loader.iterator.initializer)
        
        loss_list = []
        frame_loss_list = []
        laten_vector_list = []
        n_normal = 0
        n_abnormal = 0
        
        
        for wav_path in inputs_list:
            wav_name = os.path.splitext(os.path.basename(wav_path))[0]
            data = np.load(wav_path)
            batch_inputs = np.expand_dims(data, axis=0)
            loss, frame_loss, laten_vector = self.sess.run([self.loss, self.frame_loss, self.laten_vector], feed_dict={self.inputs:batch_inputs})
            #snr = - 20 * np.log10(loss)
            
            if threshold != None:
                if loss <= threshold :
                    is_abnormal = "NORMAL"
                    n_normal += 1
                else:
                    is_abnormal = "ABNORMAL"
                    n_abnormal += 1
                #print(f"{name}/{wav_name} :\tError Value / Thres : {snr:.2f}dB / {threshold:.2f}dB\t\tPredict : {is_abnormal}")
            
            loss_list.append(loss)
            laten_vector_list.append(laten_vector)
            
        
        loss_list = np.array(loss_list)
        laten_vector_list = np.concatenate(laten_vector_list, axis=0)
        
        return loss_list, laten_vector_list, n_normal, n_abnormal
        
                                                                                 
    def load_weights(self):
        assert os.path.exists(self.save_path), "Model's checkpoint not found!"
        self.saver.restore(self.sess,'{}/model.ckpt'.format(self.save_path))
