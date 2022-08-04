"""
Mar. 10, 2022.

Version of python is implemented as:
Python: v3.9.6

This code includes functions for our Uncertainty-Aware Multiple Imputation (UA-MI) framework on top of base model, Multinomial Variational Autoencoder (MultVAE).
Original source code from "Variational autoencoders for collaborative filtering" is partially used.

Additionally, this code also includes functions for methods of quantifying the uncertainty in our framework: Sampling & dropout based methods (Noted as "sampling" & "dropout").

Versions of the libraries are implemented as:
Bottleneck: v1.3.2
Numpy: v1.20.3
Tensorflow-GPU for RTX-3090(CUDA v11.4.100, cuDNN v8.2.2): v2.5.0
"""

import bottleneck as bn
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

class UA_MI_base(object):

    # This function is for setting the parameters of MultVAE: dimensions of encoder and decoder, regularization parameter, learning rate.

    def __init__(self, decoder_dim, encoder_dim = None, reg_param = 0.01, learning_rate = 1e-3, random_seed = None):

        self.decoder_dim = decoder_dim

        if encoder_dim is None:
            self.encoder_dim = decoder_dim[::-1]
        else:
            assert encoder_dim[0] == decoder_dim[-1], "Input and output dimension must equal each other for autoencoders."
            assert encoder_dim[-1] == decoder_dim[0], "Latent dimension for p- and q-network mismatches."
            self.encoder_dim = encoder_dim

        self.total_dim = self.encoder_dim + self.decoder_dim[1:]
        self.reg_param = reg_param
        self.learning_rate = learning_rate
        self.random_seed = random_seed
        self.construct_placeholders()

    # This function is for variable placeholders of various input variables: input interaction, keeping probability of dropout & MC dropout, training check, annealing parameter.

    def variable_placeholders(self):

        self.input_interaction = tf.placeholder(dtype = tf.float32, shape = [None, self.total_dim[0]])
        self.keeping_prob_dropout = tf.placeholder_with_default(1.0, shape = None)
        self.check_training = tf.placeholder_with_default(0., shape = None)
        self.check_training_fill = tf.placeholder_with_default(1., shape = None)
        self.anneal_param = tf.placeholder_with_default(1., shape = None)

    # This function is for overall training process of MultVAE.

    def training_process(self):

        self.weights_var()

        saver, pred_rate, kl_divergence = self.construct_enc_dec()
        log_sm_pred_rate = tf.nn.log_softmax(pred_rate)
        neg_log_likelihood = -tf.reduce_mean(tf.reduce_sum(log_sm_pred_rate * self.input_interaction, axis = -1))

        weights = self.weights_enc + self.weights_dec

        for i in range(len(weights)):
            reg_var = 0.
            reg_var += tf.nn.l2_loss(weights[i])

        neg_ELBO = neg_log_likelihood + self.anneal_param * kl_divergence + 2 * self.reg_param * reg_var
        train_opt = tf.train.AdamOptimizer(self.learning_rate).minimize(neg_ELBO)

        tf.summary.scalar('negative_multi_ll', neg_log_likelihood)
        tf.summary.scalar('KL', kl_divergence)
        tf.summary.scalar('neg_ELBO_train', neg_ELBO)

        merged = tf.summary.merge_all()

        return saver, pred_rate, neg_ELBO, train_opt, merged

    # This function is for computing encoder network -> composites of latent variable z & KL divergence term are generated.

    def encoder(self):

        mu_enc, std_enc, kl_divergence = None, None, None

        output = tf.nn.l2_normalize(self.input_interaction, axis = 1)
        output = tf.nn.dropout(output, keep_prob = self.keeping_prob_dropout)
        
        for i, (weight, bias) in enumerate(zip(self.weights_enc, self.biases_enc)):
            output = tf.matmul(output, weight) + bias
            
            if i != len(self.weights_enc) - 1:
                output = tf.nn.tanh(output)
            else:
                mu_enc = output[:, :self.encoder_dim[-1]]
                log_var_enc = output[:, self.encoder_dim[-1]:]
                std_enc = tf.exp(0.5 * log_var_enc)
                kl_divergence = tf.reduce_mean(tf.reduce_sum(0.5 * (-log_var_enc + tf.exp(log_var_enc) + mu_enc**2 - 1), axis = 1))

        return mu_enc, std_enc, kl_divergence

    # This function is for computing decoder network -> output of predictive rate is generated.

    def decoder(self, z):

        output = z

        for i, (weight, bias) in enumerate(zip(self.weights_dec, self.biases_dec)):
            output = tf.matmul(output, weight) + bias

            if i != len(self.weights_dec) - 1:
                output = tf.nn.tanh(output)

        return output

    # This function is for constructing total encoder & decoder network which computes the final predictive rate.

    def construct_enc_dec(self):

        mu_enc, std_enc, kl_divergence = self.encoder()
        epsilon = tf.random_normal(tf.shape(std_enc))
        sampled_z = mu_enc + self.check_training * epsilon * std_enc

        pred_rate = self.decoder(sampled_z)

        return tf.train.Saver(), pred_rate, kl_divergence

    # This function is for constructing weight variables of each encoder & decoder.

    def weights_var(self):

        self.weights_enc, self.biases_enc = [], []

        for i, (dim_in, dim_out) in enumerate(zip(self.encoder_dim[:-1], self.encoder_dim[1:])):
            if i == len(self.encoder_dim[:-1]) - 1:
                dim_out *= 2

            weight_key = 'weight_enc_{}to{}'.format(i, i+1)
            bias_key = 'bias_enc_{}'.format(i+1)

            self.weights_enc.append(tf.get_variable(
                name = weight_key,
                shape = [dim_in, dim_out],
                initializer = tf.keras.initializers.glorot_uniform(seed = self.random_seed)))

            self.biass_enc.append(tf.get_variable(
                name = bias_key,
                shape = [dim_out],
                initializer = tf.keras.initializers.TruncatedNormal(stddev = 0.001, seed = self.random_seed)))

            tf.summary.histogram(weight_key, self.weights_enc[-1])
            tf.summary.histogram(bias_key, self.biases_enc[-1])

        self.weights_dec, self.biases_dec = [], []

        for i, (dim_in, dim_out) in enumerate(zip(self.decoder_dim[:-1], self.decoder_dim[1:])):
            weight_key = 'weight_dec_{}to{}'.format(i, i+1)
            bias_key = 'bias_dec_{}'.format(i+1)

            self.weights_dec.append(tf.get_variable(
                name = weight_key,
                shape = [dim_in, dim_out],
                initializer = tf.keras.initializers.glorot_uniform(seed = self.random_seed)))

            self.biass_dec.append(tf.get_variable(
                name = bias_key,
                shape = [dim_out],
                initializer = tf.keras.initializers.TruncatedNormal(stddev = 0.001, seed = self.random_seed)))
              
            tf.summary.histogram(weight_key, self.weights_dec[-1])
            tf.summary.histogram(bias_key, self.biases_dec[-1])

    # This function is for performing sampling-based method.

    def sampling(self):

        mu_enc, std_enc = None, None

        output = tf.nn.l2_normalize(self.input_interaction, axis = 1)
        output = tf.nn.dropout(output, keep_prob = self.keeping_prob_dropout)
        
        for i, (weight, bias) in enumerate(zip(self.weights_enc, self.biases_enc)):
            output = tf.matmul(output, weight) + bias
            
            if i != len(self.weights_enc) - 1:
                output = tf.nn.tanh(output)
            else:
                mu_enc = output[:, :self.encoder_dim[-1]]
                log_var_enc = output[:, self.encoder_dim[-1]:]
                std_enc = tf.exp(0.5 * log_var_enc)

        epsilon = tf.random_normal(tf.shape(std_enc))
        resampled_z = mu_enc + self.check_training_fill * epsilon * std_enc

        output = resampled_z

        for i, (weight, bias) in enumerate(zip(self.weights_dec, self.biases_dec)):
            output = tf.matmul(output, weight) + bias

            if i != len(self.weights_dec) - 1:
                output = tf.nn.tanh(output)

        pred_rate = tf.nn.softmax(output)

        return pred_rate

    # This function is for performing dropout-based method.

    def dropout(self):
        
        mu_enc = None

        output = tf.nn.l2_normalize(self.input_interaction, axis = 1)
        output = tf.nn.dropout(output, keep_prob = self.keeping_prob_dropout)
        
        for i, (weight, bias) in enumerate(zip(self.weights_enc, self.biases_enc)):
            output = tf.matmul(output, weight) + bias
            
            if i != len(self.weights_enc) - 1:
                output = tf.nn.tanh(output)
                output = tf.nn.dropout(output, keep_prob = self.keeping_prob_dropout)
            else:
                mu_enc = output[:, :self.encoder_dim[-1]]

        resampled_z = mu_enc

        output = resampled_z
        output = tf.nn.dropout(output, keep_prob = self.keeping_prob_dropout)
        
        for i, (weight, bias) in enumerate(zip(self.weights_dec, self.biases_dec)):
            output = tf.matmul(output, weight) + bias

            if i != len(self.weights_dec) - 1:
                output = tf.nn.tanh(output)
                output = tf.nn.dropout(output, keep_prob = self.keeping_prob_dropout)

        pred_rate = tf.nn.softmax(output)

        return pred_rate