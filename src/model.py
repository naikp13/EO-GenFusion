import tensorflow as tf
from keras.layers import Lambda, Input, Dense, Dropout
from keras.models import Model
from keras.losses import mse
from keras import backend as K
from keras.constraints import unit_norm

def sampling(args):
    """
    Reparameterization trick by sampling from an isotropic unit Gaussian.
    Args:
        args (tensor): Mean and log of variance of Q(z|X)
    Returns:
        z (tensor): Sampled latent vector
    """
    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    epsilon = K.random_normal(shape=(batch, dim))
    thre = K.random_uniform(shape=(batch, 1))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon

def build_vae_model(original_dim):
    """
    Build the Variational Autoencoder model.
    Args:
        original_dim (int): Dimension of input features
    Returns:
        vae (Model): VAE model
        encoder (Model): Encoder model
        decoder (Model): Decoder model
    """
    input_shape_x = (original_dim,)
    input_shape_r = (1,)
    intermediate_dim1 = original_dim - 2
    intermediate_dim2 = original_dim - 4
    latent_dim = original_dim - 6

    # Encoder input
    input_x = Input(shape=input_shape_x, name='encoder_input')
    input_r = Input(shape=input_shape_r, name='ground_truth')
    inputs_x_dropout = Dropout(0.25)(input_x)

    # Encoder layers
    inter_x1 = Dense(intermediate_dim1, activation='tanh', name='encoder_intermediate')(inputs_x_dropout)
    inter_x2 = Dense(intermediate_dim2, activation='tanh', name='encoder_intermediate_2')(inter_x1)

    # Posterior on Y
    r_mean = Dense(1, name='r_mean')(inter_x2)
    r_log_var = Dense(1, name='r_log_var')(inter_x2)

    # q(z|x)
    z_mean = Dense(latent_dim, name='z_mean')(inter_x2)
    z_log_var = Dense(latent_dim, name='z_log_var')(inter_x2)

    # Sampling
    z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])
    r = Lambda(sampling, output_shape=(1,), name='r')([r_mean, r_log_var])

    # Latent generator
    pz_mean = Dense(latent_dim, name='pz_mean', kernel_constraint=unit_norm())(r)

    # Encoder model
    encoder = Model([input_x, input_r], [z_mean, z_log_var, z, r_mean, r_log_var, r, pz_mean], name='encoder')

    # Decoder model
    latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
    inter_y1 = Dense(intermediate_dim2, activation='tanh')(latent_inputs)
    inter_y2 = Dense(intermediate_dim1, activation='tanh')(inter_y1)
    outputs = Dense(original_dim)(inter_y2)
    decoder = Model(latent_inputs, outputs, name='decoder')

    # VAE model
    outputs = decoder(encoder([input_x, input_r])[2])
    vae = Model([input_x, input_r], outputs, name='vae_mlp')

    # Custom loss
    reconstruction_loss = mse(input_x, outputs)
    kl_loss = 1 + z_log_var - K.square(z_mean - pz_mean) - K.exp(z_log_var)
    kl_loss = -0.5 * K.sum(kl_loss, axis=-1)
    label_loss = tf.divide(0.5 * K.square(r_mean - input_r), K.exp(r_log_var)) + 0.5 * r_log_var
    vae_loss = K.mean(reconstruction_loss + kl_loss + label_loss)

    vae.add_loss(vae_loss)
    vae.compile(optimizer='adam')
    vae.save_weights('weights.h5')

    return vae, encoder, decoder