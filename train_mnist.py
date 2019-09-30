from __future__ import print_function, division

import numpy as np
import pandas as pd
import sys

from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.utils.data
import torchvision

import spatial_vae.models as models

def eval_minibatch(x, y, p_net, q_net, rotate=True, translate=True, dx_scale=0.1, theta_prior=np.pi, use_cuda=False):
    b = y.size(0)
    x = x.expand(b, x.size(0), x.size(1))

    # first do inference on the latent variables
    if use_cuda:
        y = y.cuda()

    z_mu,z_logstd = q_net(y)
    z_std = torch.exp(z_logstd)
    z_dim = z_mu.size(1)

    # draw samples from variational posterior to calculate
    # E[p(x|z)]
    r = Variable(x.data.new(b,z_dim).normal_())
    z = z_std*r + z_mu
    
    kl_div = 0
    if rotate:
        # z[0] is the rotation
        theta_mu = z_mu[:,0]
        theta_std = z_std[:,0]
        theta_logstd = z_logstd[:,0]
        theta = z[:,0]
        z = z[:,1:]
        z_mu = z_mu[:,1:]
        z_std = z_std[:,1:]
        z_logstd = z_logstd[:,1:]

        # calculate rotation matrix
        rot = Variable(theta.data.new(b,2,2).zero_())
        rot[:,0,0] = torch.cos(theta)
        rot[:,0,1] = torch.sin(theta)
        rot[:,1,0] = -torch.sin(theta)
        rot[:,1,1] = torch.cos(theta)
        x = torch.bmm(x, rot) # rotate coordinates by theta

        # calculate the KL divergence term
        sigma = theta_prior
        kl_div = -theta_logstd + np.log(sigma) + (theta_std**2 + theta_mu**2)/2/sigma**2 - 0.5

    if translate:
        # z[0,1] are the translations
        dx_mu = z_mu[:,:2]
        dx_std = z_std[:,:2]
        dx_logstd = z_logstd[:,:2]
        dx = z[:,:2]*dx_scale # scale dx by standard deviation
        dx = dx.unsqueeze(1)
        z = z[:,2:]

        x = x + dx # translate coordinates

    # reconstruct
    y_hat = p_net(x.contiguous(), z)
    y_hat = y_hat.view(b, -1)

    size = y.size(1)
    log_p_x_g_z = -F.binary_cross_entropy_with_logits(y_hat, y)*size

    # unit normal prior over z and translation
    z_kl = -z_logstd + 0.5*z_std**2 + 0.5*z_mu**2 - 0.5
    kl_div = kl_div + torch.sum(z_kl, 1)
    kl_div = kl_div.mean()
    
    elbo = log_p_x_g_z - kl_div

    return elbo, log_p_x_g_z, kl_div

def train_epoch(iterator, x_coord, p_net, q_net, optim, rotate=True, translate=True
               , dx_scale=0.1, theta_prior=np.pi
               , epoch=1, num_epochs=1, N=1, use_cuda=False):
    p_net.train()
    q_net.train()

    c = 0
    gen_loss_accum = 0
    kl_loss_accum = 0
    elbo_accum = 0

    for y, in iterator:
        b = y.size(0)
        x = Variable(x_coord)
        y = Variable(y)

        elbo, log_p_x_g_z, kl_div = eval_minibatch(x, y, p_net, q_net, rotate=rotate, translate=translate
                                                  , dx_scale=dx_scale, theta_prior=theta_prior
                                                  , use_cuda=use_cuda)

        loss = -elbo
        loss.backward()
        optim.step()
        optim.zero_grad()

        elbo = elbo.item()
        gen_loss = -log_p_x_g_z.item()
        kl_loss = kl_div.item()

        c += b
        delta = b*(gen_loss - gen_loss_accum)
        gen_loss_accum += delta/c

        delta = b*(elbo - elbo_accum)
        elbo_accum += delta/c

        delta = b*(kl_loss - kl_loss_accum)
        kl_loss_accum += delta/c

        template = '# [{}/{}] training {:.1%}, ELBO={:.5f}, Error={:.5f}, KL={:.5f}'
        line = template.format(epoch+1, num_epochs, c/N, elbo_accum, gen_loss_accum
                              , kl_loss_accum)
        print(line, end='\r', file=sys.stderr)

    print(' '*80, end='\r', file=sys.stderr)
    return elbo_accum, gen_loss_accum, kl_loss_accum


def eval_model(iterator, x_coord, p_net, q_net, rotate=True, translate=True
              , dx_scale=0.1, theta_prior=np.pi, use_cuda=False):
    p_net.eval()
    q_net.eval()

    c = 0
    gen_loss_accum = 0
    kl_loss_accum = 0
    elbo_accum = 0

    for y, in iterator:
        b = y.size(0)
        x = Variable(x_coord)
        y = Variable(y)

        elbo, log_p_x_g_z, kl_div = eval_minibatch(x, y, p_net, q_net, rotate=rotate, translate=translate
                                                  , dx_scale=dx_scale, theta_prior=theta_prior
                                                  , use_cuda=use_cuda)

        elbo = elbo.item()
        gen_loss = -log_p_x_g_z.item()
        kl_loss = kl_div.item()

        c += b
        delta = b*(gen_loss - gen_loss_accum)
        gen_loss_accum += delta/c

        delta = b*(elbo - elbo_accum)
        elbo_accum += delta/c

        delta = b*(kl_loss - kl_loss_accum)
        kl_loss_accum += delta/c

    return elbo_accum, gen_loss_accum, kl_loss_accum


def main():
    import argparse

    parser = argparse.ArgumentParser('Train spatial-VAE on MNIST datasets')

    parser.add_argument('--dataset', choices=['mnist', 'mnist-rotated', 'mnist-rotated-translated'], default='mnist-rotated-translated', help='which MNIST datset to train/validate on (default: mnist-rotated-translated)')

    parser.add_argument('-z', '--z-dim', type=int, default=2, help='latent variable dimension (default: 2)')
    parser.add_argument('--hidden-dim', type=int, default=500, help='dimension of hidden layers (default: 512)')
    parser.add_argument('--num-layers', type=int, default=2, help='number of hidden layers (default: 1)')
    parser.add_argument('-a', '--activation', choices=['tanh', 'relu'], default='tanh', help='activation function (default: tanh)')

    parser.add_argument('--vanilla', action='store_true', help='use the standard MLP generator architecture, decoding each pixel with an independent function. disables structured rotation and translation inference')
    parser.add_argument('--no-rotate', action='store_true', help='do not perform rotation inference')
    parser.add_argument('--no-translate', action='store_true', help='do not perform translation inference')

    parser.add_argument('--dx-scale', type=float, default=0.1, help='standard deviation of translation latent variables (default: 0.1)')
    parser.add_argument('--theta-prior', type=float, default=np.pi/4, help='standard deviation on rotation prior (default: pi/4)')

    parser.add_argument('-l', '--learning-rate', type=float, default=1e-4, help='learning rate (default: 0.0001)')
    parser.add_argument('--minibatch-size', type=int, default=100, help='minibatch size (default: 100)')

    parser.add_argument('--save-prefix', help='path prefix to save models (optional)')
    parser.add_argument('--save-interval', default=10, type=int, help='save frequency in epochs (default: 10)')
    parser.add_argument('--num-epochs', type=int, default=100, help='number of training epochs (default: 100)')

    parser.add_argument('-d', '--device', type=int, default=-2, help='compute device to use')

    args = parser.parse_args()
    num_epochs = args.num_epochs

    digits = int(np.log10(num_epochs)) + 1

    ## load the images
    if args.dataset == 'mnist':
        print('# training on MNIST', file=sys.stderr)
        mnist_train = torchvision.datasets.MNIST('data/mnist/', train=True, download=True)
        mnist_test = torchvision.datasets.MNIST('data/mnist/', train=False, download=True)

        array = np.zeros((len(mnist_train),28,28), dtype=np.uint8)
        for i in range(len(mnist_train)):
            array[i] = np.array(mnist_train[i][0], copy=False)
        mnist_train = array

        array = np.zeros((len(mnist_test),28,28), dtype=np.uint8)
        for i in range(len(mnist_test)):
            array[i] = np.array(mnist_test[i][0], copy=False)
        mnist_test = array

    elif args.dataset == 'mnist-rotated':
        print('# training on rotated MNIST', file=sys.stderr)
        mnist_train = np.load('data/mnist_rotated/images_train.npy')
        mnist_test = np.load('data/mnist_rotated/images_test.npy')

    else:
        print('# training on rotated and translated MNIST', file=sys.stderr)
        mnist_train = np.load('data/mnist_rotated_translated/images_train.npy')
        mnist_test = np.load('data/mnist_rotated_translated/images_test.npy')

    mnist_train = torch.from_numpy(mnist_train).float()/255
    mnist_test = torch.from_numpy(mnist_test).float()/255

    n = m = 28

    ## x coordinate array
    xgrid = np.linspace(-1, 1, m)
    ygrid = np.linspace(1, -1, n)
    x0,x1 = np.meshgrid(xgrid, ygrid)
    x_coord = np.stack([x0.ravel(), x1.ravel()], 1)
    x_coord = torch.from_numpy(x_coord).float()

    y_train = mnist_train.view(-1, n*m)
    y_test = mnist_test.view(-1, n*m)

    ## set the device
    d = args.device
    use_cuda = (d != -1) and torch.cuda.is_available()
    if d >= 0:
        torch.cuda.set_device(d)
        print('# using CUDA device:', d, file=sys.stderr)

    if use_cuda:
        y_train = y_train.cuda()
        y_test = y_test.cuda()
        x_coord = x_coord.cuda()

    data_train = torch.utils.data.TensorDataset(y_train)
    data_test = torch.utils.data.TensorDataset(y_test)

    z_dim = args.z_dim
    print('# training with z-dim:', z_dim, file=sys.stderr)

    num_layers = args.num_layers
    hidden_dim = args.hidden_dim
    if args.activation == 'tanh':
        activation = nn.Tanh
    elif args.activation == 'relu':
        activation = nn.LeakyReLU

    if args.vanilla:
        print('# using the vanilla MLP generator architecture', file=sys.stderr)
        p_net = models.VanillaGenerator(n*m, z_dim, hidden_dim, num_layers=num_layers, activation=activation)
        inf_dim = z_dim
        rotate = False
        translate = False
    else:
        print('# using the spatial generator architecture', file=sys.stderr)
        rotate = not args.no_rotate
        translate = not args.no_translate
        inf_dim = z_dim
        if rotate:
            print('# spatial-VAE with rotation inference', file=sys.stderr)
            inf_dim += 1
        if translate:
            print('# spatial-VAE with translation inference', file=sys.stderr)
            inf_dim += 2
        p_net = models.SpatialGenerator(z_dim, hidden_dim, num_layers=num_layers, activation=activation)

    q_net = models.InferenceNetwork(n*m, inf_dim, hidden_dim, num_layers=num_layers, activation=activation)

    if use_cuda:
        p_net.cuda()
        q_net.cuda()

    dx_scale = args.dx_scale
    theta_prior = args.theta_prior

    print('# using priors: theta={}, dx={}'.format(theta_prior, dx_scale), file=sys.stderr)

    N = len(mnist_train)

    params = list(p_net.parameters()) + list(q_net.parameters())

    lr = args.learning_rate
    optim = torch.optim.Adam(params, lr=lr)

    minibatch_size = args.minibatch_size

    train_iterator = torch.utils.data.DataLoader(data_train, batch_size=minibatch_size,
                                                 shuffle=True)
    test_iterator = torch.utils.data.DataLoader(data_test, batch_size=minibatch_size)

    output = sys.stdout
    print('\t'.join(['Epoch', 'Split', 'ELBO', 'Error', 'KL']), file=output)

    path_prefix = args.save_prefix
    save_interval = args.save_interval

    for epoch in range(num_epochs):

        elbo_accum,gen_loss_accum,kl_loss_accum = train_epoch(train_iterator, x_coord, p_net, q_net,
                                                              optim, rotate=rotate, translate=translate,
                                                              dx_scale=dx_scale, theta_prior=theta_prior,
                                                              epoch=epoch, num_epochs=num_epochs, N=N,
                                                              use_cuda=use_cuda)

        line = '\t'.join([str(epoch+1), 'train', str(elbo_accum), str(gen_loss_accum), str(kl_loss_accum)])
        print(line, file=output)
        output.flush()

        # evaluate on the test set
        elbo_accum,gen_loss_accum,kl_loss_accum = eval_model(test_iterator, x_coord, p_net,
                                                             q_net, rotate=rotate, translate=translate,
                                                             dx_scale=dx_scale, theta_prior=theta_prior,
                                                             use_cuda=use_cuda
                                                            )
        line = '\t'.join([str(epoch+1), 'test', str(elbo_accum), str(gen_loss_accum), str(kl_loss_accum)])
        print(line, file=output)
        output.flush()


        ## save the models
        if path_prefix is not None and (epoch+1)%save_interval == 0:
            epoch_str = str(epoch+1).zfill(digits)

            path = path_prefix + '_generator_epoch{}.sav'.format(epoch_str)
            p_net.eval().cpu()
            torch.save(p_net, path)

            path = path_prefix + '_inference_epoch{}.sav'.format(epoch_str)
            q_net.eval().cpu()
            torch.save(q_net, path)

            if use_cuda:
                p_net.cuda()
                q_net.cuda()



if __name__ == '__main__':
    main()

