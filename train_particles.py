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
import spatial_vae.mrc as mrc
import spatial_vae.image as image_utils
import spatial_vae.ctf as C

def eval_minibatch(x, y, mask, ctf, p_net, q_net, rotate=True, translate=True, dx_scale=0.1, theta_prior=np.pi
                  , augment_rotation=False, z_scale=1, use_cuda=False):
    b = y.size(0)
    x = x.expand(b, x.size(0), x.size(1))
    n = int(np.sqrt(y.size(1)))

    # augment training by randomly rotating images by offset
    offset = np.zeros(b)
    y_rot = y
    if rotate and augment_rotation:
        # in order to encourage robustness of the inference network
        # randomly rotate the observed image before doing inference
        y_rot = y.clone()
        offset = np.random.uniform(0, 2*np.pi, size=b)
        if rotate < 1:
            r = np.random.binomial(1, p=rotate, size=b)
            offset *= r
        for i in range(b):
            im = Image.fromarray(y[i].view(n,n).cpu().numpy())
            im = im.rotate(360*offset[i]/2/np.pi, resample=Image.BICUBIC)
            im = torch.from_numpy(np.array(im, copy=False)).to(y.device)
            y_rot[i] = im.view(-1)

    if use_cuda:
        y = y.cuda()
        y_rot = y_rot.cuda()

    # first do inference on the latent variables
    z_mu,z_logstd = q_net(y_rot)
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

        if np.any(offset > 0):
            # invert the random rotation to reconstruct original with rotaion offset
            offset = torch.from_numpy(offset).float().to(z.device)
            theta = theta + offset

        # calculate rotation matrix
        rot = Variable(theta.data.new(b,2,2).zero_())
        rot[:,0,0] = torch.cos(theta)
        rot[:,0,1] = torch.sin(theta)
        rot[:,1,0] = -torch.sin(theta)
        rot[:,1,1] = torch.cos(theta)
        x = torch.bmm(x, rot) # rotate coordinates by theta

        # use modified KL for rotation with no penalty on mean
        sigma = theta_prior
        kl_div = -theta_logstd + np.log(sigma) + theta_std**2/2/sigma**2 - 0.5

    if translate:
        # z[0,1] are the translations
        dx_mu = z_mu[:,:2]
        dx_std = z_std[:,:2]
        dx_logstd = z_logstd[:,:2]
        dx = z[:,:2]*dx_scale # scale dx by standard deviation
        dx = dx.unsqueeze(1)
        z = z[:,2:]

        x = x + dx # translate coordinates

    z = z*z_scale

    # reconstruct
    y_params = p_net(x.contiguous(), z).view(b, -1)

    y_mu = y_params
    y_var = None

    if y_params.size(1) > y.size(1):
        y_mu = y_params[:,:y.size(1)]
        y_logvar = y_params[:,y.size(1):]
        y_var = torch.exp(y_logvar)

    if ctf is not None: # apply the CTF filter
        pad = ctf.size(2)//2
        y_mu = y_mu.view(1, -1, n, n)
        #print(ctf.size(), y_mu.size(), file=sys.stderr)

        y_mu = F.conv2d(y_mu, ctf, padding=pad, groups=ctf.size(0))
        #print(y_mu.size(), file=sys.stderr)
        y_mu = y_mu.view(-1, n*n)

        if y_var is not None:
            y_var = y_var.view(-1, 1, n, n)
            y_var = F.conv2d(y_var, ctf, padding=pad)
            y_var = y_var.view(-1, n*n)

    y = y.view(-1, n*n)
    if mask is not None:
        y = y[:,mask]
        y_mu = y_mu[:,mask]
        if y_var is not None:
            y_var = y_var[:,mask]
            y_logvar = y_logvar[:,mask]

    #print(y.size(), y_mu.size(), file=sys.stderr)

    if y_var is not None:
        log_p_x_g_z = -0.5*torch.sum((y_mu - y)**2/y_var + y_logvar, 1).mean()
    else:
        log_p_x_g_z = -0.5*torch.sum((y_mu - y)**2, 1).mean()

    # unit normal prior over z and translation
    z_kl = -z_logstd + 0.5*z_std**2 + 0.5*z_mu**2 - 0.5
    kl_div = kl_div + torch.sum(z_kl, 1)
    kl_div = kl_div.mean()
    
    elbo = log_p_x_g_z - kl_div

    return elbo, log_p_x_g_z, kl_div


def train_epoch(iterator, x_coord, mask, p_net, q_net, optim, rotate=True, translate=True
               , dx_scale=0.1, theta_prior=np.pi, augment_rotation=False, z_scale=1
               , epoch=1, num_epochs=1, N=1, use_cuda=False):
    p_net.train()
    q_net.train()

    c = 0
    gen_loss_accum = 0
    kl_loss_accum = 0
    elbo_accum = 0

    for mb in iterator:
        if len(mb) > 1:
            y,ctf = mb
        else:
            y = mb[0]
            ctf = None

        b = y.size(0)
        x = Variable(x_coord)
        y = Variable(y)

        elbo, log_p_x_g_z, kl_div = eval_minibatch(x, y, mask, ctf, p_net, q_net, rotate=rotate, translate=translate
                                                  , dx_scale=dx_scale, theta_prior=theta_prior
                                                  , augment_rotation=augment_rotation, z_scale=z_scale
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


def eval_model(iterator, x_coord, mask, p_net, q_net, rotate=True, translate=True
              , dx_scale=0.1, theta_prior=np.pi, z_scale=1, use_cuda=False):
    p_net.eval()
    q_net.eval()

    c = 0
    gen_loss_accum = 0
    kl_loss_accum = 0
    elbo_accum = 0

    for mb in iterator:
        if len(mb) > 1:
            y,ctf = mb
        else:
            y = mb[0]
            ctf = None

        b = y.size(0)
        x = Variable(x_coord)
        y = Variable(y)

        elbo, log_p_x_g_z, kl_div = eval_minibatch(x, y, mask, ctf, p_net, q_net, rotate=rotate, translate=translate
                                                  , dx_scale=dx_scale, theta_prior=theta_prior
                                                  , z_scale=z_scale
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


def load_images(path):
    if path.endswith('mrc') or path.endswith('mrcs'):
        with open(path, 'rb') as f:
            content = f.read()
        images,_,_ = mrc.parse(content)
    elif path.endswith('npy'):
        images = np.load(path)
    return images


class Dataset:
    def __init__(self, y, ctf=None):
        self.y = y
        self.ctf = ctf

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        if self.ctf is None:
            return self.y[i], None
        return self.y[i], self.ctf[i]


def main():
    import argparse

    parser = argparse.ArgumentParser('Train spatial-VAE on particle datasets')

    parser.add_argument('train_path', help='path to training data')
    parser.add_argument('test_path', help='path to testing data')

    parser.add_argument('--ctf-train', help='path to CTF parameters for training images')
    parser.add_argument('--ctf-test', help='path to CTF parameters for testing images')
    parser.add_argument('--scale', default=1, type=float, help='used to scale the ang/pix if images were binned (default: 1)')

    parser.add_argument('-z', '--z-dim', type=int, default=2, help='latent variable dimension (default: 2)')
    parser.add_argument('--p-hidden-dim', type=int, default=500, help='dimension of hidden layers for generator (default: 500)')
    parser.add_argument('--p-num-layers', type=int, default=2, help='number of hidden layers for generator (default: 2)')
    parser.add_argument('--q-hidden-dim', type=int, default=500, help='dimension of hidden layers for inference net (default: 500)')
    parser.add_argument('--q-num-layers', type=int, default=2, help='number of hidden layers for inference net (default: 2)')
    parser.add_argument('-a', '--activation', choices=['tanh', 'relu'], default='tanh', help='activation function (default: tanh)')
    parser.add_argument('--softplus', action='store_true', help='apply softplus activation to mean pixel output by generator. clamping the mean to be non-negative can reduce learning background noise')
    parser.add_argument('--resid', action='store_true', help='use residual connections in networks')
    parser.add_argument('--expand-coords', action='store_true', help='also use the second power of fthe spatial coordinates as features in the spatial generator network')
    parser.add_argument('--bilinear', action='store_true', help='use bilinear layer between coordinate and latent in spatial generator network')

    parser.add_argument('--fit-noise', action='store_true', help='also learn the standard deviation of the noise in the generative model')
    parser.add_argument('--vanilla', action='store_true', help='use the standard MLP generator architecture, decoding each pixel with an independent function. disables structured rotation and translation inference')
    parser.add_argument('--no-rotate', action='store_true', help='do not perform rotation inference')
    parser.add_argument('--no-translate', action='store_true', help='do not perform translation inference')

    parser.add_argument('--dx-scale', type=float, default=0.1, help='standard deviation of translation latent variables (default: 0.1)')
    parser.add_argument('--theta-prior', type=float, default=np.pi, help='standard deviation on rotation prior (default: pi)')

    parser.add_argument('-l', '--learning-rate', type=float, default=1e-4, help='learning rate (default: 0.0001)')
    parser.add_argument('--minibatch-size', type=int, default=100, help='minibatch size (default: 100)')

    parser.add_argument('--augment-rotation', action='store_true', help='use data augmentation by randomly rotating images before inference')
    parser.add_argument('--z-delay', type=int, default=0, help='delay using unstructured latent variables for this many training epochs (default: 0)')

    parser.add_argument('--normalize', action='store_true', help='normalize the images before training')
    parser.add_argument('-c', '--crop', type=int, default=-1, help='crop particles down to this size (default: -1 = unused)')

    parser.add_argument('--save-prefix', help='path prefix to save models (optional)')
    parser.add_argument('--save-interval', default=10, type=int, help='save frequency in epochs (default: 10)')
    parser.add_argument('--num-epochs', type=int, default=100, help='number of training epochs (default: 100)')

    parser.add_argument('-d', '--device', type=int, default=-2, help='compute device to use')
    parser.add_argument('--no-preload', action='store_true', help='do not preload data into GPU RAM')
    parser.add_argument('--mask', action='store_true', help='apply a circular mask to the images')

    args = parser.parse_args()
    num_epochs = args.num_epochs

    digits = int(np.log10(num_epochs)) + 1

    ## load the images
    images_train = load_images(args.train_path)
    images_test = load_images(args.test_path)
    print('# train:', images_train.shape, ', test:', images_test.shape, file=sys.stderr)

    crop = args.crop
    if crop > 0:
        images_train = image_utils.crop(images_train, crop)
        images_test = image_utils.crop(images_test, crop)
        print('# cropped to:', crop, file=sys.stderr)

    n,m = images_train.shape[1:]

    # normalize the images using edges to estimate background
    if args.normalize:
        print('# normalizing particles', file=sys.stderr)
        mu = images_train.reshape(-1, n*m).mean(1)
        std = images_train.reshape(-1, n*m).std(1)
        images_train = (images_train - mu[:,np.newaxis,np.newaxis])/std[:,np.newaxis,np.newaxis]

        mu = images_test.reshape(-1, n*m).mean(1)
        std = images_test.reshape(-1, n*m).std(1)
        images_test = (images_test - mu[:,np.newaxis,np.newaxis])/std[:,np.newaxis,np.newaxis]

        #radius = min(n,m)/2
        #images_train = image_utils.normalize(images_train, radius)
        #images_test = image_utils.normalize(images_test, radius)

    scale = args.scale
    ctf_train = None
    if n % 2 == 0:
        n = n - 1
    if m % 2 == 0:
        m = m - 1
    if args.ctf_train is not None:
        # load CTF params
        print('# loading CTF filters:', args.ctf_train, file=sys.stderr)
        ctf_params = C.parse_ctf(args.ctf_train)
        ctf_train = C.ctf_filter(ctf_params, n, m, scale=scale)
        ctf_train = torch.from_numpy(ctf_train).float().unsqueeze(1)

    ctf_test = None
    if args.ctf_test is not None:
        print('# loading CTF filters:', args.ctf_test, file=sys.stderr)
        ctf_params = C.parse_ctf(args.ctf_test)
        ctf_test = C.ctf_filter(ctf_params, n, m, scale=scale)
        ctf_test = torch.from_numpy(ctf_test).float().unsqueeze(1)

    n,m = images_train.shape[1:]

    ## x coordinate array
    xgrid = np.linspace(-1, 1, m)
    ygrid = np.linspace(1, -1, n)
    x0,x1 = np.meshgrid(xgrid, ygrid)
    x_coord = np.stack([x0.ravel(), x1.ravel()], 1)
    x_coord = torch.from_numpy(x_coord).float()

    images_train = torch.from_numpy(images_train).float()
    images_test = torch.from_numpy(images_test).float()
    y_train = images_train.view(-1, n*m)
    y_test = images_test.view(-1, n*m)

    mask = None
    if args.mask:
        print('# masking particles', file=sys.stderr)
        radius = min(n,m)/2
        y_grid, x_grid = np.ogrid[:n,:m]
        center = np.array([n/2, m/2])
        dist = np.sqrt((center[0] - y_grid)**2 + (center[1] - x_grid)**2)
        mask = torch.from_numpy(dist) < radius
        mask = mask.view(-1)
        print('# masking to size:', mask.sum().item(), file=sys.stderr)

    ## set the device
    d = args.device
    use_cuda = (d != -1) and torch.cuda.is_available()
    if d >= 0:
        torch.cuda.set_device(d)
        print('# using CUDA device:', d, file=sys.stderr)

    no_preload = args.no_preload
    augment_rotation = args.augment_rotation
    if use_cuda and not no_preload:
        y_train = y_train.cuda()
        y_test = y_test.cuda()
        if ctf_train is not None:
            ctf_train = ctf_train.cuda()
        if ctf_test is not None:
            ctf_test = ctf_test.cuda()

    if use_cuda:
        x_coord = x_coord.cuda()
        if mask is not None:
            mask = mask.cuda()

    data_train = torch.utils.data.TensorDataset(y_train)
    if ctf_train is not None:
        data_train = torch.utils.data.TensorDataset(y_train, ctf_train)

    data_test = torch.utils.data.TensorDataset(y_test)
    if ctf_test is not None:
        data_test = torch.utils.data.TensorDataset(y_test, ctf_test)

    z_dim = args.z_dim
    print('# training with z-dim:', z_dim, file=sys.stderr)

    num_layers = args.p_num_layers
    hidden_dim = args.p_hidden_dim
    if args.activation == 'tanh':
        activation = nn.Tanh
    elif args.activation == 'relu':
        activation = nn.LeakyReLU
    resid = args.resid
    expand_coords = args.expand_coords
    bilinear = args.bilinear

    fit_noise = args.fit_noise
    n_out = 1
    if fit_noise:
        n_out = 2
    softplus = args.softplus
    if args.vanilla:
        print('# using the vanilla MLP generator architecture', file=sys.stderr)
        p_net = models.VanillaGenerator(n*m, z_dim, hidden_dim, n_out=n_out, num_layers=num_layers
                                       , activation=activation, softplus=softplus, resid=resid)
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
        p_net = models.SpatialGenerator(z_dim, hidden_dim, n_out=n_out, num_layers=num_layers
                                       , activation=activation, softplus=softplus, resid=resid
                                       , expand_coords=expand_coords, bilinear=bilinear)

    num_layers = args.q_num_layers
    hidden_dim = args.q_hidden_dim
    q_net = models.InferenceNetwork(n*m, inf_dim, hidden_dim, num_layers=num_layers
                                   , activation=activation, resid=resid)

    if use_cuda:
        p_net.cuda()
        q_net.cuda()

    dx_scale = args.dx_scale
    theta_prior = args.theta_prior

    print('# using priors: theta={}, dx={}'.format(theta_prior, dx_scale), file=sys.stderr)

    N = len(data_train)
    params = list(p_net.parameters()) + list(q_net.parameters())

    lr = args.learning_rate
    optim = torch.optim.Adam(params, lr=lr)
    #optim = torch.optim.Adagrad(params, lr=lr)
    #optim = torch.optim.SGD(params, lr=lr, momentum=0.9)
    minibatch_size = args.minibatch_size

    train_iterator = torch.utils.data.DataLoader(data_train, batch_size=minibatch_size,
                                                 shuffle=True)
    test_iterator = torch.utils.data.DataLoader(data_test, batch_size=minibatch_size)

    output = sys.stdout
    print('\t'.join(['Epoch', 'Split', 'ELBO', 'Error', 'KL']), file=output)

    path_prefix = args.save_prefix
    save_interval = args.save_interval

    z_delay = args.z_delay
    for epoch in range(num_epochs):
        z_scale = 1
        if epoch < z_delay:
            z_scale = 0

        elbo_accum,gen_loss_accum,kl_loss_accum = train_epoch(train_iterator, x_coord, mask, p_net, q_net,
                                                              optim, rotate=rotate, translate=translate,
                                                              dx_scale=dx_scale, theta_prior=theta_prior,
                                                              augment_rotation=augment_rotation,
                                                              z_scale=z_scale,
                                                              epoch=epoch, num_epochs=num_epochs, N=N,
                                                              use_cuda=use_cuda)

        line = '\t'.join([str(epoch+1), 'train', str(elbo_accum), str(gen_loss_accum), str(kl_loss_accum)])
        print(line, file=output)
        output.flush()

        # evaluate on the test set
        elbo_accum,gen_loss_accum,kl_loss_accum = eval_model(test_iterator, x_coord, mask, p_net,
                                                             q_net, rotate=rotate, translate=translate,
                                                             dx_scale=dx_scale, theta_prior=theta_prior,
                                                             z_scale=z_scale,
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

