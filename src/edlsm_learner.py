from __future__ import print_function
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from nets import *
# from data_Loader import *
from data_Loader_middleburry import *

class edlsmLearner(object):
    def __init__(self):
        pass

    def loss_function(self, x3, t, w):
        # Three pixel error
        error = 0
        for i in range(x3.size(0)):
            sc = x3[i,t[i][0]-2:t[i][0]+2+1]
            loss_sample = torch.mul(sc, w).sum()

            error = error - loss_sample

        return error

    def train(self, opt):
        # Load Flags
        self.opt = opt

        # Type
        dtype = torch.FloatTensor

        # Data Loader
        train_loader = dataLoader(opt.directory, opt.train_val_split_dir,\
                                  opt.train_dataset_name, opt.psz,opt.half_range,\
                                  opt.image_height, opt.image_width, mode='Train')
        train_gen = train_loader.gen_data_batch(batch_size=opt.batch_size)

        # Target labels
        target  = opt.half_range # + 1
        targets = np.tile(target, (opt.batch_size, 1))
        t_batch = torch.tensor(targets, dtype=torch.int32)

        # Build training graph
        model = Net(3, opt.half_range*2+1)
        pxl_wghts = opt.pxl_wghts/np.sum(opt.pxl_wghts)
        pxl_wghts = Variable(torch.Tensor(pxl_wghts))
        if opt.gpu:
            model = model.cuda()
            pxl_wghts = pxl_wghts.cuda()
        model.train()

        # Check if training has to be continued
        if opt.continue_train:
            if opt.init_checkpoint_file is None:
                print('Enter a valid checkpoint file')
            else:
                load_model = os.path.join(opt.checkpoint_dir, opt.init_checkpoint_file)
            print("Resume training from previous checkpoint: %s" % opt.init_checkpoint_file)
            model.load_state_dict(torch.load(load_model))

        # Optimizer
        optimizer = optim.Adam(model.parameters(), lr=opt.l_rate, eps=1e-08, weight_decay=opt.l2)

        # Begin Training
        for step in range(opt.start_step, opt.max_steps):
            # Sample batch data
            left_batch, right_batch = next(train_gen)

            # Zero the gradient buffers
            optimizer.zero_grad()

            # Convert to cuda if GPU available
            if opt.gpu:
                left_batch  = left_batch.cuda()
                right_batch = right_batch.cuda()
                t_batch     = t_batch.cuda()

            # Forward pass
            x1, x2, x3 = model(Variable(left_batch), Variable(right_batch))

            # Compute loss
            loss = self.loss_function(x3, t_batch, pxl_wghts)

            # Backward pass
            loss.backward()

            # Update the weights
            optimizer.step()

            if step%opt.save_latest_freq == 0:
                model_name = 'edlsm_' + str(step) + '.ckpt'
                checkpoint_path = os.path.join(opt.checkpoint_dir, model_name)
                torch.save(model.state_dict(), checkpoint_path)

            if step%opt.summary_freq == 0:
                print('Step Loss: ', loss.data.cpu().numpy()/opt.batch_size, ' at iteration: ', step)

        # Save the latest
        model_name = 'edlsm_latest' + str(i) + '.ckpt'
        checkpoint_path = os.path.join(opt.checkpoint_dir, model_name)
        torch.save(model.state_dict(), checkpoint_path)
        print("Training Complete and latest checkpoint saved!")
