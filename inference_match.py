from __future__ import print_function
import os
import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms
from PIL import Image
import matplotlib.pyplot as plt
from src.nets_test import *

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_dir",     type=str, default='./data_scene_flow/training', help="where the dataset is stored")
parser.add_argument("--save_root",       type=str, default='./dataset', help="Where to dump the data")
parser.add_argument("--checkpoint_dir",  type=str, default='./saved_models/kitti_b128_3pxloss', help="Where the ckpt files are")
parser.add_argument("--checkpoint_file", type=str, default='edlsm_38000.ckpt', help="checkpoint file name to load")
parser.add_argument("--resize_image",    type=str, default='True', help="Resize image")
parser.add_argument("--test_num",        type=int, default=80,   help="Image number to do inference")
parser.add_argument("--disp_range",      type=int, default=128,  help="Search range for disparity")
parser.add_argument("--use_gpu",         type=int, default=1,    help="Check to use GPU")

args = parser.parse_args()
print('----------------------------------------')
print('FLAGS:')
for arg in vars(args):
    print("'", arg,"'", ": ", getattr(args, arg))
print('----------------------------------------')
print('Inference....')

# Useful functions
def load_and_resize_l_and_r_image(test_num):
    # Load the image
    l_image_path = os.path.join(args.dataset_dir, 'image_2/%06d_10.png' % (test_num))
    r_image_path = os.path.join(args.dataset_dir, 'image_3/%06d_10.png' % (test_num))
    ll_image1 = Image.open(l_image_path)
    ll_image1 = ll_image1.convert('RGB')
    rr_image1 = Image.open(r_image_path)
    rr_image1 = rr_image1.convert('RGB')

    ll_image1 = np.array(ll_image1)
    rr_image1 = np.array(rr_image1)

    ll_image = 255*transforms.ToTensor()(ll_image1)
    rr_image = 255*transforms.ToTensor()(rr_image1)

    return ll_image, rr_image, ll_image1, rr_image1

def load_disp_img(test_num):
    image_path = '%s/disp_%s_0/%06d_10.png' % ('./data_scene_flow/training', 'noc', test_num)
    reader = png.Reader(image_path)
    pngdata = reader.read()
    I_image = np.array(map(np.uint16, pngdata[2]))

    D_image = I_image / 256.0

    return D_image

#################################### Main #####################################
# Input Channels
nChannel = 3

# Search range
disp_range = args.disp_range

# Trained model file
model_fn = os.path.join(args.checkpoint_dir, args.checkpoint_file)

# Build Test Graph
net = Net(nChannel)
# Loading the trained model
net.load_state_dict(torch.load(model_fn))
net.eval()
print(net)
print('Model Loaded')

# Check to use GPU
if args.use_gpu:
    net = net.cuda()

# Load the images
ll_image, rr_image, ll_image1, rr_image1 = load_and_resize_l_and_r_image(args.test_num)

# Normalize images. All the patches used for training were normalized.
l_img = (ll_image - ll_image.mean())/(ll_image.std())
r_img = (rr_image - rr_image.mean())/(rr_image.std())

img_h = l_img.size(1)
img_w = l_img.size(2)
print('Image size:', img_h, img_w)

# Convert to batch x channel x height x width format
l_img = l_img.view(1, l_img.size(0), l_img.size(1), l_img.size(2))
r_img = r_img.view(1, r_img.size(0), r_img.size(1), r_img.size(2))

if args.use_gpu:
    l_img = l_img.cuda()
    r_img = r_img.cuda()

# Forward pass. extract deep features
left_feat = net(Variable(l_img, requires_grad=False))
# forward pass right image
right_feat = net(Variable(r_img, requires_grad=False))

# output tensor
output = torch.Tensor(img_h, img_w, disp_range).zero_()

start_id = 0
end_id = img_w -1
total_loc = disp_range

# Output tensor
unary_vol = torch.Tensor(img_h, img_w, total_loc).zero_()
right_unary_vol = torch.Tensor(img_h, img_w, total_loc).zero_()

while start_id <= end_id:
    for loc_idx in range(0, total_loc):
        x_off = -loc_idx + 1 # always <= 0
        if end_id+x_off >= 1 and img_w >= start_id+x_off:
            l =  left_feat[:, :, :, np.max([start_id, -x_off+1]): np.min([end_id, img_w-x_off])]
            r = right_feat[:, :, :, np.max([1, x_off+start_id]) : np.min([img_w, end_id+x_off])]

            p = torch.mul(l,r)
            q = torch.sum(p, 1)

            unary_vol[:, np.max([start_id, -x_off+1]): np.min([end_id, img_w-x_off]) ,loc_idx] = q.data.view(q.data.size(1), q.data.size(2))
            right_unary_vol[:, np.max([1, x_off+start_id]) : np.min([img_w, end_id+x_off]) ,loc_idx] = q.data.view(q.data.size(1), q.data.size(2))

    start_id = end_id + 1

#misc.imsave('pred_disp_' + str(test_img_num) + '.png', pred_disp)

max_disp1, pred_1 = torch.max(unary_vol, 2)
max_disp2, pred_2 = torch.max(right_unary_vol, 2)

# image_path_1 = '%s/cost_img/%06d_10.t7' % ('./save_disp', test_img_num)
# image_path_2 = '%s/cost_img_r/%06d_10.t7' % ('./save_disp', test_img_num)

# torch.save(unary_vol, image_path_1)
# torch.save(right_unary_vol, image_path_2)

# disparity map (height x width)
pred_disp1 = pred_1.view(unary_vol.size(0), unary_vol.size(1))
pred_disp2 = pred_2.view(unary_vol.size(0), unary_vol.size(1))

# Display the images
plt.subplot(411)
plt.imshow(ll_image1)
plt.title('Left Image')
plt.axis('off')
plt.subplot(412)
plt.imshow(rr_image1)
plt.title('Right Image')
plt.axis('off')
plt.subplot(413)
plt.imshow(pred_disp1, cmap='gray')
plt.title('Predicted Disparity')
plt.axis('off')
plt.subplot(414)
plt.imshow(pred_disp2, cmap='gray')
plt.title('Right Disparity')
plt.axis('off')
plt.show()

print('Complete!')
