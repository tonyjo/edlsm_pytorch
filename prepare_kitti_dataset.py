from __future__ import print_function
import os
import cv2
import png
import random
import argparse
import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_dir",  type=str, default='./data_scene_flow/training', help="where the dataset is stored")
parser.add_argument("--dataset_name", type=str, default="kitti_2015", choices=["kitti_2012", "kitti_2015"])
parser.add_argument("--dump_root",    type=str, default='./dataset',  help="Where to dump the data")
parser.add_argument("--psz",          type=int, default=18,   help="Patch size")
parser.add_argument("--half_range",   type=int, default=100,  help="Half range size")
parser.add_argument("--img_height",   type=int, default=375,  help="image height")
parser.add_argument("--img_width",    type=int, default=1242, help="image width")
parser.add_argument("--train_set",    type=int, default=160,  help="Set of training images")
parser.add_argument("--val_set",      type=int, default=40,   help="Set of validation images")

args = parser.parse_args()
print('----------------------------------------')
print('FLAGS:')
for arg in vars(args):
    print("'", arg,"'", ": ", getattr(args, arg))
print('----------------------------------------')

# Set seed value of ramdon number generator:
random.seed(a=123)

# Returns shuffled data
def generate_random_permutation(data):
    indices = list(range(len(data)))
    random.shuffle(indices)

    return indices

def find_sum(mat, val_to_ignore=-1):
    # Assuming 2D matrix
    total_values = 0
    for i in mat:
        for j in i:
            if j != val_to_ignore:
                total_values += 1

    return total_values

def read_img(image_path, image_height, image_width):
    reader = png.Reader(image_path)
    pngdata = reader.read()
    I_image = np.array(map(np.uint16, pngdata[2]))

    H, W = I_image.shape
    if H != image_height and W != image_width:
        I_image = cv2.resize(I_image, (image_width, image_height),\
                                          interpolation=cv2.INTER_NEAREST)
    D_image = I_image/ 256.0
    D_image[I_image == 0] = -1

    return D_image

# Generate training data
def gen_training_data(data_root, noc_occ, tr_num, img_h, img_w,\
                      psz, half_range, saveDir):
    num_type  = 2
    num_loc   = np.zeros((1, len(tr_num)), dtype=np.float32)
    num_pixel = np.zeros((1, len(tr_num)), dtype=np.float32)

    for i in range(len(tr_num)):
        image_path = '%s/disp_%s_0/%06d_10.png' % (data_root, noc_occ, tr_num[i])
        tmp_image  = read_img(image_path, img_h, img_w)

        num_loc[0, i]   = find_sum(tmp_image, val_to_ignore=-1)
        num_pixel[0, i] = tmp_image.size

    all_loc = np.zeros((int(num_type * np.sum(num_loc)), 5), dtype=np.float32)

    print('All Possible locations of point(x, y): ', all_loc.shape)

    ty1 = 0;
    ty2 = 0;
    valid_count = 0;
    for idx in range(len(tr_num)):
        image_path = '%s/disp_%s_0/%06d_10.png' % (data_root, noc_occ, tr_num[idx])
        tmp_image  = read_img(image_path, img_h, img_w)

        r, c = np.where(tmp_image != -1)
        #img_h, img_w = tmp_image.shape

        if int(r.size) != num_loc[0, idx]:
            print('Dimensions does not match for image id: ', idx)
            print('Skipping...')

        else:
            for loc in range(len(r)):
                l_center_x = c[loc]
                l_center_y = r[loc]
                r_center_x = c[loc] - tmp_image[r[loc], c[loc]]
                r_center_y = l_center_y

                # Make sure the patch falls inside the image
                ll = l_center_x+psz+1 < img_w and l_center_x-psz > 0 and \
                     l_center_y+psz+1 < img_h and l_center_y-psz > 0

                # Right image-- horizontal
                rr_type1 = r_center_x-half_range-psz > 0 and \
                           r_center_x+half_range+psz+1 < img_w and \
                           r_center_y-psz > 0 and\
                           r_center_y+psz+1 < img_h

                if ll and rr_type1:
                    all_loc[valid_count, :] = [tr_num[idx], 1, l_center_x, l_center_y, r_center_x]
                    ty1 = ty1 + 1;
                    valid_count = valid_count + 1;

                # # Right image--vertical
                # rr_type2 = r_center_y-half_range-psz > 0 and \
                #            r_center_y+half_range+psz <= img_h and \
                #            r_center_x-psz > 0 and r_center_x+psz <= img_w;

                # if ll and rr_type2:
                #     ty2 = ty2 + 1;
                #     valid_count = valid_count + 1;
                #     all_loc(:,valid_count) = [fn_idx(idx); 2; l_center_x; l_center_y; r_center_x];
                #
        print('Completion image: ', idx)

    # Select only the valid positions
    all_loc = all_loc[0:valid_count, :]

    print('Valid locations of Points: ', all_loc.shape)

    # Save to file
    if not os.path.exists(saveDir):
        os.makedirs(saveDir)

    save_file = '%s/tr_%d_%d_%d.txt' % (saveDir, len(tr_num), psz, half_range)

    f = open(save_file, 'w')

    for loc in range(all_loc.shape[0]):
        #f.write('%d, %d, %d, %d, %d' % (loc[0], loc[1], loc[2], loc[3], loc[4]))
        data = all_loc[loc]
        data = data.astype(int)
        data = data.tolist()

        f.write('%d, %d, %d, %d, %d' % (data[0], data[1], data[2], data[3], data[4]))
        f.write('\n')

        if loc != 0 and loc%10000 == 0:
            print('Writing Completion: ', loc)

    f.close()

# Generate validation data
def gen_validation_data(data_root, noc_occ, val_num, img_h, img_w,\
                        psz, half_range, saveDir):
    num_type  = 2
    num_loc   = np.zeros((1, len(val_num)), dtype=np.float32)
    num_pixel = np.zeros((1, len(val_num)), dtype=np.float32)

    for i in range(len(val_num)):
        image_path = '%s/disp_%s_0/%06d_10.png' % (data_root, noc_occ, val_num[i])
        tmp_image  = read_img(image_path, img_h, img_w)

        num_loc[0, i]   = find_sum(tmp_image, val_to_ignore=-1)
        num_pixel[0, i] = tmp_image.size

    all_loc = np.zeros((int(num_type * np.sum(num_loc)), 5), dtype=np.float32)

    print('All Possible locations of point(x, y): ', all_loc.shape)

    ty1 = 0;
    ty2 = 0;
    valid_count = 0;

    for idx in range(len(val_num)):
        image_path = '%s/disp_%s_0/%06d_10.png' % (data_root, noc_occ, val_num[idx])
        tmp_image  = read_img(image_path, img_h, img_w)

        r, c = np.where(tmp_image != -1)
        #img_h, img_w = tmp_image.shape

        if int(r.size) != num_loc[0, idx]:
            print('Dimensions does not match for image id: ', idx)
            print('Skipping...')

        else:
            for loc in range(len(r)):
                l_center_x = c[loc]
                l_center_y = r[loc]
                r_center_x = c[loc] - tmp_image[r[loc], c[loc]]
                r_center_y = l_center_y

                # Make sure the patch falls inside the image
                ll = l_center_x+psz < img_w and l_center_x-psz+1 > 0 and \
                     l_center_y+psz < img_h and l_center_y-psz+1 > 0

                # Right image-- horizontal
                rr_type1 = r_center_x-half_range-psz > 0 and \
                           r_center_x+half_range+psz+1 < img_w and \
                           r_center_y-psz > 0 and\
                           r_center_y+psz+1 < img_h

                if ll and rr_type1:
                    all_loc[valid_count, :] = [val_num[idx], 1, l_center_x, l_center_y, r_center_x]
                    ty1 = ty1 + 1;
                    valid_count = valid_count + 1;

                # # Right image--vertical
                # rr_type2 = r_center_y-half_range-psz > 0 and \
                #            r_center_y+half_range+psz <= img_h and \
                #            r_center_x-psz > 0 and r_center_x+psz <= img_w;

                # if ll and rr_type2:
                #     ty2 = ty2 + 1;
                #     valid_count = valid_count + 1;
                #     all_loc(:,valid_count) = [fn_idx(idx); 2; l_center_x; l_center_y; r_center_x];
                #
        print('Completion image: ', idx)

    # Select only the valid positions
    all_loc = all_loc[0:valid_count, :]

    print('Valid locations of Points: ', all_loc.shape)

    # Save to file
    if not os.path.exists(saveDir):
        os.makedirs(saveDir)

    save_file = '%s/val_%d_%d_%d.txt' % (saveDir, len(val_num), psz, half_range)

    f = open(save_file, 'w')

    for loc in range(all_loc.shape[0]):
        #f.write('%d, %d, %d, %d, %d' % (loc[0], loc[1], loc[2], loc[3], loc[4]))
        data = all_loc[loc]
        data = data.astype(int)
        data = data.tolist()

        f.write('%d, %d, %d, %d, %d' % (data[0], data[1], data[2], data[3], data[4]))
        f.write('\n')

        if loc != 0 and loc%10000 == 0:
            print('Writing Completion: ', loc)

    f.close()

#################################### Main #####################################
data_root    = args.dataset_dir
noc_occ      = 'noc'
trainDir     = args.dump_root
total_data   = os.listdir(data_root + '/disp_noc_0')
psz          = args.psz
half_range   = args.half_range
val_set      = args.val_set
train_set    = args.train_set
image_width  = args.img_width
image_height = args.img_height

# Randomize dataset
rdn_data = generate_random_permutation(total_data)

# Split the dataset into training and testing
train_num = rdn_data[0:train_set]
valid_num = rdn_data[train_set:]

# Generate training data
gen_training_data(data_root, noc_occ, train_num, image_height, image_width,\
                 psz, half_range, trainDir)

# Generate Validation data
gen_validation_data(data_root, noc_occ, valid_num, image_height, image_width,\
                    psz, half_range, trainDir)
