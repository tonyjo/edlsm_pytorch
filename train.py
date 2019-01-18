import os
import pprint
import tensorflow as tf
from src.edlsm_learner import edlsmLearner

flags = tf.app.flags
flags.DEFINE_string("directory", "./data_scene_flow/training", "Directory to the dataset")
flags.DEFINE_string("train_val_split_dir", "./dataset", "Directory to the dataset")
flags.DEFINE_string("train_dataset_name", "tr_18_18_100.txt", "Training set")
flags.DEFINE_string("val_dataset_name",   "val_5_18_100.txt", "Validation set")
flags.DEFINE_string("checkpoint_dir", "./checkpoints", "Directory name to save the checkpoints")
flags.DEFINE_string("logs_path", "logs", "Tensorboard log path")
flags.DEFINE_bool("continue_train", True, "Resume training")
flags.DEFINE_bool("gpu", True, "Use GPU if available")
flags.DEFINE_string("init_checkpoint_file", "edlsm_24000.ckpt", "checkpoint file")
flags.DEFINE_integer("batch_size", 128, "The size of of a sample batch")
flags.DEFINE_integer("psz", 18, "Left patch size")
flags.DEFINE_integer("half_range", 100, "Right patch half range")
flags.DEFINE_integer("image_width", 1238, "Image width, must same as prepare_dataset")
flags.DEFINE_integer("image_height", 375, "Image height, must same as prepare_dataset")
flags.DEFINE_integer("start_step",24000,    "Starting training step")
flags.DEFINE_integer("max_steps", 50000, "Maximum number of training iterations")
flags.DEFINE_float("l_rate", 0.01, "learning rate")
flags.DEFINE_float("l2", 0.0005, "Weight Decay")
flags.DEFINE_integer("reduce_l_rate_itr", 8000, "Reduce learning rate after this many iterations")
flags.DEFINE_float("pxl_wghts", [[1.0, 4.0, 10.0, 4.0, 1.0]], "Weights for three pixel error")
flags.DEFINE_integer("summary_freq", 200, "Logging every summary_freq iterations")
flags.DEFINE_integer("valid_freq",   500, "Logging every valid_freq iterations")
flags.DEFINE_integer("save_latest_freq", 2000, \
                       "Save the latest model every save_latest_freq iterations")
FLAGS = flags.FLAGS

def main(_):
    pp = pprint.PrettyPrinter()
    pp.pprint(flags.FLAGS.__flags)

    if not os.path.exists(FLAGS.checkpoint_dir):
        os.makedirs(FLAGS.checkpoint_dir)

    edlsm = edlsmLearner()
    edlsm.train(FLAGS)


if __name__ == '__main__':
    tf.app.run()
