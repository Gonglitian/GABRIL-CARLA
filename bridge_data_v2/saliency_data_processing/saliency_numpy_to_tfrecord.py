"""
Converts data from the (augmented) BridgeData numpy format to TFRecord format,
including optional gaze heatmaps produced by saliency_raw_to_numpy.py.

This mirrors bridge_data_v2/data_processing/bridgedata_numpy_to_tfrecord.py but
adds support for the keys:
  - observations/gaze: float32 [T,H,W,1]
  - next_observations/gaze: float32 [T,H,W,1]

If a trajectory lacks these keys, they are simply omitted from the TFRecord.
"""
import os
from multiprocessing import Pool

import numpy as np
import tensorflow as tf
import tqdm
from absl import app, flags, logging

FLAGS = flags.FLAGS

flags.DEFINE_string("input_path", None, "Input path", required=True)
flags.DEFINE_string("output_path", None, "Output path", required=True)
flags.DEFINE_integer(
    "depth",
    5,
    "Number of directories deep to traverse. Looks for {input_path}/dir_1/dir_2/.../dir_{depth-1}/train/out.npy",
)
flags.DEFINE_bool("overwrite", False, "Overwrite existing files")
flags.DEFINE_integer("num_workers", 8, "Number of threads to use")


def tensor_feature(value):
    return tf.train.Feature(
        bytes_list=tf.train.BytesList(value=[tf.io.serialize_tensor(value).numpy()])
    )


def process(path):
    with tf.io.gfile.GFile(path, "rb") as f:
        arr = np.load(f, allow_pickle=True)
    dirname = os.path.dirname(os.path.abspath(path))
    outpath = os.path.join(FLAGS.output_path, *dirname.split(os.sep)[-FLAGS.depth :])
    outpath = f"{outpath}/out.tfrecord"

    if tf.io.gfile.exists(outpath):
        if FLAGS.overwrite:
            logging.info(f"Deleting {outpath}")
            tf.io.gfile.rmtree(outpath)
        else:
            logging.info(f"Skipping {outpath}")
            return

    if len(arr) == 0:
        logging.info(f"Skipping {path}, empty")
        return

    tf.io.gfile.makedirs(os.path.dirname(outpath))

    with tf.io.TFRecordWriter(outpath) as writer:
        for traj in arr:
            truncates = np.zeros(len(traj["actions"]), dtype=np.bool_)
            truncates[-1] = True

            features = {
                "observations/images0": tensor_feature(
                    np.array([o["images0"] for o in traj["observations"]], dtype=np.uint8)
                ),
                "observations/state": tensor_feature(
                    np.array([o["state"] for o in traj["observations"]], dtype=np.float32)
                ),
                "next_observations/images0": tensor_feature(
                    np.array(
                        [o["images0"] for o in traj["next_observations"]],
                        dtype=np.uint8,
                    )
                ),
                "next_observations/state": tensor_feature(
                    np.array(
                        [o["state"] for o in traj["next_observations"]],
                        dtype=np.float32,
                    )
                ),
                "language": tensor_feature(traj["language"]),
                "actions": tensor_feature(np.array(traj["actions"], dtype=np.float32)),
                "terminals": tensor_feature(np.zeros(len(traj["actions"]), dtype=np.bool_)),
                "truncates": tensor_feature(truncates),
            }

            # Optional gaze fields
            try:
                obs_gaze = np.array(
                    [o["gaze"] for o in traj["observations"]], dtype=np.float32
                )
                if obs_gaze.ndim == 3:
                    obs_gaze = obs_gaze[..., None]
                features["observations/gaze"] = tensor_feature(obs_gaze)
            except Exception:
                pass
            try:
                nxt_gaze = np.array(
                    [o["gaze"] for o in traj["next_observations"]], dtype=np.float32
                )
                if nxt_gaze.ndim == 3:
                    nxt_gaze = nxt_gaze[..., None]
                features["next_observations/gaze"] = tensor_feature(nxt_gaze)
            except Exception:
                pass

            example = tf.train.Example(features=tf.train.Features(feature=features))
            writer.write(example.SerializeToString())


def main(_):
    assert FLAGS.depth >= 1

    paths = tf.io.gfile.glob(
        tf.io.gfile.join(FLAGS.input_path, *("*" * (FLAGS.depth - 1)))
    )
    paths = [f"{p}/train/out.npy" for p in paths] + [f"{p}/val/out.npy" for p in paths]
    with Pool(FLAGS.num_workers) as p:
        list(tqdm.tqdm(p.imap(process, paths), total=len(paths)))


if __name__ == "__main__":
    app.run(main)

