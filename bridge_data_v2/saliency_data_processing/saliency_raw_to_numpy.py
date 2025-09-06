"""
Converts BridgeData raw format to numpy, additionally generating per-frame
gaze heatmaps from VLM-filtered or unfiltered bbox pickle files.

Directory structure (same as bridgedata_raw_to_numpy.py), but with extra files
present per trajectory:
    .../raw/traj_group*/traj*/
        filter.pkl      # VLM-filtered boxes per frame
        no_filter.pkl   # Unfiltered boxes per frame

Pickle format:
    List[List[float]] per frame, each bbox [x1, y1, x2, y2] in normalized [0,1].

This script mirrors bridge_data_v2/data_processing/bridgedata_raw_to_numpy.py
but adds a "gaze" key under observations and next_observations containing
float32 heatmaps of shape [H, W, 1], aligned with resized images (H=W=im_size).

Usage example:
python saliency_raw_to_numpy.py \
  --input_path /data3/vla-reasoning/dataset/bdv2 \
  --output_path /tmp/bridgedata_numpy_gaze \
  --depth 5 --im_size 128 --bbox_source filter
"""

import glob
import os
import pickle
from collections import defaultdict
from datetime import datetime
from functools import partial
from multiprocessing import Pool

import numpy as np
import tensorflow as tf
import tqdm
from absl import app, flags, logging
from PIL import Image

FLAGS = flags.FLAGS

flags.DEFINE_string("input_path", None, "Input path", required=True)
flags.DEFINE_string("output_path", None, "Output path", required=True)
flags.DEFINE_integer(
    "depth",
    5,
    "Number of directories deep to traverse to the dated directory. Looks for "
    "{input_path}/dir_1/dir_2/.../dir_{depth-1}/2022-01-01_00-00-00/...",
)
flags.DEFINE_bool("overwrite", False, "Overwrite existing files")
flags.DEFINE_float(
    "train_proportion", 0.9, "Proportion of data to use for training (rather than val)"
)
flags.DEFINE_integer("num_workers", 8, "Number of threads to use")
flags.DEFINE_integer("im_size", 128, "Image size (square)")
flags.DEFINE_enum(
    "bbox_source",
    "filter",
    ["filter", "no_filter"],
    "Which bbox pickle to use: 'filter' or 'no_filter'",
)

# Gaze mask hyperparameters (temporal multi-modal Gaussian per Eq. (1))
flags.DEFINE_integer(
    "gaze_k",
    2,
    "Temporal half-window k: include past/future frames in [-k, k]",
)
flags.DEFINE_float(
    "gaze_alpha",
    0.7,
    "Intensity decay α in (0,1); weight = α^{|j|} for offset j",
)
flags.DEFINE_float(
    "gaze_beta",
    0.7,
    "Radius growth β in (0,1); sigma = γ * β^{-abs(j)}",
)
flags.DEFINE_float(
    "gaze_gamma_frac",
    0.06,
    "Base Gaussian std γ as fraction of image size (pixels).",
)


def squash(path):  # squash from 480x640 to im_size
    im = Image.open(path)
    im = im.resize((FLAGS.im_size, FLAGS.im_size), Image.Resampling.LANCZOS)
    out = np.asarray(im).astype(np.uint8)
    return out


def process_images(path):  # processes images at a trajectory level
    names = sorted(
        [x for x in os.listdir(path) if "images" in x and not "depth" in x],
        key=lambda x: int(x.split("images")[1]),
    )
    image_path = [
        os.path.join(path, x)
        for x in os.listdir(path)
        if "images" in x and not "depth" in x
    ]
    image_path = sorted(image_path, key=lambda x: int(x.split("images")[1]))

    images_out = defaultdict(list)

    tlen = len(glob.glob(image_path[0] + "/im_*.jpg"))

    for i, name in enumerate(names):
        for t in range(tlen):
            images_out[name].append(squash(image_path[i] + f"/im_{t}.jpg"))

    images_out = dict(images_out)

    obs, next_obs = dict(), dict()

    for n in names:
        obs[n] = images_out[n][:-1]
        next_obs[n] = images_out[n][1:]
    return obs, next_obs


def load_bboxes(traj_dir: str, source: str):
    pkl_name = "filter.pkl" if source == "filter" else "no_filter.pkl"
    fp = os.path.join(traj_dir, pkl_name)
    if not os.path.exists(fp):
        logging.warning("%s not found in %s; returning None", pkl_name, traj_dir)
        return None
    with open(fp, "rb") as f:
        data = pickle.load(f)
    return data  # list per frame: [[x1,y1,x2,y2], ...]


def boxes_to_heatmaps(boxes_per_frame, im_size: int):
    """Convert per-frame list of normalized boxes to [T, H, W, 1] float32.

    Implements the multimodal temporal Gaussian gaze mask:
        g_i_bar = sum_{j=-k..k} α^{|j|} N([x_{i+j},y_{i+j}], (γ^2 β^{-2|j|}) I)
    normalized to [0,1] per-frame.

    We approximate gaze points by the centers of bboxes at each frame.
    Boxes are [x1,y1,x2,y2] in [0,1].
    """
    if boxes_per_frame is None:
        return None

    T = len(boxes_per_frame)
    H = W = im_size
    k = int(FLAGS.gaze_k)
    alpha = float(FLAGS.gaze_alpha)
    beta = float(FLAGS.gaze_beta)
    gamma_px = max(1.0, float(FLAGS.gaze_gamma_frac) * im_size)

    # Precompute meshgrid for Gaussian evaluation
    yy, xx = np.meshgrid(np.arange(H, dtype=np.float32), np.arange(W, dtype=np.float32), indexing="ij")

    # Precompute bbox centers per frame in pixel coords
    centers = []  # list of list of (cx, cy) in pixels
    for t in range(T):
        pts = []
        frame_boxes = boxes_per_frame[t] or []
        for box in frame_boxes:
            x1, y1, x2, y2 = box
            # clamp to [0,1]
            x1 = float(np.clip(x1, 0.0, 1.0))
            y1 = float(np.clip(y1, 0.0, 1.0))
            x2 = float(np.clip(x2, 0.0, 1.0))
            y2 = float(np.clip(y2, 0.0, 1.0))
            if x2 <= x1 or y2 <= y1:
                continue
            cx = 0.5 * (x1 + x2) * (W - 1)
            cy = 0.5 * (y1 + y2) * (H - 1)
            pts.append((cx, cy))
        centers.append(pts)

    heatmaps = np.zeros((T, H, W, 1), dtype=np.float32)

    for i in range(T):
        acc = np.zeros((H, W), dtype=np.float32)
        # accumulate from temporal neighborhood
        for j in range(-k, k + 1):
            t = i + j
            if t < 0 or t >= T:
                continue
            pts = centers[t]
            if not pts:
                continue
            sigma = gamma_px * (beta ** (-abs(j)))
            if sigma < 1e-6:
                continue
            denom = (2.0 * (sigma ** 2))
            w = (alpha ** abs(j))
            norm_const = 1.0 / (2.0 * np.pi * (sigma ** 2))
            for (cx, cy) in pts:
                # Normalized 2D Gaussian PDF with std=sigma
                dx2 = (xx - cx) ** 2
                dy2 = (yy - cy) ** 2
                g = norm_const * np.exp(-(dx2 + dy2) / denom)
                acc += w * g

        # Normalize to [0,1]
        mx = float(acc.max())
        if mx > 0:
            acc = acc / mx
        heatmaps[i, :, :, 0] = acc.astype(np.float32)

    return heatmaps


def process_actions(path):
    with open(os.path.join(path, "policy_out.pkl"), "rb") as f:
        out = pickle.load(f)
    # extract actions
    act = out
    return act


def process_state(path):
    fp = os.path.join(path, "obs_dict.pkl")
    with open(fp, "rb") as f:
        x = pickle.load(f)
    return x["full_state"][:-1], x["full_state"][1:]


def process_time(path):
    fp = os.path.join(path, "obs_dict.pkl")
    with open(fp, "rb") as f:
        x = pickle.load(f)
    return x["time_stamp"][:-1], x["time_stamp"][1:]


def process_dc(path, train_ratio=0.9):
    all_dicts_train = []
    all_dicts_test = []
    all_rews_train = []
    all_rews_test = []

    date_time = datetime.strptime(path.split("/")[-1], "%Y-%m-%d_%H-%M-%S")
    latency_shift = date_time < datetime(2021, 7, 23)

    search_path = os.path.join(path, "raw", "traj_group*", "traj*")
    all_traj = glob.glob(search_path)
    if all_traj == []:
        logging.info(f"no trajs found in {search_path}")
        return [], [], [], []

    for itraj, tp in tqdm.tqdm(enumerate(all_traj)):
        try:
            out = dict()

            ld = os.listdir(tp)
            assert "obs_dict.pkl" in ld, tp + ":" + str(ld)
            assert "policy_out.pkl" in ld, tp + ":" + str(ld)

            obs, next_obs = process_images(tp)
            acts = process_actions(tp)
            state, next_state = process_state(tp)
            time_stamp, next_time_stamp = process_time(tp)

            # Gaze heatmaps
            boxes = load_bboxes(tp, FLAGS.bbox_source)
            gaze = boxes_to_heatmaps(boxes, FLAGS.im_size) if boxes is not None else None
            if gaze is not None and gaze.shape[0] >= 2:
                obs["gaze"] = list(gaze[:-1])
                next_obs["gaze"] = list(gaze[1:])

            term = [0] * len(acts)
            if "lang.txt" in ld:
                with open(os.path.join(tp, "lang.txt")) as f:
                    lang = list(f)
                    lang = [l.strip() for l in lang if "confidence" not in l]
            else:
                lang = [""]

            out["observations"] = obs
            out["observations"]["state"] = state
            out["observations"]["time_stamp"] = time_stamp
            out["next_observations"] = next_obs
            out["next_observations"]["state"] = next_state
            out["next_observations"]["time_stamp"] = next_time_stamp

            out["observations"] = [
                dict(zip(out["observations"], t))
                for t in zip(*out["observations"].values())
            ]
            out["next_observations"] = [
                dict(zip(out["next_observations"], t))
                for t in zip(*out["next_observations"].values())
            ]

            out["actions"] = acts
            out["terminals"] = term
            out["language"] = lang

            # latency shift: shift observations and associated fields by 1
            if latency_shift:
                out["observations"] = out["observations"][1:]
                out["next_observations"] = out["next_observations"][1:]
                out["actions"] = out["actions"][:-1]
                out["terminals"] = term[:-1]

            labeled_rew = np.array(out["terminals"]).astype(int).tolist()
            if len(labeled_rew) >= 2:
                labeled_rew[-2:] = [1, 1]

            traj_len = len(out["observations"])
            assert len(out["next_observations"]) == traj_len
            assert len(out["actions"]) == traj_len
            assert len(out["terminals"]) == traj_len

            if itraj < int(len(all_traj) * train_ratio):
                all_dicts_train.append(out)
                all_rews_train.append(labeled_rew)
            else:
                all_dicts_test.append(out)
                all_rews_test.append(labeled_rew)
        except FileNotFoundError as e:
            logging.error(e)
            continue
        except AssertionError as e:
            logging.error(e)
            continue

    return all_dicts_train, all_dicts_test, all_rews_train, all_rews_test


def make_numpy(path, train_proportion):
    dirname = os.path.abspath(path)
    outpath = os.path.join(
        FLAGS.output_path, *dirname.split(os.sep)[-(max(FLAGS.depth - 1, 1)) :]
    )

    if os.path.exists(outpath):
        if FLAGS.overwrite:
            logging.info(f"Deleting {outpath}")
            tf.io.gfile.rmtree(outpath)
        else:
            logging.info(f"Skipping {outpath}")
            return

    outpath_train = tf.io.gfile.join(outpath, "train")
    outpath_val = tf.io.gfile.join(outpath, "val")
    tf.io.gfile.makedirs(outpath_train)
    tf.io.gfile.makedirs(outpath_val)

    lst_train = []
    lst_val = []
    rew_train_l = []
    rew_val_l = []

    # Each 'path' is a directory that directly contains dated directories
    for dated_folder in os.listdir(path):
        curr_train, curr_val, rew_train, rew_val = process_dc(
            os.path.join(path, dated_folder), train_ratio=FLAGS.train_proportion
        )
        lst_train.extend(curr_train)
        lst_val.extend(curr_val)
        rew_train_l.extend(rew_train)
        rew_val_l.extend(rew_val)

    if len(lst_train) == 0 or len(lst_val) == 0:
        return

    with tf.io.gfile.GFile(tf.io.gfile.join(outpath_train, "out.npy"), "wb") as f:
        np.save(f, lst_train)
    with tf.io.gfile.GFile(tf.io.gfile.join(outpath_val, "out.npy"), "wb") as f:
        np.save(f, lst_val)


def main(_):
    assert FLAGS.depth >= 1
    # each path is a directory that contains dated directories
    paths = glob.glob(os.path.join(FLAGS.input_path, *("*" * (FLAGS.depth - 1))))

    worker_fn = partial(make_numpy, train_proportion=FLAGS.train_proportion)

    with Pool(FLAGS.num_workers) as p:
        list(tqdm.tqdm(p.imap(worker_fn, paths), total=len(paths)))


if __name__ == "__main__":
    app.run(main)
