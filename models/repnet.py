# for repnet
import base64
import io
from PIL import Image
from scipy.signal import medfilt
import tensorflow as tf
import numpy as np
import cv2
import os
# Model definition
layers = tf.keras.layers
regularizers = tf.keras.regularizers


class ResnetPeriodEstimator(tf.keras.models.Model):
  """RepNet model."""

  def __init__(
      self,
      num_frames=64,
      image_size=112,
      base_model_layer_name='conv4_block3_out',
      temperature=13.544,
      dropout_rate=0.25,
      l2_reg_weight=1e-6,
      temporal_conv_channels=512,
      temporal_conv_kernel_size=3,
      temporal_conv_dilation_rate=3,
      conv_channels=32,
      conv_kernel_size=3,
      transformer_layers_config=((512, 4, 512),),
      transformer_dropout_rate=0.0,
      transformer_reorder_ln=True,
      period_fc_channels=(512, 512),
      within_period_fc_channels=(512, 512)):
    super(ResnetPeriodEstimator, self).__init__()

    # Model params.
    self.num_frames = num_frames
    self.image_size = image_size

    self.base_model_layer_name = base_model_layer_name

    self.temperature = temperature

    self.dropout_rate = dropout_rate
    self.l2_reg_weight = l2_reg_weight

    self.temporal_conv_channels = temporal_conv_channels
    self.temporal_conv_kernel_size = temporal_conv_kernel_size
    self.temporal_conv_dilation_rate = temporal_conv_dilation_rate

    self.conv_channels = conv_channels
    self.conv_kernel_size = conv_kernel_size
    # Transformer config in form of (channels, heads, bottleneck channels).
    self.transformer_layers_config = transformer_layers_config
    self.transformer_dropout_rate = transformer_dropout_rate
    self.transformer_reorder_ln = transformer_reorder_ln

    self.period_fc_channels = period_fc_channels
    self.within_period_fc_channels = within_period_fc_channels

    # Base ResNet50 Model.
    base_model = tf.keras.applications.ResNet50V2(
        include_top=False, weights=None, pooling='max')
    self.base_model = tf.keras.models.Model(
        inputs=base_model.input,
        outputs=base_model.get_layer(self.base_model_layer_name).output)

    # 3D Conv on k Frames
    self.temporal_conv_layers = [
        layers.Conv3D(self.temporal_conv_channels,
                      self.temporal_conv_kernel_size,
                      padding='same',
                      dilation_rate=(self.temporal_conv_dilation_rate, 1, 1),
                      kernel_regularizer=regularizers.l2(self.l2_reg_weight),
                      kernel_initializer='he_normal')]
    self.temporal_bn_layers = [layers.BatchNormalization()
                               for _ in self.temporal_conv_layers]

    # Counting Module (Self-sim > Conv > Transformer > Classifier)
    self.conv_3x3_layer = layers.Conv2D(self.conv_channels,
                                        self.conv_kernel_size,
                                        padding='same',
                                        activation=tf.nn.relu)

    channels = self.transformer_layers_config[0][0]
    self.input_projection = layers.Dense(
        channels, kernel_regularizer=regularizers.l2(self.l2_reg_weight),
        activation=None)
    self.input_projection2 = layers.Dense(
        channels, kernel_regularizer=regularizers.l2(self.l2_reg_weight),
        activation=None)

    length = self.num_frames
    self.pos_encoding = tf.compat.v1.get_variable(
        name='resnet_period_estimator/pos_encoding',
        shape=[1, length, 1],
        initializer=tf.compat.v1.truncated_normal_initializer(stddev=0.02))
    self.pos_encoding2 = tf.compat.v1.get_variable(
        name='resnet_period_estimator/pos_encoding2',
        shape=[1, length, 1],
        initializer=tf.compat.v1.truncated_normal_initializer(stddev=0.02))

    self.transformer_layers = []
    for d_model, num_heads, dff in self.transformer_layers_config:
      self.transformer_layers.append(
          TransformerLayer(d_model, num_heads, dff,
                           self.transformer_dropout_rate,
                           self.transformer_reorder_ln))

    self.transformer_layers2 = []
    for d_model, num_heads, dff in self.transformer_layers_config:
      self.transformer_layers2.append(
          TransformerLayer(d_model, num_heads, dff,
                           self.transformer_dropout_rate,
                           self.transformer_reorder_ln))

    # Period Prediction Module.
    self.dropout_layer = layers.Dropout(self.dropout_rate)
    num_preds = self.num_frames//2
    self.fc_layers = []
    for channels in self.period_fc_channels:
      self.fc_layers.append(layers.Dense(
          channels, kernel_regularizer=regularizers.l2(self.l2_reg_weight),
          activation=tf.nn.relu))
    self.fc_layers.append(layers.Dense(
        num_preds, kernel_regularizer=regularizers.l2(self.l2_reg_weight)))

    # Within Period Module
    num_preds = 1
    self.within_period_fc_layers = []
    for channels in self.within_period_fc_channels:
      self.within_period_fc_layers.append(layers.Dense(
          channels, kernel_regularizer=regularizers.l2(self.l2_reg_weight),
          activation=tf.nn.relu))
    self.within_period_fc_layers.append(layers.Dense(
        num_preds, kernel_regularizer=regularizers.l2(self.l2_reg_weight)))

  def call(self, x):
    # Ensures we are always using the right batch_size during train/eval.
    batch_size = tf.shape(x)[0]
    # Conv Feature Extractor.
    x = tf.reshape(x, [-1, self.image_size, self.image_size, 3])
    x = self.base_model(x)
    h = tf.shape(x)[1]
    w = tf.shape(x)[2]
    c = tf.shape(x)[3]
    x = tf.reshape(x, [batch_size, -1, h, w, c])

    # 3D Conv to give temporal context to per-frame embeddings. 
    for bn_layer, conv_layer in zip(self.temporal_bn_layers,
                                    self.temporal_conv_layers):
      x = conv_layer(x)
      x = bn_layer(x)
      x = tf.nn.relu(x)

    x = tf.reduce_max(x, [2, 3])

    # Reshape and prepare embs for output.
    final_embs = x

    # Get self-similarity matrix.
    x = get_sims(x, self.temperature)

    # 3x3 conv layer on self-similarity matrix.
    x = self.conv_3x3_layer(x)
    x = tf.reshape(x, [batch_size, self.num_frames, -1])
    within_period_x = x

    # Period prediction.
    x = self.input_projection(x)
    x += self.pos_encoding
    for transformer_layer in self.transformer_layers:
      x = transformer_layer(x)
    x = flatten_sequential_feats(x, batch_size, self.num_frames)
    for fc_layer in self.fc_layers:
      x = self.dropout_layer(x)
      x = fc_layer(x)

    # Within period prediction.
    within_period_x = self.input_projection2(within_period_x)
    within_period_x += self.pos_encoding2
    for transformer_layer in self.transformer_layers2:
      within_period_x = transformer_layer(within_period_x)
    within_period_x = flatten_sequential_feats(within_period_x,
                                               batch_size,
                                               self.num_frames)
    for fc_layer in self.within_period_fc_layers:
      within_period_x = self.dropout_layer(within_period_x)
      within_period_x = fc_layer(within_period_x)

    return x, within_period_x, final_embs

  @tf.function
  def preprocess(self, imgs):
    imgs = tf.cast(imgs, tf.float32)
    imgs -= 127.5
    imgs /= 127.5
    imgs = tf.image.resize(imgs, (self.image_size, self.image_size))
    return imgs


def get_sims(embs, temperature):
  """Calculates self-similarity between batch of sequence of embeddings."""
  batch_size = tf.shape(embs)[0]
  seq_len = tf.shape(embs)[1]
  embs = tf.reshape(embs, [batch_size, seq_len, -1])

  def _get_sims(embs):
    """Calculates self-similarity between sequence of embeddings."""
    dist = pairwise_l2_distance(embs, embs)
    sims = -1.0 * dist
    return sims

  sims = tf.map_fn(_get_sims, embs)
  sims /= temperature
  sims = tf.nn.softmax(sims, axis=-1)
  sims = tf.expand_dims(sims, -1)
  return sims


def flatten_sequential_feats(x, batch_size, seq_len):
  """Flattens sequential features with known batch size and seq_len."""
  x = tf.reshape(x, [batch_size, seq_len, -1])
  return x


# Transformer from https://www.tensorflow.org/tutorials/text/transformer .
def scaled_dot_product_attention(q, k, v, mask):
  """Calculate the attention weights.

  q, k, v must have matching leading dimensions.
  k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
  The mask has different shapes depending on its type(padding or look ahead)
  but it must be broadcastable for addition.

  Args:
    q: query shape == (..., seq_len_q, depth)
    k: key shape == (..., seq_len_k, depth)
    v: value shape == (..., seq_len_v, depth_v)
    mask: Float tensor with shape broadcastable
          to (..., seq_len_q, seq_len_k). Defaults to None.

  Returns:
    outputs: shape == (..., seq_len_q, depth_v)
    attention_weights: shape == (..., seq_len_q, seq_len_k)
  """

  matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)

  # scale matmul_qk.
  dk = tf.cast(tf.shape(k)[-1], tf.float32)
  scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

  # add the mask to the scaled tensor.
  if mask is not None:
    scaled_attention_logits += (mask * -1e9)

  # softmax is normalized on the last axis (seq_len_k) so that the scores
  # add up to 1.
  # (..., seq_len_q, seq_len_k)
  attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)

  outputs = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

  return outputs, attention_weights


def point_wise_feed_forward_network(d_model, dff):
  return tf.keras.Sequential([
      tf.keras.layers.Dense(dff, activation='relu'),
      tf.keras.layers.Dense(d_model)
  ])


class MultiHeadAttention(tf.keras.layers.Layer):
  """Multi-headed attention layer."""

  def __init__(self, d_model, num_heads):
    super(MultiHeadAttention, self).__init__()
    self.num_heads = num_heads
    self.d_model = d_model

    assert d_model % self.num_heads == 0

    self.depth = d_model // self.num_heads

    self.wq = tf.keras.layers.Dense(d_model)
    self.wk = tf.keras.layers.Dense(d_model)
    self.wv = tf.keras.layers.Dense(d_model)

    self.dense = tf.keras.layers.Dense(d_model)

  def split_heads(self, x, batch_size):
    """Split the last dimension into (num_heads, depth)."""
    x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
    return tf.transpose(x, perm=[0, 2, 1, 3])

  def call(self, v, k, q, mask):
    batch_size = tf.shape(q)[0]

    q = self.wq(q)  # (batch_size, seq_len, d_model)
    k = self.wk(k)  # (batch_size, seq_len, d_model)
    v = self.wv(v)  # (batch_size, seq_len, d_model)

    q = self.split_heads(
        q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
    k = self.split_heads(
        k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
    v = self.split_heads(
        v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

    # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
    # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
    scaled_attention, attention_weights = scaled_dot_product_attention(
        q, k, v, mask)

    scaled_attention = tf.transpose(
        scaled_attention,
        perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)

    concat_attention = tf.reshape(
        scaled_attention,
        (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)

    outputs = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)

    return outputs, attention_weights


class TransformerLayer(tf.keras.layers.Layer):
  """Implements a single transformer layer (https://arxiv.org/abs/1706.03762).
  """

  def __init__(self, d_model, num_heads, dff,
               dropout_rate=0.1,
               reorder_ln=False):
    super(TransformerLayer, self).__init__()

    self.mha = MultiHeadAttention(d_model, num_heads)
    self.ffn = point_wise_feed_forward_network(d_model, dff)

    self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    self.dropout1 = tf.keras.layers.Dropout(dropout_rate)
    self.dropout2 = tf.keras.layers.Dropout(dropout_rate)

    self.reorder_ln = reorder_ln

  def call(self, x):
    inp_x = x

    if self.reorder_ln:
      x = self.layernorm1(x)

    # (batch_size, input_seq_len, d_model)
    attn_output, _ = self.mha(x, x, x, mask=None)
    attn_output = self.dropout1(attn_output)

    if self.reorder_ln:
      out1 = inp_x + attn_output
      x = out1
    else:
      # (batch_size, input_seq_len, d_model)
      out1 = self.layernorm1(x + attn_output)
      x = out1

    if self.reorder_ln:
      x = self.layernorm2(x)

    # (batch_size, input_seq_len, d_model)
    ffn_output = self.ffn(x)
    ffn_output = self.dropout2(ffn_output)

    if self.reorder_ln:
      out2 = out1 + ffn_output
    else:
      # (batch_size, input_seq_len, d_model)
      out2 = self.layernorm2(out1 + ffn_output)

    return out2


def pairwise_l2_distance(a, b):
  """Computes pairwise distances between all rows of a and all rows of b."""
  norm_a = tf.reduce_sum(tf.square(a), 1)
  norm_a = tf.reshape(norm_a, [-1, 1])
  norm_b = tf.reduce_sum(tf.square(b), 1)
  norm_b = tf.reshape(norm_b, [1, -1])
  dist = tf.maximum(norm_a - 2.0 * tf.matmul(a, b, False, True) + norm_b, 0.0)
  return dist


def get_repnet_model(logdir):
  """Returns a trained RepNet model.

  Args:
    logdir (string): Path to directory where checkpoint will be downloaded.

  Returns:
    model (Keras model): Trained RepNet model.
  """
  # Check if we are in eager mode.
  
  assert tf.executing_eagerly()
  
  
  # Models will be called in eval mode.
  tf.keras.backend.set_learning_phase(0)

  # Define RepNet model.
  model = ResnetPeriodEstimator()
  # tf.function for speed.
  model.call = tf.function(model.call)

  # Define checkpoint and checkpoint manager.
  ckpt = tf.train.Checkpoint(model=model)
  ckpt_manager = tf.train.CheckpointManager(
      ckpt, directory=logdir, max_to_keep=10)
  latest_ckpt = ckpt_manager.latest_checkpoint
  print('Loading from: ', latest_ckpt)
  if not latest_ckpt:
    raise ValueError('Path does not have a checkpoint to load.')
  # Restore weights.
  ckpt.restore(latest_ckpt).expect_partial()

  # Pass dummy frames to build graph.
  model(tf.random.uniform((1, 64, 112, 112, 3)))
  return model


def unnorm(query_frame):
  min_v = query_frame.min()
  max_v = query_frame.max()
  query_frame = (query_frame - min_v) / max(1e-7, (max_v - min_v))
  return query_frame

def data_uri_to_img(uri):
  """Convert base64image to Numpy array."""
  image = base64.b64decode(uri.split(',')[1], validate=True)
  # Binary string to PIL image.
  image = Image.open(io.BytesIO(image))
  image = image.resize((224, 224))
  # PIL to Numpy array.
  image = np.array(np.array(image, dtype=np.uint8), np.float32)
  return image


def read_video(video_filename, width=224, height=224):
  """Read video from file."""
  cap = cv2.VideoCapture(video_filename)
  fps = cap.get(cv2.CAP_PROP_FPS)
  frames = []
  if cap.isOpened():
    while True:
      success, frame_bgr = cap.read()
      if not success:
        break
      frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
      frame_rgb = cv2.resize(frame_rgb, (width, height))
      frames.append(frame_rgb)
  frames = np.asarray(frames)
  return frames, fps

def get_score(period_score, within_period_score):
  """Combine the period and periodicity scores."""
  within_period_score = tf.nn.sigmoid(within_period_score)[:, 0]
  per_frame_periods = tf.argmax(period_score, axis=-1) + 1
  pred_period_conf = tf.reduce_max(
      tf.nn.softmax(period_score, axis=-1), axis=-1)
  pred_period_conf = np.where(
      tf.math.less(per_frame_periods, 3), 0.0, pred_period_conf)
  within_period_score *= pred_period_conf
  within_period_score = np.sqrt(within_period_score)
  pred_score = tf.reduce_mean(within_period_score)
  return pred_score, within_period_score


def get_counts(model, frames, strides, batch_size,
               threshold,
               within_period_threshold,
               constant_speed=False,
               median_filter=False,
               fully_periodic=False):
  """Pass frames through model and conver period predictions to count."""
  seq_len = len(frames)
  raw_scores_list = []
  scores = []
  within_period_scores_list = []

  if fully_periodic:
    within_period_threshold = 0.0

  #print('frames shape' , frames.shape)
  frames = model.preprocess(frames)

  for stride in strides:
    num_batches = int(np.ceil(seq_len/model.num_frames/stride/batch_size))
    raw_scores_per_stride = []
    within_period_score_stride = []
    for batch_idx in range(num_batches):
      idxes = tf.range(batch_idx*batch_size*model.num_frames*stride,
                       (batch_idx+1)*batch_size*model.num_frames*stride,
                       stride)
      idxes = tf.clip_by_value(idxes, 0, seq_len-1)
      curr_frames = tf.gather(frames, idxes)
      curr_frames = tf.reshape(
          curr_frames,
          [batch_size, model.num_frames, model.image_size, model.image_size, 3])

      raw_scores, within_period_scores, _ = model(curr_frames)
      raw_scores_per_stride.append(np.reshape(raw_scores.numpy(),
                                              [-1, model.num_frames//2]))
      within_period_score_stride.append(np.reshape(within_period_scores.numpy(),
                                                   [-1, 1]))
    raw_scores_per_stride = np.concatenate(raw_scores_per_stride, axis=0)
    raw_scores_list.append(raw_scores_per_stride)
    within_period_score_stride = np.concatenate(
        within_period_score_stride, axis=0)
    pred_score, within_period_score_stride = get_score(
        raw_scores_per_stride, within_period_score_stride)
    scores.append(pred_score)
    within_period_scores_list.append(within_period_score_stride)

  # Stride chooser
  argmax_strides = np.argmax(scores)
  chosen_stride = strides[argmax_strides]
  raw_scores = np.repeat(
      raw_scores_list[argmax_strides], chosen_stride, axis=0)[:seq_len]
  within_period = np.repeat(
      within_period_scores_list[argmax_strides], chosen_stride,
      axis=0)[:seq_len]
  within_period_binary = np.asarray(within_period > within_period_threshold)
  if median_filter:
    within_period_binary = medfilt(within_period_binary, 5)

  # Select Periodic frames
  periodic_idxes = np.where(within_period_binary)[0]

  if constant_speed:
    # Count by averaging predictions. Smoother but
    # assumes constant speed.
    scores = tf.reduce_mean(
        tf.nn.softmax(raw_scores[periodic_idxes], axis=-1), axis=0)
    max_period = np.argmax(scores)
    pred_score = scores[max_period]
    pred_period = chosen_stride * (max_period + 1)
    per_frame_counts = (
        np.asarray(seq_len * [1. / pred_period]) *
        np.asarray(within_period_binary))
  else:
    # Count each frame. More noisy but adapts to changes in speed.
    pred_score = tf.reduce_mean(within_period)
    per_frame_periods = tf.argmax(raw_scores, axis=-1) + 1
    per_frame_counts = np.where(
        tf.math.less(per_frame_periods, 3),
        0.0,
        tf.math.divide(1.0,
                       tf.cast(chosen_stride * per_frame_periods, tf.float32)),
    )
    if median_filter:
      per_frame_counts = medfilt(per_frame_counts, 5)

    per_frame_counts *= np.asarray(within_period_binary)

    pred_period = seq_len/np.sum(per_frame_counts)

  if pred_score < threshold:
    print('No repetitions detected in video as score '
          '%0.2f is less than threshold %0.2f.'%(pred_score, threshold))
    per_frame_counts = np.asarray(len(per_frame_counts) * [0.])

  return (pred_period, pred_score, within_period,
          per_frame_counts, chosen_stride)
print("setup is ok")

# out_dir=r"E:\itsspm\itsspm\models\slow_video\fplower"

def video_transform_image(video_path,out_dir):
    
    name = os.path.basename(video_path)
    name=name[:-4]
    
    input_file = video_path
    cap = cv2.VideoCapture(input_file)
    
    

    # Read the video frame-by-frame and write it to the output file

    i=0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_path = os.path.join(out_dir, 'fplower_%s_%i.png'%(name,i)) # Change the extension to match your desired image format

        i=i+1
        cv2.imwrite(frame_path, frame)
      
    cap.release()
    
    cv2.destroyAllWindows()
    return 

def slow_video_generate(video_path, out_dir):
    # Open the input video file
   
    name = os.path.basename(video_path)
    name=name[:-4]
    input_file = video_path
    cap = cv2.VideoCapture(input_file)
    # Get the video frame rate and calculate the new frame rate
    fps = cap.get(cv2.CAP_PROP_FPS)


    # Define the codec and create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    #output_file = r"E:\itsspm\itsspm\models\slow_video\%s_slow.mp4"%(name)
    print ('slow_video_generate outfile name: ', out_dir + r"\%s_slow.mp4"%(name))
    output_file = out_dir + r"\%s_slow.mp4"%(name)
    out = cv2.VideoWriter(output_file, fourcc, fps, (int(cap.get(3)), int(cap.get(4))))
    i=0
    slow_down_factor=2
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            image_path = os.path.join(out_dir, 'fplower_%s_%i.png'%(name,i))
            i=i+1
            image = cv2.imread(image_path)
            for j in range(slow_down_factor):
            
                # print(image.shape)
                # Write the combined frame to the output video file
                out.write(image)
        else:
            break
    cap.release()
    out.release()

    return output_file 

# #set parameter
# # FPS while recording video from webcam.
# WEBCAM_FPS = 16#@param {type:"integer"}

# # Time in seconds to record video on webcam. 
# RECORDING_TIME_IN_SECONDS = 8. #@param {type:"number"}

# # Threshold to consider periodicity in entire video.
# THRESHOLD = 0.001#@param {type:"number"}

# # Threshold to consider periodicity for individual frames in video.
# WITHIN_PERIOD_THRESHOLD = 0.005#@param {type:"number"}

# # Use this setting for better results when it is 
# # known action is repeating at constant speed.
# CONSTANT_SPEED = False#@param {type:"boolean"}

# # Use median filtering in time to ignore noisy frames.
# MEDIAN_FILTER = True#@param {type:"boolean"}

# # Use this setting for better results when it is 
# # known the entire video is periodic/reapeating and
# # has no aperiodic frames.
# FULLY_PERIODIC = True#@param {type:"boolean"}

# # Plot score in visualization video.
# PLOT_SCORE = False#@param {type:"boolean"}

# # Visualization video's FPS.
# VIZ_FPS = 30#@param {type:"integer"}

# print('loading twist model')
# global twist_model
# PATH_TO_CKPT = r'E:\itsspm\itsspm\modelcheckpoints\repnet_ckpt'
# tf.compat.v1.enable_v2_behavior()
# model = get_repnet_model(PATH_TO_CKPT)
# twist_model=model

# print('loading somersault model')
# global somersault_model
# PATH_TO_CKPT = r'E:\itsspm\itsspm\modelcheckpoints\repnet_ckpt'
# tf.compat.v1.enable_v2_behavior()
# model = get_repnet_model(PATH_TO_CKPT)
# somersault_model=model
# out_dir=r"E:\itsspm\itsspm\models\slow_video\fplower"
# video_transform_image(r'E:\itsspm\itsspm\01_1_somersault.mp4',out_dir)
# slow_video_generate(r'E:\itsspm\itsspm\01_1_somersault.mp4',out_dir)
# def predict_num_somersaults():
   
#     video_transform_image(r'E:\itsspm\itsspm\01_1_somersault.mp4',out_dir) # when merge, can use imgfolder format
#     output_video=slow_video_generate(r'E:\itsspm\itsspm\01_1_somersault.mp4',out_dir)
#     imgs, vid_fps = read_video(output_video)
#     print('Running RepNet...') 
#     (pred_period, pred_score, within_period,
#     per_frame_counts, chosen_stride) = get_counts(
#         somersault_model,
#         imgs,
#         strides=[1,2,3,4],
#         batch_size=20,
#         threshold=THRESHOLD,
#         within_period_threshold=WITHIN_PERIOD_THRESHOLD,
#         constant_speed=CONSTANT_SPEED,
#         median_filter=MEDIAN_FILTER,
#         fully_periodic=FULLY_PERIODIC)
#     frames=imgs
#     count=per_frame_counts
#     if isinstance(count, list):
#         counts = len(frames) * [count/len(frames)]
#     else:
#         counts = count
#     sum_counts = np.cumsum(counts)
#     num= round(float(sum_counts[-1]),2)
    
   
#     return num

# def predict_num_twists():
    
#     video_transform_image(r'E:\itsspm\itsspm\01_3_twist.mp4',out_dir) # when merge, can use imgfolder format
#     output_video=slow_video_generate(r'E:\itsspm\itsspm\01_3_twist.mp4',out_dir)
#     imgs, vid_fps = read_video(output_video)
#     print('Running RepNet...') 
#     (pred_period, pred_score, within_period,
#     per_frame_counts, chosen_stride) = get_counts(
#         twist_model,
#         imgs,
#         strides=[1,2,3,4],
#         batch_size=20,
#         threshold=THRESHOLD,
#         within_period_threshold=WITHIN_PERIOD_THRESHOLD,
#         constant_speed=CONSTANT_SPEED,
#         median_filter=MEDIAN_FILTER,
#         fully_periodic=FULLY_PERIODIC)
#     frames=imgs
#     count=per_frame_counts
#     if isinstance(count, list):
#         counts = len(frames) * [count/len(frames)]
#     else:
#         counts = count
#     sum_counts = np.cumsum(counts)
#     num= round(float(sum_counts[-1]),2)
#     return num

# num1=predict_num_somersaults()
# num2=predict_num_twists()
# print(num1)
# print(num2)

# Citation

# If you found our paper/code useful in your research, consider citing our paper:


# ```
# @InProceedings{Dwibedi_2020_CVPR,
# author = {Dwibedi, Debidatta and Aytar, Yusuf and Tompson, Jonathan and Sermanet, Pierre and Zisserman, Andrew},
# title = {Counting Out Time: Class Agnostic Video Repetition Counting in the Wild},
# booktitle = {IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
# month = {June},
# year = {2020}
# }
# ```
