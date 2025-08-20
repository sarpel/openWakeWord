import torch
from torch import optim, nn
import torchinfo
import torchmetrics
import copy
import os
import sys
import tempfile
import uuid
import numpy as np
import scipy
from scipy.signal import fftconvolve
import collections
import argparse
import random
import logging
from tqdm import tqdm
import yaml
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
import openwakeword
from openwakeword.data import generate_adversarial_texts, augment_clips, mmap_batch_generator
from openwakeword.utils import compute_features_from_generator
from openwakeword.utils import AudioFeatures

class EarlyStopping:
    def __init__(self, patience=200, min_delta=0.0, mode="min", restore_best_weights=True):
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.mode = mode
        self.restore_best_weights = restore_best_weights
        self.best = None
        self.count = 0
        self.best_state = None

# ------------------ Config-driven Augmentation Helpers ------------------
def _safe_imports_for_aug():
    """try to import optional libs for augmentation"""
    mods = {}
    try:
        import soundfile as sf  # preferred for wav IO
        mods['sf'] = sf
    except Exception:
        mods['sf'] = None
    try:
        import librosa  # pitch/time-stretch + griffin-lim
        mods['librosa'] = librosa
    except Exception:
        mods['librosa'] = None
    try:
        from scipy.io import wavfile as wavio
        mods['wavio'] = wavio
    except Exception:
        mods['wavio'] = None
    return mods

_AUG_MODS = _safe_imports_for_aug()

def _load_wav_mono(path, target_len, sr=16000):
    """Load wav mono, resample to sr if we can, pad/trim to target_len samples."""
    sf = _AUG_MODS.get('sf')
    librosa = _AUG_MODS.get('librosa')
    wavio = _AUG_MODS.get('wavio')
    y = None
    got_sr = None
    if sf is not None and isinstance(path, str):
        try:
            y, got_sr = sf.read(path, dtype="float32", always_2d=False)
            if hasattr(y, 'ndim') and y.ndim > 1:
                import numpy as _np
                y = _np.mean(y, axis=1)
        except Exception:
            y = None
    if y is None and wavio is not None and isinstance(path, str):
        try:
            got_sr, y = wavio.read(path)
            import numpy as _np
            if hasattr(y, 'ndim') and y.ndim > 1:
                y = y.mean(axis=1)
            y = y.astype("float32")/32768.0
        except Exception:
            y = None
    if y is None and librosa is not None and isinstance(path, str):
        try:
            y, got_sr = librosa.load(path, sr=None, mono=True)
        except Exception:
            y = None
    if y is None:
        import numpy as _np
        y = _np.zeros(target_len, dtype=_np.float32)
        got_sr = sr
    # resample if needed
    if got_sr != sr and _AUG_MODS.get('librosa') is not None:
        try:
            y = _AUG_MODS['librosa'].resample(y, orig_sr=got_sr, target_sr=sr)
        except Exception:
            pass
    # pad/trim
    import numpy as _np
    if len(y) < target_len:
        y = _np.pad(y, (0, target_len - len(y)))
    elif len(y) > target_len:
        y = y[:target_len]
    return y.astype("float32")

def _snr_mix(x, n, snr_db):
    """Mix noise/background n into signal x at a desired SNR in dB."""
    import numpy as _np
    x_pow = _np.mean(x**2) + 1e-9
    n_pow = _np.mean(n**2) + 1e-9
    target_np = x_pow / (10**(snr_db/10.0))
    scale = (target_np / n_pow) ** 0.5
    return x + scale*n

def _apply_reverb(x, rir, wet=1.0):
    """Convolve x with rir and mix wet."""
    if rir is None or len(rir) == 0:
        return x
    y = fftconvolve(x, rir, mode='full')[:len(x)]
    return (1.0 - wet) * x + wet * y

def _spec_augment_waveform(x, sr=16000, time_mask_param=25, freq_mask_param=12, n_mels=64):
    """Apply simple SpecAugment by masking on mel-spectrogram and invert via Griffin-Lim (heavy)."""
    librosa = _AUG_MODS.get('librosa')
    if librosa is None:
        return x  # skip if librosa missing
    S = librosa.feature.melspectrogram(y=x, sr=sr, n_mels=n_mels, power=2.0)
    # time mask
    t = S.shape[1]
    if t > time_mask_param:
        t0 = np.random.randint(0, t - time_mask_param)
        S[:, t0:t0+time_mask_param] = 0.0
    # freq mask
    f = S.shape[0]
    if f > freq_mask_param:
        f0 = np.random.randint(0, f - freq_mask_param)
        S[f0:f0+freq_mask_param, :] = 0.0
    # invert
    y = librosa.feature.inverse.mel_to_audio(S, sr=sr)
    if len(y) < len(x):
        import numpy as _np
        y = _np.pad(y, (0, len(x)-len(y)))
    elif len(y) > len(x):
        y = y[:len(x)]
    return y.astype(np.float32)

def _choose_random_file(paths):
    if not paths:
        return None
    return random.choice(paths)

def augment_clips_with_config(file_list, total_length, batch_size, background_clip_paths, RIR_paths, config):
    """Wrapper that yields augmented batches honoring YAML config."""
    sr = 16000
    # probs
    p_all = float(config.get('augmentation_probability', 1.0) or 1.0)
    p_bg = float(config.get('background_augmentation_probability', 0.0) or 0.0)
    p_noise = float(config.get('noise_augmentation_probability', 0.0) or 0.0)
    aug_cfg = config.get('augmentation_config', {}) or {}

    # pitch
    pitch_cfg = aug_cfg.get('pitch_shift', {}) or {}
    p_pitch = float(pitch_cfg.get('prob', 0.0) or 0.0)
    min_semi = float(pitch_cfg.get('min_semitones', -2.0))
    max_semi = float(pitch_cfg.get('max_semitones', 2.0))

    # time stretch
    ts_cfg = aug_cfg.get('time_stretch', {}) or {}
    p_ts = float(ts_cfg.get('prob', 0.0) or 0.0)
    min_rate = float(ts_cfg.get('min_rate', 0.9))
    max_rate = float(ts_cfg.get('max_rate', 1.1))

    # reverb
    rv_cfg = aug_cfg.get('reverb', {}) or {}
    p_rv = float(rv_cfg.get('prob', 0.0) or 0.0)
    # allow overriding RIR paths in yaml under augmentation_config.reverb.rir_paths
    rir_override = rv_cfg.get('rir_paths') or None
    rir_pool = [str(p) for p in (rir_override if rir_override else RIR_paths)] if (rir_override or RIR_paths) else []

    # spec augment
    sa_cfg = aug_cfg.get('spec_augment', {}) or {}
    p_sa = float(sa_cfg.get('prob', 0.0) or 0.0)
    time_mask_param = int(sa_cfg.get('time_mask_param', 25))
    freq_mask_param = int(sa_cfg.get('freq_mask_param', 12))

    # SNRs
    bg_snr_min = float(aug_cfg.get('background_snr_db_min', 5.0))
    bg_snr_max = float(aug_cfg.get('background_snr_db_max', 20.0))
    nz_snr_min = float(aug_cfg.get('noise_snr_db_min', 8.0))
    nz_snr_max = float(aug_cfg.get('noise_snr_db_max', 25.0))

    def _batch_iter():
        batch = []
        for fp in file_list:
            x = _load_wav_mono(fp, target_len=total_length, sr=sr)
            if random.random() < p_all:
                if p_bg > 0 and random.random() < p_bg and background_clip_paths:
                    bg_fp = _choose_random_file(background_clip_paths)
                    if bg_fp is not None:
                        bg = _load_wav_mono(bg_fp, target_len=total_length, sr=sr)
                        snr = random.uniform(bg_snr_min, bg_snr_max)
                        x = _snr_mix(x, bg, snr)
                if p_noise > 0 and random.random() < p_noise:
                    n = np.random.randn(total_length).astype(np.float32)
                    snr = random.uniform(nz_snr_min, nz_snr_max)
                    x = _snr_mix(x, n, snr)
                if p_pitch > 0 and random.random() < p_pitch and _AUG_MODS.get('librosa') is not None:
                    semi = random.uniform(min_semi, max_semi)
                    try:
                        x = _AUG_MODS['librosa'].effects.pitch_shift(x, sr=sr, n_steps=semi)
                    except Exception:
                        pass
                if p_ts > 0 and random.random() < p_ts and _AUG_MODS.get('librosa') is not None:
                    rate = random.uniform(min_rate, max_rate)
                    try:
                        xt = _AUG_MODS['librosa'].effects.time_stretch(x, rate=rate)
                        if len(xt) < total_length:
                            import numpy as _np
                            xt = _np.pad(xt, (0, total_length - len(xt)))
                        elif len(xt) > total_length:
                            xt = xt[:total_length]
                        x = xt.astype(np.float32)
                    except Exception:
                        pass
                if p_rv > 0 and random.random() < p_rv and rir_pool:
                    rir_fp = _choose_random_file(rir_pool)
                    try:
                        rir = _load_wav_mono(rir_fp, target_len=total_length, sr=sr)
                        rir = rir / (np.max(np.abs(rir)) + 1e-6)
                        x = _apply_reverb(x, rir, wet=1.0)
                    except Exception:
                        pass
                if p_sa > 0 and random.random() < p_sa:
                    x = _spec_augment_waveform(x, sr=sr, time_mask_param=time_mask_param, freq_mask_param=freq_mask_param)

            m = float(np.max(np.abs(x)) + 1e-9)
            if m > 1.0:
                x = x / m
            batch.append(x.astype(np.float32))
            if len(batch) == batch_size:
                import numpy as _np
                # GPU memory optimization: ensure contiguous arrays
                batch_array = _np.stack(batch, axis=0)
                # Ensure contiguous memory layout for faster GPU transfer
                if not batch_array.flags['C_CONTIGUOUS']:
                    batch_array = np.ascontiguousarray(batch_array)
                yield batch_array
                batch = []
        if batch:
            import numpy as _np
            batch_array = _np.stack(batch, axis=0)
            # Ensure contiguous memory layout for faster GPU transfer
            if not batch_array.flags['C_CONTIGUOUS']:
                batch_array = np.ascontiguousarray(batch_array)
            yield batch_array

    return _batch_iter()
# ---------------- End Augmentation Helpers ----------------

    def _is_improvement(self, current, best):
        if self.mode == "min":
            return current < (best - self.min_delta)
        else:
            return current > (best + self.min_delta)

    def step(self, current, model=None):
        if self.best is None:
            self.best = current
            if self.restore_best_weights and model is not None:
                self.best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            return False

        if self._is_improvement(current, self.best):
            self.best = current
            self.count = 0
            if self.restore_best_weights and model is not None:
                self.best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            return False
        else:
            self.count += 1
            if self.count >= self.patience:
                if self.restore_best_weights and model is not None and self.best_state is not None:
                    model.load_state_dict(self.best_state, strict=False)
                return True
            return False

# Base model class for an openwakeword model
class Model(nn.Module):
    def __init__(self, n_classes=1, input_shape=(16, 96), model_type="dnn",
                 layer_dim=128, n_blocks=1, seconds_per_example=None, dropout=0.0):
        super().__init__()

        # Store inputs as attributes
        self.n_classes = n_classes
        self.input_shape = input_shape
        self.seconds_per_example = seconds_per_example
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.best_models = []
        self.best_model_scores = []
        self.best_val_fp = 1000
        self.best_val_accuracy = 0
        self.best_val_recall = 0
        self.best_train_recall = 0

        # TensorBoard writer (set during training)
        self.writer = None
        self._early_stopped = False

        # Define model (currently on fully-connected network supported)
        if model_type == "dnn":
            # self.model = nn.Sequential(
            #     nn.Flatten(),
            #     nn.Linear(input_shape[0]*input_shape[1], layer_dim),
            #     nn.LayerNorm(layer_dim),
            #     nn.ReLU(),
            #     nn.Linear(layer_dim, layer_dim),
            #     nn.LayerNorm(layer_dim),
            #     nn.ReLU(),
            #     nn.Linear(layer_dim, n_classes),
            #     nn.Sigmoid() if n_classes == 1 else nn.ReLU(),
            # )

            class FCNBlock(nn.Module):
                def __init__(self, layer_dim, p_drop: float):
                    super().__init__()
                    self.fcn_layer = nn.Linear(layer_dim, layer_dim)
                    self.relu = nn.ReLU()
                    self.layer_norm = nn.LayerNorm(layer_dim)
                    self.drop = nn.Dropout(p=p_drop) if p_drop and p_drop > 0 else nn.Identity()

                def forward(self, x):
                    x = self.fcn_layer(x)
                    x = self.layer_norm(x)
                    x = self.relu(x)
                    x = self.drop(x)
                    return x

            class Net(nn.Module):
                def __init__(self, input_shape, layer_dim, n_blocks=1, n_classes=1, p_drop: float = 0.0):
                    super().__init__()
                    self.flatten = nn.Flatten()
                    self.layer1 = nn.Linear(input_shape[0]*input_shape[1], layer_dim)
                    self.relu1 = nn.ReLU()
                    self.layernorm1 = nn.LayerNorm(layer_dim)
                    self.drop1 = nn.Dropout(p=p_drop) if p_drop and p_drop > 0 else nn.Identity()
                    self.blocks = nn.ModuleList([FCNBlock(layer_dim, p_drop) for i in range(n_blocks)])
                    self.last_layer = nn.Linear(layer_dim, n_classes)
                    self.last_act = nn.Sigmoid() if n_classes == 1 else nn.ReLU()

                def forward(self, x):
                    x = self.flatten(x)
                    x = self.layer1(x)
                    x = self.layernorm1(x)
                    x = self.relu1(x)
                    x = self.drop1(x)
                    for block in self.blocks:
                        x = block(x)
                    x = self.last_act(self.last_layer(x))
                    return x
            self.model = Net(input_shape, layer_dim, n_blocks=n_blocks, n_classes=n_classes, p_drop=dropout)
        elif model_type == "rnn":
            class Net(nn.Module):
                def __init__(self, input_shape, n_classes=1):
                    super().__init__()
                    self.layer1 = nn.LSTM(input_shape[-1], 64, num_layers=2, bidirectional=True,
                                          batch_first=True, dropout=0.0)
                    self.layer2 = nn.Linear(64*2, n_classes)
                    self.layer3 = nn.Sigmoid() if n_classes == 1 else nn.ReLU()

                def forward(self, x):
                    out, h = self.layer1(x)
                    return self.layer3(self.layer2(out[:, -1]))
            self.model = Net(input_shape, n_classes)

        # Define metrics
        if n_classes == 1:
            self.fp = lambda pred, y: (y-pred <= -0.5).sum()
            self.recall = torchmetrics.Recall(task='binary')
            self.accuracy = torchmetrics.Accuracy(task='binary')
        else:
            def multiclass_fp(p, y, threshold=0.5):
                probs = torch.nn.functional.softmax(p, dim=1)
                neg_ndcs = y == 0
                fp = (probs[neg_ndcs].argmax(axis=1) != 0 & (probs[neg_ndcs].max(axis=1)[0] > threshold)).sum()
                return fp

            def positive_class_recall(p, y, negative_class_label=0, threshold=0.5):
                probs = torch.nn.functional.softmax(p, dim=1)
                pos_ndcs = y != 0
                rcll = (probs[pos_ndcs].argmax(axis=1) > 0
                        & (probs[pos_ndcs].max(axis=1)[0] >= threshold)).sum()/pos_ndcs.sum()
                return rcll

            def positive_class_accuracy(p, y, negative_class_label=0):
                probs = torch.nn.functional.softmax(p, dim=1)
                pos_preds = probs.argmax(axis=1) != negative_class_label
                acc = (probs[pos_preds].argmax(axis=1) == y[pos_preds]).sum()/pos_preds.sum()
                return acc

            self.fp = multiclass_fp
            self.acc = positive_class_accuracy
            self.recall = positive_class_recall

        self.n_fp = 0
        self.val_fp = 0

        # Define logging dict (in-memory)
        self.history = collections.defaultdict(list)

        # Define optimizer and loss
        self.loss = torch.nn.functional.binary_cross_entropy if n_classes == 1 else nn.functional.cross_entropy
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0001)

    def save_model(self, output_path):
        """
        Saves the weights of a trained Pytorch model
        """
        if self.n_classes == 1:
            torch.save(self.model, output_path)

    def export_to_onnx(self, output_path, class_mapping=""):
        obj = self
        # Make simple model for export based on model structure
        if self.n_classes == 1:
            # Save ONNX model
            torch.onnx.export(self.model.to("cpu"), torch.rand(self.input_shape)[None, ], output_path,
                              output_names=[class_mapping])

        elif self.n_classes >= 1:
            class M(nn.Module):
                def __init__(self):
                    super().__init__()

                    # Define model
                    self.model = obj.model.to("cpu")

                def forward(self, x):
                    return torch.nn.functional.softmax(self.model(x), dim=1)

            # Save ONNX model
            torch.onnx.export(M(), torch.rand(self.input_shape)[None, ], output_path,
                              output_names=[class_mapping])

    def lr_warmup_cosine_decay(self,
                               global_step,
                               warmup_steps=0,
                               hold=0,
                               total_steps=0,
                               start_lr=0.0,
                               target_lr=1e-3
                               ):
        # Cosine decay
        learning_rate = 0.5 * target_lr * (1 + np.cos(np.pi * (global_step - warmup_steps - hold)
                                           / float(total_steps - warmup_steps - hold)))

        # Target LR * progress of warmup (=1 at the final warmup step)
        warmup_lr = target_lr * (global_step / warmup_steps)

        # Choose between `warmup_lr`, `target_lr` and `learning_rate` based on whether
        # `global_step < warmup_steps` and we're still holding.
        # i.e. warm up if we're still warming up and use cosine decayed lr otherwise
        if hold > 0:
            learning_rate = np.where(global_step > warmup_steps + hold,
                                     learning_rate, target_lr)

        learning_rate = np.where(global_step < warmup_steps, warmup_lr, learning_rate)
        return learning_rate

    def forward(self, x):
        return self.model(x)

    def summary(self):
        return torchinfo.summary(self.model, input_size=(1,) + self.input_shape, device='cpu')

    def average_models(self, models=None):
        """Averages the weights of the provided models together to make a new model"""

        if models is None:
            models = self.best_models

        # Clone a model from the list as the base for the averaged model
        averaged_model = copy.deepcopy(models[0])
        averaged_model_dict = averaged_model.state_dict()

        # Initialize a running total of the weights
        for key in averaged_model_dict:
            averaged_model_dict[key] *= 0  # set to 0

        for model in models:
            model_dict = model.state_dict()
            for key, value in model_dict.items():
                averaged_model_dict[key] += value

        for key in averaged_model_dict:
            averaged_model_dict[key] /= len(models)

        # Load the averaged weights into the model
        averaged_model.load_state_dict(averaged_model_dict)

        return averaged_model

    def _select_best_model(self, false_positive_validate_data, val_set_hrs=11.3, max_fp_per_hour=0.5, min_recall=0.20):
        """
        Select the top model based on the false positive rate on the validation data

        Args:
            false_positive_validate_data (torch.DataLoader): A dataloader with validation data
            n (int): The number of models to select

        Returns:
            list: A list of the top n models
        """
        # Get false positive rates for each model
        false_positive_rates = [0]*len(self.best_models)
        for batch in false_positive_validate_data:
            x_val, y_val = batch[0].to(self.device), batch[1].to(self.device)
            for mdl_ndx, model in tqdm(enumerate(self.best_models), total=len(self.best_models),
                                       desc="Find best checkpoints by false positive rate"):
                with torch.no_grad():
                    val_ps = model(x_val)
                    false_positive_rates[mdl_ndx] = false_positive_rates[mdl_ndx] + self.fp(val_ps, y_val[..., None]).detach().cpu().numpy()
        false_positive_rates = [fp/val_set_hrs for fp in false_positive_rates]

        candidate_model_ndx = [ndx for ndx, fp in enumerate(false_positive_rates) if fp <= max_fp_per_hour]
        candidate_model_recall = [self.best_model_scores[ndx]["val_recall"] for ndx in candidate_model_ndx]
        if max(candidate_model_recall) <= min_recall:
            logging.warning(f"No models with recall >= {min_recall} found!")
            return None
        else:
            best_model = self.best_models[candidate_model_ndx[np.argmax(candidate_model_recall)]]
            best_model_training_step = self.best_model_scores[candidate_model_ndx[np.argmax(candidate_model_recall)]]["training_step_ndx"]
            logging.info(f"Best model from training step {best_model_training_step} out of {len(candidate_model_ndx)}"
                         f"models has recall of {np.max(candidate_model_recall)} and false positive rate of"
                         f" {false_positive_rates[candidate_model_ndx[np.argmax(candidate_model_recall)]]}")

        return best_model

    
    def auto_train(self, X_train, X_val, false_positive_val_data, config):
        """Single, config-driven training run with TensorBoard + EarlyStopping."""
        import datetime, os
        ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        logdir = os.path.join("runs", f"oww-{config.get('model_name','model')}-{ts}")
        self.writer = SummaryWriter(log_dir=logdir)
        logging.info(f"[TensorBoard] Logging to: {logdir}")

        steps = int(config.get('steps', 15000))
        lr = float(config.get('learning_rate', 0.0001))
        warmup_steps = int(config.get('warmup_steps', max(1, int(steps*0.03))))
        hold_steps = int(config.get('hold_steps', max(1, int(steps*0.10))))
        max_negative_weight = float(config.get('max_negative_weight', 1.0))
        val_steps_interval = int(config.get('val_steps', 200))
        val_steps = list(range(val_steps_interval, steps, val_steps_interval))
        val_set_hrs = float(config.get('val_set_hrs', 11.3))

        weights = [max_negative_weight for _ in range(steps)]

        self.train_model(
            X=X_train,
            X_val=X_val,
            false_positive_val_data=false_positive_val_data,
            max_steps=steps,
            negative_weight_schedule=weights,
            val_steps=val_steps,
            warmup_steps=warmup_steps,
            hold_steps=hold_steps,
            lr=lr,
            val_set_hrs=val_set_hrs,
            config=config
        )
        return self.model

    def predict_on_features(self, features, model=None):
        """
        Predict on Tensors of openWakeWord features corresponding to single audio clips

        Args:
            features (torch.Tensor): A Tensor of openWakeWord features with shape (batch, features)
            model (torch.nn.Module): A Pytorch model to use for prediction (default None, which will use self.model)

        Returns:
            torch.Tensor: An array of predictions of shape (batch, prediction), where 0 is negative and 1 is positive
        """
        if len(features) < 3:
            features = features[None, ]

        # Ensure features are contiguous for optimal GPU performance
        features = features.contiguous().to(self.device)
        predictions = []
        for x in tqdm(features, desc="Predicting on clips"):
            x = x[None, ]
            batch = []
            for i in range(0, x.shape[1]-16, 1):  # step size of 1 (80 ms)
                batch.append(x[:, i:i+16, :])
            batch = torch.vstack(batch)
            if model is None:
                preds = self.model(batch)
            else:
                preds = model(batch)
            predictions.append(preds.detach().cpu().numpy()[None, ])

        return np.vstack(predictions)

    def predict_on_clips(self, clips, model=None):
        """
        Predict on Tensors of 16-bit 16 khz audio data

        Args:
            clips (np.ndarray): A Numpy array of audio clips with shape (batch, samples)
            model (torch.nn.Module): A Pytorch model to use for prediction (default None, which will use self.model)

        Returns:
            np.ndarray: An array of predictions of shape (batch, prediction), where 0 is negative and 1 is positive
        """
        # CRITICAL FIX: Use GPU when available for AudioFeatures
        device_for_features = 'cuda' if torch.cuda.is_available() else 'cpu'
        # Get features from clips using GPU if available
        F = AudioFeatures(device=device_for_features, ncpu=4 if device_for_features == 'cpu' else 1)
        features = F.embed_clips(clips, batch_size=16)

        # Predict on features
        preds = self.predict_on_features(torch.from_numpy(features), model=model)

        return preds

    def export_model(self, model, model_name, output_dir):
        """Saves the trained openwakeword model to both onnx and tflite formats"""

        if self.n_classes != 1:
            raise ValueError("Exporting models to both onnx and tflite with more than one class is currently not supported! "
                             "Use the `export_to_onnx` function instead.")

        # Save ONNX model
        logging.info(f"####\nSaving ONNX mode as '{os.path.join(output_dir, model_name + '.onnx')}'")
        model_to_save = copy.deepcopy(model)
        torch.onnx.export(model_to_save.to("cpu"), torch.rand(self.input_shape)[None, ],
                          os.path.join(output_dir, model_name + ".onnx"), opset_version=13)

        return None

    def train_model(self, X, max_steps, warmup_steps, hold_steps, X_val=None,
                    false_positive_val_data=None, positive_test_clips=None,
                    negative_weight_schedule=[1],
                    val_steps=[250], lr=0.0001, val_set_hrs=1, config=None):
        # Move models and main class to target device
        self.to(self.device)
        self.model.to(self.device)

        # Train model
        monitor_key = (config or {}).get('early_stopping_monitor', 'val_loss')
        monitor_mode = (config or {}).get('early_stopping_mode', 'min')
        patience = int((config or {}).get('early_stopping_patience', 4000))
        min_delta = float((config or {}).get('early_stopping_min_delta', 0.0))
        clip_norm = float((config or {}).get('gradient_clip_norm', 0) or 0)
        lr_schedule = str(((config or {}).get('learning_rate_schedule', 'cosine_annealing') or 'cosine_annealing')).lower()
        stopper = EarlyStopping(patience=patience, min_delta=min_delta, mode=monitor_mode, restore_best_weights=True)

        accumulation_steps = 1
        accumulated_samples = 0
        accumulated_predictions = torch.Tensor([]).to(self.device)
        accumulated_labels = torch.Tensor([]).to(self.device)
        for step_ndx, data in tqdm(enumerate(X, 0), total=max_steps, desc="Training"):
            # get the inputs; data is a list of [inputs, labels]
            x, y = data[0].to(self.device), data[1].to(self.device)
            y_ = y[..., None].to(torch.float32)

            # Ensure tensors are contiguous for optimal GPU performance
            x = x.contiguous()
            y_ = y_.contiguous()

            # Update learning rates
            for g in self.optimizer.param_groups:
                if lr_schedule == 'constant':
                    g['lr'] = lr
                elif lr_schedule in ('linear', 'linear_decay'):
                    if step_ndx < warmup_steps:
                        g['lr'] = lr * (step_ndx / max(1, warmup_steps))
                    elif step_ndx < warmup_steps + hold_steps:
                        g['lr'] = lr
                    else:
                        progress = (step_ndx - warmup_steps - hold_steps) / max(1, (max_steps - warmup_steps - hold_steps))
                        g['lr'] = lr * max(0.0, 1.0 - progress)
                else:
                    g['lr'] = self.lr_warmup_cosine_decay(step_ndx, warmup_steps=warmup_steps, hold=hold_steps, total_steps=max_steps, target_lr=lr)

            # zero the parameter gradients
            self.optimizer.zero_grad()

            # Get predictions for batch
            predictions = self.model(x)

            # Construct batch with only samples that have high loss
            neg_high_loss = predictions[(y == 0) & (predictions.squeeze() >= 0.001)]  # thresholds were chosen arbitrarily but work well
            pos_high_loss = predictions[(y == 1) & (predictions.squeeze() < 0.999)]
            y = torch.cat((y[(y == 0) & (predictions.squeeze() >= 0.001)], y[(y == 1) & (predictions.squeeze() < 0.999)]))
            y_ = y[..., None].to(torch.float32)
            predictions = torch.cat((neg_high_loss, pos_high_loss))

            # Set weights for batch
            if len(negative_weight_schedule) == 1:
                w = torch.ones(y.shape[0])*negative_weight_schedule[0]
                pos_ndcs = y == 1
                w[pos_ndcs] = 1
                w = w[..., None]
            else:
                if self.n_classes == 1:
                    w = torch.ones(y.shape[0])*negative_weight_schedule[step_ndx]
                    pos_ndcs = y == 1
                    w[pos_ndcs] = 1
                    w = w[..., None]

            if predictions.shape[0] != 0:
                # Ensure all tensors are contiguous before loss calculation
                predictions = predictions.contiguous()
                y_ = y_.contiguous()
                w = w.contiguous().to(self.device)
                
                # Do backpropagation, with gradient accumulation if the batch-size after selecting high loss examples is too small
                loss = self.loss(predictions, y_ if self.n_classes == 1 else y, w.to(self.device))
                loss = loss/accumulation_steps
                accumulated_samples += predictions.shape[0]

                if predictions.shape[0] >= 128:
                    accumulated_predictions = predictions
                    accumulated_labels = y_
                if accumulated_samples < 128:
                    accumulation_steps += 1
                    accumulated_predictions = torch.cat((accumulated_predictions, predictions))
                    accumulated_labels = torch.cat((accumulated_labels, y_))
                else:
                    loss.backward()
                    if clip_norm and clip_norm > 0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=clip_norm)
                    self.optimizer.step()
                    try:
                        if self.writer is not None:
                            self.writer.add_scalar('Train/loss', float(loss.detach().cpu().numpy()), step_ndx)
                            self.writer.add_scalar('Train/lr', float(self.optimizer.param_groups[0]['lr']), step_ndx)
                    except Exception:
                        pass
                    accumulation_steps = 1
                    accumulated_samples = 0

                    self.history["loss"].append(loss.detach().cpu().numpy())

                    # Compute training metrics and log them
                    fp = self.fp(accumulated_predictions, accumulated_labels if self.n_classes == 1 else y)
                    self.n_fp += fp
                    self.history["recall"].append(self.recall(accumulated_predictions, accumulated_labels).detach().cpu().numpy())

                    accumulated_predictions = torch.Tensor([]).to(self.device)
                    accumulated_labels = torch.Tensor([]).to(self.device)

            # Run validation and log validation metrics
            if step_ndx in val_steps and step_ndx > 1 and false_positive_val_data is not None:
                # Get false positives per hour with false positive data
                val_fp = 0
                for val_step_ndx, data in enumerate(false_positive_val_data):
                    with torch.no_grad():
                        x_val, y_val = data[0].to(self.device), data[1].to(self.device)
                        # Ensure validation tensors are contiguous
                        x_val = x_val.contiguous()
                        y_val = y_val.contiguous()
                        val_predictions = self.model(x_val)
                        val_fp += self.fp(val_predictions, y_val[..., None])
                val_fp_per_hr = (val_fp/val_set_hrs).detach().cpu().numpy()
                self.history["val_fp_per_hr"].append(val_fp_per_hr)
                # Compute optional val_loss/recall/accuracy on X_val for early stopping/monitoring
                val_loss = None
                val_recall = None
                val_acc = None
                if X_val is not None:
                    for val_step_ndx2, data2 in enumerate(X_val):
                        self.recall.reset()
                        with torch.no_grad():
                            x_val2, y_val2 = data2[0].to(self.device), data2[1].to(self.device)
                            # Ensure validation tensors are contiguous
                            x_val2 = x_val2.contiguous()
                            y_val2 = y_val2.contiguous()
                            val_pred2 = self.model(x_val2)
                            try:
                                val_loss = self.loss(val_pred2, y_val2[..., None] if self.n_classes == 1 else y_val2)
                                val_loss = float(val_loss.detach().cpu().numpy())
                            except Exception:
                                val_loss = None
                            try:
                                val_recall = float(self.recall(val_pred2, y_val2[..., None]).detach().cpu().numpy())
                            except Exception:
                                val_recall = None
                            try:
                                val_acc = float(self.accuracy(val_pred2, y_val2[..., None].to(torch.int64)))
                            except Exception:
                                val_acc = None
                # TensorBoard logging
                try:
                    if self.writer is not None:
                        self.writer.add_scalar('Val/fp_per_hour', float(val_fp_per_hr), step_ndx)
                        if val_loss is not None: self.writer.add_scalar('Val/loss', float(val_loss), step_ndx)
                        if val_recall is not None: self.writer.add_scalar('Val/recall', float(val_recall), step_ndx)
                        if val_acc is not None: self.writer.add_scalar('Val/accuracy', float(val_acc), step_ndx)
                except Exception:
                    pass
                # Early stopping monitor selection
                if monitor_key == 'val_fp_per_hr':
                    monitor_value = float(val_fp_per_hr)
                elif monitor_key == 'val_n_fp':
                    monitor_value = float(val_fp.detach().cpu().numpy()) if 'val_fp' in locals() else float(val_fp_per_hr)
                elif monitor_key == 'val_recall' and (val_recall is not None):
                    monitor_value = float(val_recall)
                else:
                    monitor_value = float(val_loss) if (val_loss is not None) else float(val_fp_per_hr)
                if stopper.step(monitor_value, model=self.model):
                    logging.info(f"Early stopping triggered at step {step_ndx} on monitor '{monitor_key}' (patience={patience}).")
                    self._early_stopped = True
                    break

            # Get recall on test clips
            if step_ndx in val_steps and step_ndx > 1 and positive_test_clips is not None:
                tp = 0
                fn = 0
                for val_step_ndx, data in enumerate(positive_test_clips):
                    with torch.no_grad():
                        x_val = data[0].to(self.device)
                        x_val = x_val.contiguous()  # Ensure contiguous for optimal performance
                        batch = []
                        for i in range(0, x_val.shape[1]-16, 1):
                            batch.append(x_val[:, i:i+16, :])
                        batch = torch.vstack(batch)
                        preds = self.model(batch)
                        if any(preds >= 0.5):
                            tp += 1
                        else:
                            fn += 1
                self.history["positive_test_clips_recall"].append(tp/(tp + fn))

            if step_ndx in val_steps and step_ndx > 1 and X_val is not None:
                # Get metrics for balanced test examples of positive and negative clips
                for val_step_ndx, data in enumerate(X_val):
                    with torch.no_grad():
                        x_val, y_val = data[0].to(self.device), data[1].to(self.device)
                        # Ensure validation tensors are contiguous
                        x_val = x_val.contiguous()
                        y_val = y_val.contiguous()
                        val_predictions = self.model(x_val)
                        val_recall = self.recall(val_predictions, y_val[..., None]).detach().cpu().numpy()
                        val_acc = self.accuracy(val_predictions, y_val[..., None].to(torch.int64))
                        val_fp = self.fp(val_predictions, y_val[..., None])
                self.history["val_accuracy"].append(val_acc.detach().cpu().numpy())
                self.history["val_recall"].append(val_recall)
                self.history["val_n_fp"].append(val_fp.detach().cpu().numpy())

            # Save models with a validation score above/below the 90th percentile
            # of the validation scores up to that point
            if step_ndx in val_steps and step_ndx > 1:
                if self.history["val_n_fp"][-1] <= np.percentile(self.history["val_n_fp"], 50) and \
                   self.history["val_recall"][-1] >= np.percentile(self.history["val_recall"], 5):
                    # logging.info("Saving checkpoint with metrics >= to targets!")
                    self.best_models.append(copy.deepcopy(self.model))
                    self.best_model_scores.append({"training_step_ndx": step_ndx, "val_n_fp": self.history["val_n_fp"][-1],
                                                   "val_recall": self.history["val_recall"][-1],
                                                   "val_accuracy": self.history["val_accuracy"][-1],
                                                   "val_fp_per_hr": self.history.get("val_fp_per_hr", [0])[-1]})
                    self.best_val_recall = self.history["val_recall"][-1]
                    self.best_val_accuracy = self.history["val_accuracy"][-1]

            if step_ndx == max_steps-1:
                break

# Separate function to convert onnx models to tflite format

def _print_training_data_summary(config):
    import os
    from pathlib import Path
    import numpy as np
    print("\n" + "#"*70)
    print("# TRAINING DATA SUMMARY")
    print("#"*70)
    print(f"Model Name       : {config.get('model_name')}")
    print(f"Model Type       : {config.get('model_type', 'dnn')}")
    print(f"Target Phrases   : {config.get('target_phrase')}")
    print(f"Steps            : {config.get('steps')}  |  Val every {config.get('val_steps')} steps")
    print(f"Layer Size       : {config.get('layer_size')}  |  Dropout {config.get('dropout_rate')}  |  GradClip {config.get('gradient_clip_norm')}")
    print(f"LR               : {config.get('learning_rate')}  |  Schedule: {config.get('learning_rate_schedule')}  |  Warmup {config.get('warmup_steps')}  |  Hold {config.get('hold_steps')}")
    print(f"Batch per class  : {config.get('batch_n_per_class')}")
    # Backgrounds
    bgs = config.get('background_paths', []) or []
    dups = config.get('background_paths_duplication_rate', []) or []
    print(f"Background paths ({len(bgs)}):")
    for i, bg in enumerate(bgs):
        try:
            n = len([e for e in os.scandir(bg) if e.is_file()])
        except Exception:
            n = "?"
        dup = dups[i] if i < len(dups) else 1
        print(f"  - {bg}  (files={n}, duplication_rate={dup})")
    # Features
    fdf = config.get('feature_data_files', {}) or {}
    if fdf:
        print("Feature data files:")
        for k, v in fdf.items():
            exists = os.path.exists(v)
            shape = None
            if exists and v.endswith('.npy'):
                try:
                    shape = np.load(v, mmap_mode='r').shape
                except Exception:
                    shape = "?"
            print(f"  - {k}: {v} (exists={exists}, shape={shape})")
    print("#"*70 + "\\n")

def convert_onnx_to_tflite(onnx_model_path, output_path):
    """Converts an ONNX version of an openwakeword model to the Tensorflow tflite format."""
    # imports
    import onnx
    from onnx_tf.backend import prepare
    import tensorflow as tf

    # Convert to tflite from onnx model
    onnx_model = onnx.load(onnx_model_path)
    tf_rep = prepare(onnx_model, device="CPU")
    with tempfile.TemporaryDirectory() as tmp_dir:
        tf_rep.export_graph(os.path.join(tmp_dir, "tf_model"))
        converter = tf.lite.TFLiteConverter.from_saved_model(os.path.join(tmp_dir, "tf_model"))
        tflite_model = converter.convert()

        logging.info(f"####\nSaving tflite mode to '{output_path}'")
        with open(output_path, 'wb') as f:
            f.write(tflite_model)

    return None

if __name__ == '__main__':
    # Get training config file
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--training_config",
        help="The path to the training config file (required)",
        type=str,
        required=True
    )
    parser.add_argument(
        "--generate_clips",
        help="Execute the synthetic data generation process",
        action="store_true",
        default="False",
        required=False
    )
    parser.add_argument(
        "--augment_clips",
        help="Execute the synthetic data augmentation process",
        action="store_true",
        default="False",
        required=False
    )
    parser.add_argument(
        "--overwrite",
        help="Overwrite existing openwakeword features when the --augment_clips flag is used",
        action="store_true",
        default="False",
        required=False
    )
    parser.add_argument(
        "--train_model",
        help="Execute the model training process",
        action="store_true",
        default="False",
        required=False
    )

    args = parser.parse_args()
    config = yaml.load(open(args.training_config, 'r').read(), yaml.Loader)

    # imports Piper for synthetic sample generation
    sys.path.insert(0, os.path.abspath(config["piper_sample_generator_path"]))
    from generate_samples import generate_samples

    # Define output locations
    config["output_dir"] = os.path.abspath(config["output_dir"])
    if not os.path.exists(config["output_dir"]):
        os.mkdir(config["output_dir"])
    if not os.path.exists(os.path.join(config["output_dir"], config["model_name"])):
        os.mkdir(os.path.join(config["output_dir"], config["model_name"]))

    positive_train_output_dir = os.path.join(config["output_dir"], config["model_name"], "positive_train")
    positive_test_output_dir = os.path.join(config["output_dir"], config["model_name"], "positive_test")
    negative_train_output_dir = os.path.join(config["output_dir"], config["model_name"], "negative_train")
    negative_test_output_dir = os.path.join(config["output_dir"], config["model_name"], "negative_test")
    feature_save_dir = os.path.join(config["output_dir"], config["model_name"])

    # Get paths for impulse response and background audio files
    rir_paths = [i.path for j in config["rir_paths"] for i in os.scandir(j)]
    background_paths = []
    if len(config["background_paths_duplication_rate"]) != len(config["background_paths"]):
        config["background_paths_duplication_rate"] = [1]*len(config["background_paths"])
    for background_path, duplication_rate in zip(config["background_paths"], config["background_paths_duplication_rate"]):
        background_paths.extend([i.path for i in os.scandir(background_path)]*duplication_rate)

    if args.generate_clips is True:
        # Generate positive clips for training
        logging.info("#"*50 + "\nGenerating positive clips for training\n" + "#"*50)
        if not os.path.exists(positive_train_output_dir):
            os.mkdir(positive_train_output_dir)
        n_current_samples = len(os.listdir(positive_train_output_dir))
        if n_current_samples <= 0.95*config["n_samples"]:
            generate_samples(
                text=config["target_phrase"], max_samples=config["n_samples"]-n_current_samples,
                batch_size=config["tts_batch_size"],
                noise_scales=[0.98], noise_scale_ws=[0.98], length_scales=[0.75, 1.0, 1.25],
                output_dir=positive_train_output_dir, auto_reduce_batch_size=True,
                file_names=[uuid.uuid4().hex + ".wav" for i in range(config["n_samples"])]
            )
            torch.cuda.empty_cache()
        else:
            logging.warning(f"Skipping generation of positive clips for training, as ~{config['n_samples']} already exist")

        # Generate positive clips for testing
        logging.info("#"*50 + "\nGenerating positive clips for testing\n" + "#"*50)
        if not os.path.exists(positive_test_output_dir):
            os.mkdir(positive_test_output_dir)
        n_current_samples = len(os.listdir(positive_test_output_dir))
        if n_current_samples <= 0.95*config["n_samples_val"]:
            generate_samples(text=config["target_phrase"], max_samples=config["n_samples_val"]-n_current_samples,
                             batch_size=config["tts_batch_size"],
                             noise_scales=[1.0], noise_scale_ws=[1.0], length_scales=[0.75, 1.0, 1.25],
                             output_dir=positive_test_output_dir, auto_reduce_batch_size=True)
            torch.cuda.empty_cache()
        else:
            logging.warning(f"Skipping generation of positive clips testing, as ~{config['n_samples_val']} already exist")

        # Generate adversarial negative clips for training
        logging.info("#"*50 + "\nGenerating negative clips for training\n" + "#"*50)
        if not os.path.exists(negative_train_output_dir):
            os.mkdir(negative_train_output_dir)
        n_current_samples = len(os.listdir(negative_train_output_dir))
        if n_current_samples <= 0.95*config["n_samples"]:
            adversarial_texts = config["custom_negative_phrases"]
            for target_phrase in config["target_phrase"]:
                adversarial_texts.extend(generate_adversarial_texts(
                    input_text=target_phrase,
                    N=config["n_samples"]//len(config["target_phrase"]),
                    include_partial_phrase=1.0,
                    include_input_words=0.2))
            generate_samples(text=adversarial_texts, max_samples=config["n_samples"]-n_current_samples,
                             batch_size=config["tts_batch_size"]//7,
                             noise_scales=[0.98], noise_scale_ws=[0.98], length_scales=[0.75, 1.0, 1.25],
                             output_dir=negative_train_output_dir, auto_reduce_batch_size=True,
                             file_names=[uuid.uuid4().hex + ".wav" for i in range(config["n_samples"])]
                             )
            torch.cuda.empty_cache()
        else:
            logging.warning(f"Skipping generation of negative clips for training, as ~{config['n_samples']} already exist")

        # Generate adversarial negative clips for testing
        logging.info("#"*50 + "\nGenerating negative clips for testing\n" + "#"*50)
        if not os.path.exists(negative_test_output_dir):
            os.mkdir(negative_test_output_dir)
        n_current_samples = len(os.listdir(negative_test_output_dir))
        if n_current_samples <= 0.95*config["n_samples_val"]:
            adversarial_texts = config["custom_negative_phrases"]
            for target_phrase in config["target_phrase"]:
                adversarial_texts.extend(generate_adversarial_texts(
                    input_text=target_phrase,
                    N=config["n_samples_val"]//len(config["target_phrase"]),
                    include_partial_phrase=1.0,
                    include_input_words=0.2))
            generate_samples(text=adversarial_texts, max_samples=config["n_samples_val"]-n_current_samples,
                             batch_size=config["tts_batch_size"]//7,
                             noise_scales=[1.0], noise_scale_ws=[1.0], length_scales=[0.75, 1.0, 1.25],
                             output_dir=negative_test_output_dir, auto_reduce_batch_size=True)
            torch.cuda.empty_cache()
        else:
            logging.warning(f"Skipping generation of negative clips for testing, as ~{config['n_samples_val']} already exist")

    # Set the total length of the training clips based on the ~median generated clip duration, rounding to the nearest 1000 samples
    # and setting to 32000 when the median + 750 ms is close to that, as it's a good default value
    n = 50  # sample size
    positive_clips = [str(i) for i in Path(positive_test_output_dir).glob("*.wav")]
    duration_in_samples = []
    for i in range(n):
        sr, dat = scipy.io.wavfile.read(positive_clips[np.random.randint(0, len(positive_clips))])
        duration_in_samples.append(len(dat))

    config["total_length"] = int(round(np.median(duration_in_samples)/1000)*1000) + 12000  # add 750 ms to clip duration as buffer
    if config["total_length"] < 32000:
        config["total_length"] = 32000  # set a minimum of 32000 samples (2 seconds)
    elif abs(config["total_length"] - 32000) <= 4000:
        config["total_length"] = 32000

    # Do Data Augmentation
    if args.augment_clips is True:
        if not os.path.exists(os.path.join(feature_save_dir, "positive_features_train.npy")) or args.overwrite is True:
            positive_clips_train = [str(i) for i in Path(positive_train_output_dir).glob("*.wav")]*config["augmentation_rounds"]
            positive_clips_train_generator = augment_clips_with_config(positive_clips_train, total_length=config["total_length"],
                                                           batch_size=config["augmentation_batch_size"],
                                                           background_clip_paths=background_paths,
                                                           RIR_paths=rir_paths, config=config)

            positive_clips_test = [str(i) for i in Path(positive_test_output_dir).glob("*.wav")]*config["augmentation_rounds"]
            positive_clips_test_generator = augment_clips_with_config(positive_clips_test, total_length=config["total_length"],
                                                          batch_size=config["augmentation_batch_size"],
                                                          background_clip_paths=background_paths,
                                                          RIR_paths=rir_paths, config=config)

            negative_clips_train = [str(i) for i in Path(negative_train_output_dir).glob("*.wav")]*config["augmentation_rounds"]
            negative_clips_train_generator = augment_clips_with_config(negative_clips_train, total_length=config["total_length"],
                                                           batch_size=config["augmentation_batch_size"],
                                                           background_clip_paths=background_paths,
                                                           RIR_paths=rir_paths, config=config)

            negative_clips_test = [str(i) for i in Path(negative_test_output_dir).glob("*.wav")]*config["augmentation_rounds"]
            negative_clips_test_generator = augment_clips_with_config(negative_clips_test, total_length=config["total_length"],
                                                          batch_size=config["augmentation_batch_size"],
                                                          background_clip_paths=background_paths,
                                                          RIR_paths=rir_paths, config=config)

            # Compute features and save to disk via memmapped arrays
            logging.info("#"*50 + "\nComputing openwakeword features for generated samples\n" + "#"*50)
            
            # CRITICAL GPU OPTIMIZATION: Use GPU for feature computation when available
            device_for_features = "cuda" if torch.cuda.is_available() else "cpu"
            n_cpus = os.cpu_count()
            if n_cpus is None:
                n_cpus = 1
            else:
                n_cpus = n_cpus//2
            
            # Use GPU if available, otherwise CPU with multiple workers
            ncpu_workers = 1 if device_for_features == "cuda" else n_cpus
            
            compute_features_from_generator(positive_clips_train_generator, n_total=len(os.listdir(positive_train_output_dir)),
                                            clip_duration=config["total_length"],
                                            output_file=os.path.join(feature_save_dir, "positive_features_train.npy"),
                                            device=device_for_features,  # Use cuda/cpu instead of gpu/cpu
                                            ncpu=ncpu_workers)

            compute_features_from_generator(negative_clips_train_generator, n_total=len(os.listdir(negative_train_output_dir)),
                                            clip_duration=config["total_length"],
                                            output_file=os.path.join(feature_save_dir, "negative_features_train.npy"),
                                            device=device_for_features,  # Use cuda/cpu instead of gpu/cpu
                                            ncpu=ncpu_workers)

            compute_features_from_generator(positive_clips_test_generator, n_total=len(os.listdir(positive_test_output_dir)),
                                            clip_duration=config["total_length"],
                                            output_file=os.path.join(feature_save_dir, "positive_features_test.npy"),
                                            device=device_for_features,  # Use cuda/cpu instead of gpu/cpu
                                            ncpu=ncpu_workers)

            compute_features_from_generator(negative_clips_test_generator, n_total=len(os.listdir(negative_test_output_dir)),
                                            clip_duration=config["total_length"],
                                            output_file=os.path.join(feature_save_dir, "negative_features_test.npy"),
                                            device=device_for_features,  # Use cuda/cpu instead of gpu/cpu
                                            ncpu=ncpu_workers)
        else:
            logging.warning("Openwakeword features already exist, skipping data augmentation and feature generation")

    # Create openwakeword model
    if args.train_model is True:
        # CRITICAL GPU OPTIMIZATION: Use GPU for AudioFeatures when available
        device_for_features = 'cuda' if torch.cuda.is_available() else 'cpu'
        F = openwakeword.utils.AudioFeatures(device=device_for_features)
        input_shape = np.load(os.path.join(feature_save_dir, "positive_features_test.npy")).shape[1:]

        oww = Model(n_classes=1, input_shape=input_shape, model_type=config["model_type"], layer_dim=config["layer_size"], seconds_per_example=1280*input_shape[0]/16000, dropout=float(config.get('dropout_rate', 0) or 0.0))

        # Create data transform function for batch generation to handle differ clip lengths (todo: write tests for this)
        def f(x, n=input_shape[0]):
            """Simple transformation function to ensure negative data is the appropriate shape for the model size"""
            if n > x.shape[1] or n < x.shape[1]:
                x = np.vstack(x)
                new_batch = np.array([x[i:i+n, :] for i in range(0, x.shape[0]-n, n)])
            else:
                return x
            return new_batch

        # Create label transforms as needed for model (currently only supports binary classification models)
        data_transforms = {key: f for key in config["feature_data_files"].keys()}
        label_transforms = {}
        for key in ["positive"] + list(config["feature_data_files"].keys()) + ["adversarial_negative"]:
            if key == "positive":
                label_transforms[key] = lambda x: [1 for i in x]
            else:
                label_transforms[key] = lambda x: [0 for i in x]

        # Add generated positive and adversarial negative clips to the feature data files dictionary
        config["feature_data_files"]['positive'] = os.path.join(feature_save_dir, "positive_features_train.npy")
        config["feature_data_files"]['adversarial_negative'] = os.path.join(feature_save_dir, "negative_features_train.npy")

        # Make PyTorch data loaders for training and validation data
        batch_generator = mmap_batch_generator(
            config["feature_data_files"],
            n_per_class=config["batch_n_per_class"],
            data_transform_funcs=data_transforms,
            label_transform_funcs=label_transforms
        )

        class IterDataset(torch.utils.data.IterableDataset):
            def __init__(self, generator):
                self.generator = generator

            def __iter__(self):
                return self.generator

        n_cpus = os.cpu_count()
        if n_cpus is None:
            n_cpus = 1
        else:
            n_cpus = n_cpus//2
        X_train = torch.utils.data.DataLoader(IterDataset(batch_generator),
                                              batch_size=None, num_workers=n_cpus, prefetch_factor=16)

        X_val_fp = np.load(config["false_positive_validation_data_path"])
        X_val_fp = np.array([X_val_fp[i:i+input_shape[0]] for i in range(0, X_val_fp.shape[0]-input_shape[0], 1)])  # reshape to match model
        X_val_fp_labels = np.zeros(X_val_fp.shape[0]).astype(np.float32)
        X_val_fp = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(torch.from_numpy(X_val_fp), torch.from_numpy(X_val_fp_labels)),
            batch_size=len(X_val_fp_labels)
        )

        X_val_pos = np.load(os.path.join(feature_save_dir, "positive_features_test.npy"))
        X_val_neg = np.load(os.path.join(feature_save_dir, "negative_features_test.npy"))
        labels = np.hstack((np.ones(X_val_pos.shape[0]), np.zeros(X_val_neg.shape[0]))).astype(np.float32)

        X_val = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(
                torch.from_numpy(np.vstack((X_val_pos, X_val_neg))),
                torch.from_numpy(labels)
                ),
            batch_size=len(labels)
        )

        # Run auto training
        best_model = oww.auto_train(
            X_train=X_train,
            X_val=X_val,
            false_positive_val_data=X_val_fp,
            config=config
        )

        # Export the trained model to onnx
        oww.export_model(model=best_model, model_name=config["model_name"], output_dir=config["output_dir"])

        # Convert the model from onnx to tflite format
        convert_onnx_to_tflite(os.path.join(config["output_dir"], config["model_name"] + ".onnx"),
                               os.path.join(config["output_dir"], config["model_name"] + ".tflite"))
