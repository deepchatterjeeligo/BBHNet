# BBHNet train
This project provides the utilities to train the BBHNet model. The following steps assume a local clone of the repository and a conda environment using the environment.yaml provided at the root of the project.

In order to install this project, navigate to the `train` directory, and install the dependencies of the project using `poetry`.
```
$ poetry install
Installing dependencies from lock file
.
.
.
```

It also assumes that the training data files are present. For a quickstart, set the `BASE_DIR` and `DATA_DIR` environment variables, where `DATA_DIR` contains the signal, background, and glitch dataset, e.g.
```
$ ls ${DATA_DIR}
glitches.h5  H1_background.h5  L1_background.h5  signals.h5
```
`BASE_DIR` points to the location to store models, logs, other files. During training it will look something like this,
```
$ ls ${BASE_DIR}
log  training

$ tail -f ${BASE_DIR}/log/train.log 
2022-11-09 08:56:53,572 - root - INFO - Duration 47.22s, Throughput 2168.6 samples/s
2022-11-09 08:57:07,389 - root - INFO - Train Loss: 4.5484e-04, Valid Loss: 1.7388e+01
2022-11-09 08:57:07,389 - root - INFO - === Epoch 11/200 ===
.
.
.
```

## Launching a run
The training executable is called `train`. For a quickstart, the training can be run using the configuration mentioned in the `pyproject.toml` in the root of the `sandbox` by running (assuming appropriate compute environment)
```
$ train --typeo .:train:resnet
```
This uses the project configuration stored in the `pyproject.toml` file at the root of the `sandbox`.

## Configuration 
Fine-grained control can be done using the command-line arguments. The output of `train --help` shows the configuration.
```
$ train --help
usage: main [-h] --hanford-background HANFORD_BACKGROUND --livingston-background LIVINGSTON_BACKGROUND
            --glitch-dataset GLITCH_DATASET --waveform-dataset WAVEFORM_DATASET --outdir OUTDIR
            --logdir LOGDIR --glitch-prob GLITCH_PROB --waveform-prob WAVEFORM_PROB --kernel-length
            KERNEL_LENGTH --sample-rate SAMPLE_RATE --batch-size BATCH_SIZE
            [--preprocessor PREPROCESSOR] [--max-epochs MAX_EPOCHS] [--init-weights INIT_WEIGHTS]
            [--lr LR] [--min-lr MIN_LR] [--decay-steps DECAY_STEPS] [--weight-decay WEIGHT_DECAY]
            [--early-stop EARLY_STOP] [--use-amp] [--profile] [--mean-snr MEAN_SNR] [--std-snr STD_SNR]
            [--min-snr MIN_SNR] [--highpass HIGHPASS] [--batches-per-epoch BATCHES_PER_EPOCH]
            [--fduration FDURATION] [--trigger-distance TRIGGER_DISTANCE] [--valid-frac VALID_FRAC]
            [--valid-stride VALID_STRIDE] [--device DEVICE] [--verbose]
            {resnet,bottleneck} ...

    Prepare a dataset of background, pre-computed glitches,
    and pre-computed event waveforms to train and validate
    a BBHNet architecture.

positional arguments:
  {resnet,bottleneck}

optional arguments:
  -h, --help            show this help message and exit
  --hanford-background HANFORD_BACKGROUND
                        Path to file containing background data for Hanford strain channel to train on.
                        Should be an HDF5 archive with an `"hoft"` dataset containing the strain data.
                        (default: None)
  --livingston-background LIVINGSTON_BACKGROUND
                        Path to file containing background data for Livingston strain channel to train
                        on. Should be an HDF5 archive with an `"hoft"` dataset containing the strain
                        data. (default: None)
  --glitch-dataset GLITCH_DATASET
                        Path to file containing short segments of data with non-Gaussian noise
                        transients. Should be an HDF5 archive with datasets `"<IFO ID>_glitches"`,
                        where `IFO_ID` is the short ID for each interferometer used for training (H1
                        and L1 for now). These glitches will be used to randomly replace the
                        corresponding interferometer channel during training with some probability
                        given by `glitch_prob`. Note that the samples selected for insertion on each
                        channel are sample independently, so glitches will be inserted into both
                        channels with probability `glitch_prob**2`. (default: None)
  --waveform-dataset WAVEFORM_DATASET
                        Path to file containing pre-computed gravitational wave polarization waveforms
                        for binary-blackhole merger events. Should be an HDF5 archive with a
                        `"signals"` dataset consisting of a tensor of shape `(num_waveforms,
                        num_polarizations, waveform_size)`. At data-loading time, extrinsic parameters
                        will be sampled for these events, which will be used to project them to
                        interferometer responses which will then be injected into the corresponding
                        channel with probability given by `waveform_prob`. Note that the samples
                        selected for injection will be chosen independently of those selected for
                        glitch insertion, so there is a nonzero likelihood that a waveform will be
                        injected over a glitch. This will still be marked as a positive event in the
                        training target. (default: None)
  --outdir OUTDIR       Location to save training artifacts like optimized weights, preprocessing
                        objects, and visualizations (default: None)
  --logdir LOGDIR
  --glitch-prob GLITCH_PROB
                        The probability with which each sample in a batch will have each of its
                        interferometer channels replaced with a glitch from the `glitch_dataset`.
                        (default: None)
  --waveform-prob WAVEFORM_PROB
                        The probability with which each sample in a batch will have a BBH waveform
                        injected into its background. (default: None)
  --kernel-length KERNEL_LENGTH
                        The length, in seconds, of each batch element to produce during iteration.
                        (default: None)
  --sample-rate SAMPLE_RATE
                        The rate at which all relevant input data has been sampled. (default: None)
  --batch-size BATCH_SIZE
                        Number of samples to over which to compute each gradient update during
                        training. (default: None)
  --preprocessor PREPROCESSOR
  --max-epochs MAX_EPOCHS
                        Maximum number of epochs over which to train. (default: 40)
  --init-weights INIT_WEIGHTS
                        Path to weights with which to initialize network. If left as `None`, network
                        will be randomly initialized. If `init_weights` is a directory, it will be
                        assumed that this directory contains a file called `weights.pt`. (default:
                        None)
  --lr LR               Learning rate to use during training. (default: 0.001)
  --min-lr MIN_LR       Minimum learning rate to decay to throughout training. (default: 1e-05)
  --decay-steps DECAY_STEPS
                        The number of steps over which to decay from lr to min_lr. (default: 10000)
  --weight-decay WEIGHT_DECAY
                        Amount of regularization to apply during training. (default: 0.0)
  --early-stop EARLY_STOP
                        Number of epochs without improvement in validation loss before training
                        terminates altogether. Ignored if `valid_data is None`. (default: 20)
  --use-amp
  --profile             Whether to generate a tensorboard profile of the training step on the first
                        epoch. This will make this first epoch slower. (default: False)
  --mean-snr MEAN_SNR   Mean SNR of the log-normal distribution from which to sample SNR values for
                        injected waveforms at data loading-time. (default: 8)
  --std-snr STD_SNR     Standard deviation of the log-normal distribution from which to sample SNR
                        values for injected waveforms at data loading-time. (default: 4)
  --min-snr MIN_SNR     Minimum SNR to use for SNR values for injected waveforms at data loading-time.
                        Samples drawn from the log-normal SNR distribution below this value will be
                        clipped to it. If left as `None`, all sampled SNRs will be used as-is.
                        (default: None)
  --highpass HIGHPASS   Minimum frequency over which to compute SNR values for waveform injection, in
                        Hz. If left as `None`, the SNR will be computed over all frequency bins.
                        (default: None)
  --batches-per-epoch BATCHES_PER_EPOCH
                        Number of gradient updates in between each validation step. Implicitly controls
                        the rate at which the learning can be decayed when training plateaus (since
                        this is based on validation scores). (default: None)
  --fduration FDURATION
                        Duration of the time domain filter used to whiten the data as a preprocessing
                        step. Note that `fduration / 2` seconds worth of data will be cropped from both
                        ends of the kernel of length `kernel_length` before passing it to the neural
                        network. (default: None)
  --trigger-distance TRIGGER_DISTANCE
                        The max length, in seconds, from the center of each waveform or glitch segment
                        that a sampled kernel's edge can fall. The default value of `0` means that
                        every kernel must contain the center of the corresponding segment (where the
                        "trigger" or its equivalent is assumed to lie). (default: 0)
  --valid-frac VALID_FRAC
                        Fraction of background, glitch, and waveform data to reserve for validation.
                        Glitches and waveforms will be sampled once each, with the center of the
                        segment in the center of the kernel, and either inserted or injected into
                        windows of background. (default: None)
  --valid-stride VALID_STRIDE
                        Distance, in seconds, between windows taken from the validation timeseries to
                        pass to the network for validation. (default: None)
  --device DEVICE       Device on which to perform training. Either `"cpu"`, `"cuda"`, or
                        `"cuda:<device index>"` to train on a specific GPU. (default: cpu)
  --verbose             Controls log verbosity, with the default value of `False` logging at level
                        `INFO`, and `True` logging at level `DEBUG`. (default: False)
```
