# DisruptCNN
DisruptCNN presents code using the Temporal Convolutional Network (a convolutional neural network with dilated convolutions, see https://github.com/locuslab/TCN) to predict tokamak disruption using raw, high temporal resolution, multi-scale diagnostic data. Specifically, the Electron Cyclotron Emission imaging (ECEi) diagnostic at the DIII-D tokamak was used. Paper here: https://arxiv.org/abs/1911.00149, please cite using:

**Data selection (from paper):** The paper uses only "good ECEi data (**SNR > 3**)" — i.e. shots whose *minimum* SNR over the entire signal is greater than 3. Shots with SNR ≤ 3 are excluded as low-quality. The shot list files have an "SNR min" column; use `snr_min_threshold=3.0` in `EceiDataset` to replicate this filter (see `loader.py`).

	@article{ChurchillNeurIPS2019,
		author    = {R.M. Churchill and the DIII-D team},
		title     = {Deep convolutional neural networks for multi-scale time-series classification and application to disruption prediction in fusion devices},
		journal   = {arXiv:1911.00149v1},
		year      = {2019},
	}
    
