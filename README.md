<p align="center">
  <a href="https://github.com/SeatizenDOI/drone-inference/graphs/contributors"><img src="https://img.shields.io/github/contributors/SeatizenDOI/drone-inference" alt="GitHub contributors"></a>
  <a href="https://github.com/SeatizenDOI/drone-inference/network/members"><img src="https://img.shields.io/github/forks/SeatizenDOI/drone-inference" alt="GitHub forks"></a>
  <a href="https://github.com/SeatizenDOI/drone-inference/issues"><img src="https://img.shields.io/github/issues/SeatizenDOI/drone-inference" alt="GitHub issues"></a>
  <a href="https://github.com/SeatizenDOI/drone-inference/blob/master/LICENSE"><img src="https://img.shields.io/github/license/SeatizenDOI/drone-inference" alt="License"></a>
  <a href="https://github.com/SeatizenDOI/drone-inference/pulls"><img src="https://img.shields.io/github/issues-pr/SeatizenDOI/drone-inference" alt="GitHub pull requests"></a>
  <a href="https://github.com/SeatizenDOI/drone-inference/stargazers"><img src="https://img.shields.io/github/stars/SeatizenDOI/drone-inference" alt="GitHub stars"></a>
  <a href="https://github.com/SeatizenDOI/drone-inference/watchers"><img src="https://img.shields.io/github/watchers/SeatizenDOI/drone-inference" alt="GitHub watchers"></a>
</p>
<div align="center">
  <a href="https://github.com/SeatizenDOI/drone-inference">View framework</a>
  ·
  <a href="https://github.com/SeatizenDOI/drone-inference/issues">Report Bug</a>
  ·
  <a href="https://github.com/SeatizenDOI/drone-inference/issues">Request Feature</a>
</div>

<div align="center">

# Drone Inference

</div>

Drone-inference is used to apply a multilabel model from hugging-face like [DinoVdeau](https://github.com/SeatizenDOI/DinoVdeau)

This repository works with sessions that contain an orthophoto in a folder called PROCESSED_DATA/PHOTOGRAMMETRY

At the end of the process, this code will create a PROCESSED_DATA/IA folder that contains prediction and score files in CSV format, as well as raster files of the predictions.


* [Docker](./docker.README.md)
* [Installation](#installation)
* [Usage](#usage)
* [Contributing](#contributing)
* [License](#license)


## Installation

All the sessions was proceed on this hardware (Dell Precision 7770):

- Intel Core i9-12950HX
- 64.0 Gio RAM
- NVIDIA RTX A3000 12GB Laptop GPU

To ensure a consistent environment for all users, this project uses a Conda environment defined in a `inference_env.yml` file. Follow these steps to set up your environment:

I wish you good luck for the installation.

1. **Setup Nvidia Driver:** Please install your [nvidia driver](https://www.nvidia.com/fr-fr/drivers/unix/).

2. **Download and install CudaToolkit:** Check [TensorRT requirements](https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html) to get the good [CudaToolkit](https://developer.nvidia.com/cuda-toolkit).

3. **Install Conda:** If you do not have Conda installed, download and install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/products/distribution).

4. **Create the Conda Environment:** Navigate to the root of the project directory and run the following command to create a new environment from the `inference.yml` file:
   ```bash
   conda env create -f drone_inference_env.yml
   ```

5. **Activate the Environment:** Once the environment is created, activate it using:
   ```bash
   conda activate drone_inference_env
   ```


## Usage

To run the workflow, navigate to the project root and execute:

```bash
python inference.py [OPTIONS]
```


## Contributing

Contributions are welcome! To contribute, please follow these steps:

1. Fork the repository.
2. Create a new branch for your feature or fix.
3. Commit your changes with clear, descriptive messages.
4. Push your branch and submit a pull request.

## License

This framework is distributed under the wtfpl license. See `LICENSE` for more information.

<div align="center">
  <img src="https://github.com/SeatizenDOI/.github/blob/main/images/logo_partenaire.png?raw=True" alt="Partenaire logo" width="700">
</div>