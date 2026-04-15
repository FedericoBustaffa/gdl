# Introduction to TensorFlow, Keras, and PyTorch

Introduction to TensorFlow, Keras, and PyTorch for the course on Intelligent Systems and Pattern Recognition (2025), M.Sc. in Computer Science, University of Pisa.

## Usage

The Jupyter notebooks in this repository introduce the fundamental concepts of tensor manipulation and neural networks using TensorFlow and Keras. To run the notebooks you have three options:

1. Docker/Podman (the *good*)
2. Google Colab (the *bad*)
3. Local machine (the *ugly*)

### Docker/Podman

Using Docker or Podman, you can run the notebooks in a containerized environment. This is the recommended approach as it ensures that all dependencies are correctly installed and isolated from your local machine. There is also a `run.sh` script that automatically builds and runs the image.

```bash
git clone https://github.com/di-unipi/gdl-lab
cd gdl-lab
docker build -t gdl-lab .
docker run -p 8888:8888 -v $(pwd)/notebooks:/app/notebooks:z -v $(pwd)/_static:/app/_static:z gdl-lab
```

```bash
git clone https://github.com/di-unipi/gdl-lab
cd gdl-lab
podman build -t gdl-lab .
podman run -p 8888:8888 -v $(pwd)/notebooks:/app/notebooks:z -v $(pwd)/_static:/app/_static:z gdl-lab
```

### Google Colab

[Google Colab](https://colab.research.google.com/) is a free cloud service that supports Jupyter notebooks. It provides a simple way to run the notebooks without any installation. TensorFlow and Keras should be pre-installed in the Colab environment, so you can run the notebooks directly without any additional setup.

Just look for this repository (`di-unipi/gdl-lab`) in the GitHub tab of Google Colab. You can open the notebooks directly from there.

### Local Machine

If you are brave enough to mess with your local Python environment, either using Conda, virtualenv, you can run the notebooks directly on your machine. You can install the necessary requirements as follows:

```bash
git clone https://github.com/di-unipi/gdl-lab
cd gdl-lab
pip install -r requirements.txt
```

Ensure that you're selecting the correct Python kernel in your Jupyter Notebook and good luck! 🤞

![As always XKCD knows it better.](https://imgs.xkcd.com/comics/python_environment.png)

## Contributors

- [Riccardo Massidda](https://pages.di.unipi.it/massidda/) (2025 — …)
- [Valerio De Caro](https://vdecaro.github.io/) (2023 — …)
