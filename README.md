# Bombay Beach Autonomous Artificial Artist

This is a project to create a prototype of an [autonomous artificial artist (AAA)](https://medium.com/@genekogan/artist-in-the-cloud-8384824a75c7) distributed across a collection of stations in Bombay Beach containing a TV, camera, and a Raspberry Pi. The goal is to make a federated generative model (VAE) trained on incoming camera data (or possibly other sensors), such that the data is never collected in one place. This is a limited prototype as the model is not encrypted or decentralized in anyway, but it achieves the irreproducibility criteria and is an important first step.

The first design is to instantiate and train on an ongoing basis a [federated VAE using PySyft](https://github.com/OpenMined/PySyft).


# Setting up Raspberry Pi for PySyft

Boot up Pi. To find your device, search for devices connected to network with `arp -a` and any Pis mac address starts with `b8:27:eb`. Make note of yours and ssh into the Pi `ssh pi@<your_rpi_ip>`.

### Setup PyTorch on the Pi

Install pre-requisites for PyTorch.

    sudo apt install libopenblas-dev libblas-dev libatlas-base-dev m4 cmake cython python3-dev python3-yaml python3-setuptools python3-pip

Download a wheel to Install PyTorch for your Pi's specific architecture (can be found with `uname -a`). For Pi 4 (armv71), [this repository](https://github.com/sungjuGit/Pytorch-and-Vision-for-Raspberry-Pi-4B) works:

    git clone https://github.com/sungjuGit/Pytorch-and-Vision-for-Raspberry-Pi-4B
    cd Pytorch-and-Vision-for-Raspberry-Pi-4B
    pip3 install torch-1.4.0a0+f43194e-cp37-cp37m-linux_armv7l.whl
    pip3 install torchvision-0.5.0a0+9cdc814-cp37-cp37m-linux_armv7l.whl 

*Note* If PySyft fails later for your version of PyTorch, fall back to [this build of PyTorch 1.3](https://discuss.pytorch.org/t/pytorch-1-3-wheels-for-raspberry-pi-python-3-7/58580) instead.

### Install PySyft

    pip3 install Flask flask-socketio lz4 msgpack websockets zstd tblib
    pip3 install syft

or if you are installing from source:

    git clone https://github.com/OpenMined/PySyft
    cd PySyft
    pip3 install . --no-deps

If you get an error [about `syft_proto`](https://github.com/OpenMined/PySyft/issues/2921), install that separately with `pip3 install syft-proto`.


# Usage Instructions

Launch 4 workers in 4 separate terminals.

    python3 run_worker.py --port 8182 --host localhost --id 0 --verbose
    python3 run_worker.py --port 8183 --host localhost --id 1 --verbose
    python3 run_worker.py --port 8184 --host localhost --id 2 --verbose
    python3 run_worker.py --port 8185 --host localhost --id 3 --verbose

Then run the main program.

    python3 main.py