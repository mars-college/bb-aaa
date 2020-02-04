# Bombay Beach Autonomous Artificial Artist

This is a project to create a prototype of an [autonomous artificial artist (AAA)](https://medium.com/@genekogan/artist-in-the-cloud-8384824a75c7) distributed across a collection of stations in Bombay Beach containing a TV, camera, and a Raspberry Pi. The goal is to make a federated generative model (VAE) trained on incoming camera data (or possibly other sensors), such that the data is never collected in one place. This is a limited prototype as the model is not encrypted or decentralized in anyway, but it achieves the irreproducibility criteria and is an important first step.

The first design is to instantiate and train on an ongoing basis a [federated VAE using PySyft](https://github.com/OpenMined/PySyft).


# Setting up Raspberry Pi for PySyft and PyGrid

Boot up Pi. To find your device, search for devices connected to network with `arp -a` and any Pis mac address starts with `b8:27:eb`. Make note of yours and ssh into the Pi `ssh pi@<your_rpi_ip>`.

### Setup PyTorch on the Pi

Install pre-requisites for PyTorch.

    sudo apt install libopenblas-dev libblas-dev libatlas-base-dev m4 cmake cython python3-dev python3-yaml python3-setuptools python3-pip

### Install PySyft and PyGrid

If we install PyGrid library, we will get PySyft installed as a dependency. Since Grid and Syft is currently under development, it is recommended to install PyGrid from the BRAHMAN-AI fork:


    git clone https://github.com/brahman-ai/PyGrid
    cd PyGrid
    pip3 install -r requirements.txt
    python3 setup.py install

Executing `pip3 list | grep "syft\|grid"` should display an output like:


    grid               0.1.0.0a1.dev440
    syft               0.2.2a1-brahman  /home/pi/brahman/venv/src/syft
    syft-proto         0.1.0a1.post38

If you get an error [about `syft_proto`](https://github.com/OpenMined/PySyft/issues/2921), install that separately with `pip3 install syft-proto`.


# Usage Instructions

Launch 4 workers in 4 separate terminals.

    python3 run_worker_mnist.py --port 8182 --host localhost --id 0 --verbose
    python3 run_worker_mnist.py --port 8183 --host localhost --id 1 --verbose
    python3 run_worker_mnist.py --port 8184 --host localhost --id 2 --verbose
    python3 run_worker_mnist.py --port 8185 --host localhost --id 3 --verbose

Then run the main program.

    python3 main_mnist.py

# Setting up a Grid Gateway

Install Gateway requirements:

    cd PyGrid/gateway/
    pip3 install -r requirements.txt

Now we can start a gateway with:

    python3 gateway.py --start_local_db --port=5000

The IP of the Pi where the Gateway is executed will be needed when launching Grid Nodes, as shown in the instructions below.

# Setting up a Grid Node

Install Node requirements:

    cd PyGrid/app/websocket
    pip3 install -r requirements.txt

Now we can start a node, providing the Gateway IP and port:

    python3 websocket_app.py --start_local_db --id=bob --port=3000 --gateway_url=http://<GATEWAY_IP>:5000


# Grid usage example


### Public Grid Network

    import torch as th
    import syft as sy

    hook = sy.TorchHook(th)

    gateway = sy.grid.public_grid.PublicGridNetwork(hook, gateway_url="http://<GATEWAY_IP>:5000")

    bob = sy.grid.public_grid.NodeClient(hook, address="http://<BOB_NODE_IP>:3000")
    bob.connect()

    alice = sy.grid.public_grid.NodeClient(hook, address="http://<ALICE_NODE_IP>:3000")
    alice.connect()

    x = th.tensor([1, 2, 3, 4, 5]).tag("#bombay_beach_data").send(bob)
    y = th.tensor([10, 20, 30, 40, 50]).tag("#bombay_beach_data").send(alice)

    gateway.search("#bombay_beach_data")
    #{'bob': [(Wrapper)>[PointerTensor | me:36135518786 -> bob:28833842327]
    #    Tags: #bombay_beach_data
    #    Shape: torch.Size([5])], 'alice': [(Wrapper)>[PointerTensor | me:71446878279 -> alice:1834179188]
    #    Tags: #bombay_beach_data
    #    Shape: torch.Size([5])]}
