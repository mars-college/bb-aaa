## Bombay Beach Autonomous Artificial Artist

This is a project to create a prototype of an [autonomous artificial artist (AAA)](https://medium.com/@genekogan/artist-in-the-cloud-8384824a75c7) distributed across a collection of stations in Bombay Beach containing a TV, camera, and a Raspberry Pi. The goal is to make a federated generative model (VAE) trained on incoming camera data (or possibly other sensors), such that the data is never collected in one place. This is a limited prototype as the model is not encrypted or decentralized in anyway, but it achieves the irreproducibility criteria and is an important first step.

The first design is to instantiate and train on an ongoing basis a [federated VAE using PySyft](https://github.com/OpenMined/PySyft).

More documentation soon.
