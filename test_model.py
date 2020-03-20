from threading import Thread
import argparse
import json
import time
import websockets
import functools
import queue
from PIL import Image
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from models import VAE, ConvVAE





##########
import fnmatch
import os
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from tqdm import tqdm
import random
from sklearn.decomposition import PCA
from random import shuffle

lfw_path = "/Users/gene/Downloads/lfw-deepfunneled"

people = {}
for root, dirnames, filenames in os.walk(lfw_path):
    for filename in fnmatch.filter(filenames, '*.jpg'):
        name = root.split('/')[-1]
        if name in people:
            people[name].append(os.path.join(root, filename))
        else:
            people[name] = [os.path.join(root, filename)]

all_names = [name for name in people]
nums = [len(people[name]) for name in people]
idx = list(reversed(np.argsort(nums)))
top_names = [all_names[i] for i in idx]
person = people[top_names[0]]

#######



def search_for_face(frame, faceCascade):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(50, 50)
    )
    n_faces = len(faces)
    if n_faces == 0:
        return None 
    else:  
        areas = [face[2] * face[3] for face in faces]
        idx_max_area = areas.index(max(areas))
        return faces[idx_max_area]


def main():
    parser = argparse.ArgumentParser(description="Test VAE and ConvVAE")
    #parser.add_argument("--image_nc", type=int, default=1, help="number of channels")    
    parser.add_argument("--plain_vae", action="store_true", help="if set, uses simple fully-connected VAE, otherwise uses ConvVAE")
    parser.add_argument("--image_dim", type=int, default=64, help="image size")    
    parser.add_argument("--h_dim", type=int, default=300, help="neurons in hidden layers")    
    parser.add_argument("--z_dim", type=int, default=20, help="dimensionality of latent space")    
    parser.add_argument("--input", type=str, choices=['cam', 'picam', 'lfw'], default='lfw', help="input data source (cam, picam, lfw)")
    parser.add_argument("--cascadeFile", type=str, default='haarcascade_frontalface_default.xml', help="location of haar cascade file")
    parser.add_argument("--batch_size", type=int, default=8, help="number of batches")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="learning rate of optimizer")    
    parser.add_argument("--verbose", action="store_true", help="if set, websocket server worker will be started in verbose mode")
    args = parser.parse_args()
    
    faceCascade = cv2.CascadeClassifier(args.cascadeFile)

    if args.input == 'picam':
        import picamera

    image_dim, batch_size = args.image_dim, args.batch_size

    # initialize model and workers
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # create model
    if args.plain_vae:
        print('Making a plain fully-connected VAE with h-dimension=%d, z-dimension=%d' % (args.h_dim, args.z_dim))
        model = VAE(image_dim=args.image_dim, h_dim=args.h_dim, z_dim=args.z_dim).to(device)
    else:
        print('Making a convolutional VAE with z-dimension=%d' % args.z_dim)
        model = ConvVAE(z_dim=args.z_dim).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    
    batch = 1
    captures = []
    model.train()

    if args.input == 'cam':
        camera = cv2.VideoCapture(0)


    while True:
        
        if args.input == 'cam':
            print('Get camera input')
            if not camera.isOpened():
                print('Unable to load camera.')
                time.sleep(2)
                return
            _, image = camera.read()

        elif args.input == 'picam':
            print('Get picamera input')
            stream = io.BytesIO()
            with picamera.PiCamera() as camera:
                camera.start_preview()
                time.sleep(1)
                camera.capture(stream, format='jpeg')
            stream.seek(0)
            image = np.array(Image.open(stream))

        elif args.input == 'lfw':
            print('Get lfw input')
            idx_random = int(len(person) * random.random())
            image = Image.open(person[idx_random]).convert('RGB')
            image = np.array(image)

        face = search_for_face(image, faceCascade)
        
        if face is not None:
            ih, iw = image.shape[0:2]
            x, y, w, h = face

            x1, x2 = int(x - 0.33 * w), int(x + 1.33 * w)
            y1, y2 = int(y - 0.33 * h), int(y + 1.33 * h)

            if x1 < 0 or x2 >= iw or y1 < 0 or y2 >= ih:
                print('Face not fully inside frame')

            else:
                print('Found face inside', y1, y2, x1, x2)

                # pick out face image
                image = image[y1:y2, x1:x2]
                image = cv2.resize(image, (64, 64), interpolation=cv2.INTER_CUBIC)
                image = np.array(image).astype(np.float32) / 255.

                captures.append(image)

        else:
            print('No face found')


        if len(captures) == batch_size:

            # forward pass
            data = np.array(captures)
            x = torch.tensor(np.mean(data, -1))
            x_reconst, mu, log_var = model(x)
            
            # get loss
            reconst_loss = F.binary_cross_entropy(x_reconst, x, size_average=False)
            kl_div = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
            
            # backward pass
            loss = reconst_loss + kl_div
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print("Reconstruction Loss: {:.4f}, KL Divergence: {:.4f}".format(reconst_loss, kl_div))

            # draw first reconstructed sample 
            if batch % 20 == 0:
                reconstructed_samples = x_reconst
                batch_size = reconstructed_samples.shape[0]
                reconstructed_samples = reconstructed_samples.reshape([batch_size, image_dim, image_dim])
                reconstructed_samples = (255 * reconstructed_samples.detach().numpy()).astype(np.uint8)
                Image.fromarray(reconstructed_samples[0]).save('test_reconstructed_%03d.png' % batch)
            
            batch += 1
            captures = []


    

if __name__== "__main__":
    main()