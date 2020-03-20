from threading import Thread
from socket import gethostbyname_ex, gethostname
from enum import Enum
from sys import platform
from PIL import Image
import argparse
import time
import asyncio
import websockets
import io
import os
import json
import numpy as np
import cv2
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import syft as sy
from syft.workers.websocket_server import WebsocketServerWorker

# USE_PICAMERA = False





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
import glob

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
files = glob.glob('%s/*/*.jpg' % lfw_path)
person = files
#######








# if USE_PICAMERA:
#     import picamera
# else:
#     cascPath = '/Users/gene/Downloads/Webcam-Face-Detect-master/haarcascade_frontalface_default.xml'
#     faceCascade = cv2.CascadeClassifier(cascPath)


def get_local_ip_address():
    if platform == "linux" or platform == "linux2":
        from pyroute2 import IPRoute
        ip = IPRoute()
        label = 'wlan0' if ip.get_addr(label='wlan0') != () else 'eth0'
        local_ip_address = dict(ip.get_addr(label=label)[0]['attrs'])['IFA_LOCAL']
    elif platform == "darwin":
        from socket import gethostbyname_ex, gethostname
        local_ip_address = gethostbyname_ex(gethostname())[-1][-1]
    return local_ip_address


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


class Worker:

    class Mode(Enum):
        CAPTURING = 1
        READY_TO_UPDATE = 2
        WAITING = 3

    def __init__(self, name, worker_ip, conductor_ip, conductor_port, syft_port, batch_size, data_input, hook):
        self.name = name
        self.worker_ip = worker_ip
        self.conductor_ip = conductor_ip
        self.conductor_port = conductor_port
        self.syft_port = syft_port
        self.batch_size = batch_size
        self.input = data_input
        self.hook = hook

        self.registered = False
        self.active = False

        self.captures = []
        self.n_updates = 0
        self.socket_uri = 'ws://%s:%d' % (conductor_ip, conductor_port)

        if self.input == 'cam':
            self.camera = cv2.VideoCapture(0)

    def set_mode(self, mode):
        self.mode = mode
            
    def setup_data(self, image_size):
        self.image_size = image_size
        trans = transforms.ToTensor() 
        mnist_loader = torch.utils.data.DataLoader(
            dataset=datasets.MNIST(root='./data', train=True, transform=trans, download=True), 
            batch_size=self.batch_size, shuffle=True)
        self.mnist_iterator = iter(mnist_loader)

    def ready_to_update(self):
        ready = (self.num_captures() == self.batch_size)
        return ready

    def get_next_batch(self):
        print('Get next batch...')
        # for c, cap in enumerate(self.captures):
        #     print(cap.shape, cap.dtype)
        #     Image.fromarray((255 * cap).astype(np.uint8)).save('batch%03d.png'%(c+1))
        data = np.array(self.captures)
        data = torch.tensor(np.mean(data, -1))
        self.x_ptr = data.tag('#x').send(self.local_worker)
        
    def activate(self):
        kwargs = { "hook": self.hook,
            "id": self.name,
            "host": "0.0.0.0",
            "port": self.syft_port }
        self.active = True
        self.local_worker = WebsocketServerWorker(**kwargs)
        self.local_worker.start()

    async def capture(self, faceCascade):
        if self.input == 'cam':
            print('Get camera input')
            if not self.camera.isOpened():
                print('Unable to load camera.')
                time.sleep(2)
                return
            _, image = self.camera.read()

        elif self.input == 'picam':
            print('Get picamera input')
            stream = io.BytesIO()
            with picamera.PiCamera() as camera:
                camera.start_preview()
                time.sleep(1)
                camera.capture(stream, format='jpeg')
            stream.seek(0)
            image = np.array(Image.open(stream))

        elif self.input == 'lfw':
            print('Get lfw input')
            idx_random = int(len(person) * random.random())
            print(idx_random, len(person))
            print('load %s'%person[idx_random])
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
                return False 

            print('Found face inside', y1, y2, x1, x2)

            # pick out face image
            image = image[y1:y2, x1:x2]
            image = cv2.resize(image, (64, 64), interpolation=cv2.INTER_CUBIC)
            image = np.array(image).astype(np.float32) / 255.

            self.captures.append(image)
            return True

        else:
            print('No face found')
            return False

    def num_captures(self):
        return len(self.captures)

    async def try_register(self):
        print('Register with conductor')
        async with websockets.connect(self.socket_uri) as websocket:
            message = json.dumps({'action': 'register', 'name': self.name, 'syft_port': self.syft_port, 'host': get_local_ip_address()})
            await websocket.send(message)
            result = json.loads(await websocket.recv())
            if result['success']:
                print('Registration successful.')
                self.registered = True

    async def request_update(self):
        print('Request to make a model update')
        async with websockets.connect(self.socket_uri) as websocket:
            message = json.dumps({'action': 'request_model', 'name': self.name})
            await websocket.send(message)
            result = json.loads(await websocket.recv())
            if result['success']:
                print('Ready to make an update...')
    
    async def check_if_in_queue(self):
        async with websockets.connect(self.socket_uri) as websocket:
            message = json.dumps({'action': 'check_if_in_queue', 'name': self.name})
            await websocket.send(message)
            result = json.loads(await websocket.recv())
            return result['in_queue']



async def update_worker(loop: asyncio.AbstractEventLoop, worker, faceCascade) -> None:

    while not worker.registered:

        await worker.try_register()
        time.sleep(1)

    while worker.active:

        if worker.mode == Worker.Mode.CAPTURING:
            await worker.capture(faceCascade)
            if worker.ready_to_update():
                worker.set_mode(Worker.Mode.READY_TO_UPDATE)

        elif worker.mode == Worker.Mode.READY_TO_UPDATE:
            worker.get_next_batch()
            await worker.request_update()
            worker.set_mode(Worker.Mode.WAITING)

        elif worker.mode == Worker.Mode.WAITING:
            done = await worker.check_if_in_queue()
            if done:
                worker.n_updates += 1
                worker.captures = []
                worker.set_mode(Worker.Mode.CAPTURING)

        time.sleep(1)
    

def start_background_loop(loop: asyncio.AbstractEventLoop, worker, faceCascade) -> None:
    asyncio.set_event_loop(loop)
    asyncio.run_coroutine_threadsafe(update_worker(loop, worker, faceCascade), loop)
    loop.run_forever()


def main():
    parser = argparse.ArgumentParser(description="Run websocket server worker.")
    parser.add_argument("--name", type=str, help="name of the worker, e.g. worker001")
    parser.add_argument("--worker_ip", type=str, default="auto", help="ip address of worker (if \"auto\", then find automatically, otherwise override")
    parser.add_argument("--conductor_ip", type=str, default="0.0.0.0", help="host for the socket connection")
    parser.add_argument("--conductor_port", type=int, default=8765, help="port number of the websocket server worker, e.g. --conductor_port 8765")
    parser.add_argument("--syft_port", type=int, help="port for syft worker")
    parser.add_argument("--batch_size", type=int, default=8, help="number of batches")
    parser.add_argument("--input", type=str, choices=['cam', 'picam', 'lfw'], default='lfw', help="input data source (cam, picam, lfw)")
    parser.add_argument("--cascadeFile", type=str, default='haarcascade_frontalface_default.xml', help="location of haar cascade file")
    parser.add_argument("--verbose", action="store_true", help="if set, websocket server worker will be started in verbose mode")
    args = parser.parse_args()
    
    faceCascade = cv2.CascadeClassifier(args.cascadeFile)

    if args.input == 'picam':
        import picamera
    
    # setup worker
    hook = sy.TorchHook(torch)
    worker_ip = get_local_ip_address() if args.worker_ip == 'auto' else args.worker_ip
    worker = Worker(args.name, worker_ip, args.conductor_ip, args.conductor_port, args.syft_port, args.batch_size, args.input, hook)    
    worker.setup_data(64 * 64)
    worker.set_mode(Worker.Mode.CAPTURING)

    # setup update thread
    loop = asyncio.new_event_loop()
    update_thread = Thread(target=start_background_loop, args=(loop, worker, faceCascade,), daemon=True)
    update_thread.start()

    # launch syft worker
    worker.activate()

    
if __name__== "__main__":
    main()