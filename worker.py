from threading import Thread
from socket import gethostbyname_ex, gethostname
from enum import Enum
from sys import platform
from PIL import ImageDraw, ImageFont, Image
import argparse
import time
import asyncio
import websockets
import io
import os
from datetime import datetime
import time
import json
import numpy as np
import cv2
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import syft as sy
from syft.workers.websocket_server import WebsocketServerWorker
from util import *

parser = argparse.ArgumentParser(description="Run websocket server worker.")
parser.add_argument("--name", type=str, default=None, help="name of the worker, e.g. worker001, generates random name if not set")
parser.add_argument("--worker_ip", type=str, default="auto", help="ip address of worker (if \"auto\", then find automatically, otherwise override")
parser.add_argument("--conductor_ip", type=str, default="0.0.0.0", help="host for the socket connection")
parser.add_argument("--conductor_port", type=int, default=8765, help="port number of the websocket server worker, e.g. --conductor_port 8765")
# parser.add_argument("--syft_port", type=int, help="port for syft worker")
parser.add_argument("--image_size", type=int, default=64, help="dimension of input images")
parser.add_argument("--batch_size", type=int, default=8, help="number of batches")
parser.add_argument("--input", type=str, choices=['cam', 'picam', 'lfw'], default='lfw', help="input data source (cam, picam, lfw)")
parser.add_argument("--cascadeFile", type=str, default='haarcascade_frontalface_default.xml', help="location of haar cascade file")
parser.add_argument("--verbose_event", action="store_true", help="if set, events are verbose")
parser.add_argument("--verbose_capture", action="store_true", help="if set, captures are verbose")
parser.add_argument("--verbose_draw", action="store_true", help="if set, draws are verbose")
parser.add_argument("--hide_gui", action="store_true", help="if set, don't draw gui")


# gui settings
t0 = time.time()
gui = None
FONT = 'verdana.ttf'
GUI_W, GUI_H = 1280, 720
FONT_SIZE_1 = 28
FONT_SIZE_2 = 18
FONT_SIZE_3 = 14
FRAME_W, FRAME_H = 640, 360
FRAME_X, FRAME_Y = 50, 60
FACE_W, FACE_H = 200, 200
FACE_X, FACE_Y = 75, 500
FACER_X, FACER_Y = 450, 500
FACEG_X, FACEG_Y = 825, 500
DASH_W, DASH_H = 480, 400
DASH_X, DASH_Y = 740, 36
INSET_DIM = 200

# video and face tracking settings
IDX_WEBCAM = 0
MIN_FACE_AREA = 10000

# set up gui
font1 = ImageFont.truetype(FONT, FONT_SIZE_1)
font2 = ImageFont.truetype(FONT, FONT_SIZE_2)
font3 = ImageFont.truetype(FONT, FONT_SIZE_3)
gui = Image.new('RGB', (GUI_W, GUI_H))
ctx = ImageDraw.Draw(gui)

# empty
im_face_empty = Image.new('RGB', (FACE_W, FACE_H))
ctx_empty = ImageDraw.Draw(im_face_empty)
ctx_empty.rectangle((0, 0, FACE_W-1, FACE_H-1), fill='#222', outline='#00f') 




##########
# import fnmatch
# import os
# import matplotlib.pyplot as plt
# from PIL import Image
# import numpy as np
# from tqdm import tqdm
# import random
# from sklearn.decomposition import PCA
# from random import shuffle
# import glob

# lfw_path = "/Users/gene/Downloads/lfw-deepfunneled"

# people = {}
# for root, dirnames, filenames in os.walk(lfw_path):
#     for filename in fnmatch.filter(filenames, '*.jpg'):
#         name = root.split('/')[-1]
#         if name in people:
#             people[name].append(os.path.join(root, filename))
#         else:
#             people[name] = [os.path.join(root, filename)]

# all_names = [name for name in people]
# nums = [len(people[name]) for name in people]
# idx = list(reversed(np.argsort(nums)))
# top_names = [all_names[i] for i in idx]
# person = people[top_names[0]]
# files = glob.glob('%s/*/*.jpg' % lfw_path)
# person = files
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
        idx_top = np.argsort(areas)[-1]
        if areas[idx_top] < MIN_FACE_AREA:
            return None

        return faces[idx_top]


class Worker:

    class Mode(Enum):
        COLLECTING = 1
        READY_TO_UPDATE = 2
        WAITING = 3

    def __init__(self, name, worker_ip, conductor_ip, conductor_port, batch_size, hook, faceCascade, verbose):
        self.name = name
        self.worker_ip = worker_ip
        self.conductor_ip = conductor_ip
        self.conductor_port = conductor_port
        self.syft_port = None
        self.num_peers = 0
        self.batch_size = batch_size
        self.verbose = verbose
        self.hook = hook
        self.registered = False
        self.active = False
        self.dataset = []
        self.current_face = None
        self.reconstructed_capture = None
        self.reconstruction_loss = 0
        self.kl_loss = 0
        self.random_image = None
        self.socket_uri = 'ws://%s:%d' % (conductor_ip, conductor_port)
        self.faceCascade = faceCascade
        self.waiting_for_conductor = False
        self.last_updated_time = None
        self.num_updates = 0

    def set_mode(self, mode):
        self.mode = mode
            
    def setup_data_source(self, data_input, image_size):
        self.image_size = image_size
        self.input = data_input
        if self.input == 'cam':
            self.camera = cv2.VideoCapture(0)

    def setup_lfw_loader(self, lfw_loader):
        self.lfw_loader = lfw_loader

    def ready_to_update(self):
        ready = (self.num_samples() == self.batch_size)
        return ready

    def get_next_batch(self):
        log('Get next batch...', self.verbose)
        data = np.array(self.dataset)
        data = torch.tensor(np.mean(data, -1))
        log('Found batch of size %s' % str(data.shape), self.verbose)
        self.x_ptr = data.tag('#x').send(self.local_worker)
        
    def activate(self):
        kwargs = { "hook": self.hook,
            "id": self.name,
            "host": "0.0.0.0",
            "port": self.syft_port }
        self.active = True
        self.local_worker = WebsocketServerWorker(**kwargs)
        self.local_worker.start()

    async def capture(self):
        if self.input == 'cam':
            log('Get camera input', self.verbose.capture)
            if not self.camera.isOpened():
                log('Unable to load camera.', self.verbose.capture)
                time.sleep(2)
                return
            _, image = self.camera.read()

        elif self.input == 'picam':
            log('Get picamera input', self.verbose.capture)
            stream = io.BytesIO()
            with picamera.PiCamera() as camera:
                camera.start_preview()
                #time.sleep(1)
                camera.capture(stream, format='jpeg')
            stream.seek(0)
            image = np.array(Image.open(stream))

        elif self.input == 'lfw':
            log('Get LFW input', self.verbose.capture)
            image = self.lfw_loader.get_random_image()
        
        self.current_capture = image

    async def process_capture(self, faceCascade):
        if self.waiting_for_conductor or self.num_samples() >= self.batch_size:
            self.current_face = None
            return False

        face = search_for_face(self.current_capture, self.faceCascade)
        if face is not None:
            ih, iw = self.current_capture.shape[0:2]
            x, y, w, h = face
            x1, x2 = int(x - 0.33 * w), int(x + 1.33 * w)
            y1, y2 = int(y - 0.33 * h), int(y + 1.33 * h)
            
            if x1 < 0 or x2 >= iw or y1 < 0 or y2 >= ih:
                log('Face not fully inside frame', self.verbose.capture)
                self.current_face = None
                return False 

            log('Found face inside [%d, %d, %d, %d]' % (y1, y2, x1, x2), self.verbose.capture)
            
            # crop out face image
            face_image = self.current_capture[y1:y2, x1:x2]
            face_image = cv2.resize(face_image, (self.image_size, self.image_size), interpolation=cv2.INTER_CUBIC)
            face_image = np.array(face_image).astype(np.float32) / 255.
            
            log('Image reshaped to %s' % str(face_image.shape), self.verbose.capture)
            self.current_face = np.expand_dims(face_image, 0) #face_image
            self.dataset.append(face_image)
            return True

        else:
            log('No face found', self.verbose.capture)
            self.current_face = None
            return False


    def draw(self):
        log('Draw loop', self.verbose.draw)

        if self.last_updated_time:
            last_updated_str = time.strftime("%H:%M:%S", time.gmtime(self.last_updated_time))
            last_updated_ago = time.time() - self.last_updated_time
        else:
            last_updated_str = 'N/A'
            last_updated_ago = 0

        model_type, num_params = 'fully-connected VAE', 48812 # 'convolutional'

        # draw images
        im_frame = Image.fromarray(self.current_capture)
        log('Resize frame_image from %dx%d to %dx%d' % (im_frame.width, im_frame.height, FRAME_W, FRAME_H), self.verbose.draw)
        im_frame = im_frame.resize((FRAME_W, FRAME_H), Image.BICUBIC)
        log('Drawing...', self.verbose.draw)
        ctx.rectangle((0, 0, GUI_W, GUI_H), fill='#000', outline='#000')
        gui.paste(im_frame, (FRAME_X, FRAME_Y))

        if self.current_face is None and not self.waiting_for_conductor:
            ctx.text((FRAME_X + 3, FRAME_Y + 2), "Place your face inside the red ellipse :)", font=font2, fill='#00f')
            ctx.ellipse((FRAME_X + 20, FRAME_Y + 25, FRAME_X + FRAME_W - 20, FRAME_Y + FRAME_H - 20), width=12, fill=None, outline='#0000ff44')
        elif not self.waiting_for_conductor:
            ctx.rectangle((FRAME_X, FRAME_Y, FRAME_X + FRAME_W, FRAME_Y + FRAME_H), width=8, fill=None, outline='#0f0')

        if self.current_face is not None:
            img_capture = Image.fromarray((255 * self.current_face[0]).astype(np.uint8))
            img_capture = img_capture.resize((INSET_DIM, INSET_DIM), Image.NEAREST)
            gui.paste(img_capture, (FACE_X, FACE_Y))
        if self.reconstructed_capture is not None:
            img_reconstructed = Image.fromarray(self.reconstructed_capture.astype(np.uint8))
            img_reconstructed = img_reconstructed.resize((INSET_DIM, INSET_DIM), Image.NEAREST)
            gui.paste(img_reconstructed, (FACER_X, FACER_Y))

        currTime = time.strftime('%l:%M%p')
        ctx.text((FRAME_X, FRAME_Y - FONT_SIZE_1 - 5), "Camera feed %s" % currTime, font=font1, fill=(0, 255, 0, 255))
        ctx.text((FACE_X, FACE_Y - FONT_SIZE_2 - 5), "detected", font=font2, fill=(0, 255, 0, 255))
        ctx.text((FACER_X, FACER_Y - FONT_SIZE_2 - 5), "reconstructed", font=font2, fill=(0, 255, 0, 255))
        ctx.text((FACEG_X, FACEG_Y - FONT_SIZE_2 - 5), "randomly generated", font=font2, fill=(0, 255, 0, 255))

        # draw dashboard
        log('Draw dashboard', self.verbose.draw)
        gui_color = '#0ff' if self.mode == Worker.Mode.WAITING else '#0f0'
        gui_color_rgba = (0, 255, 255, 255) if self.mode == Worker.Mode.WAITING else (0, 255, 0, 255)
        dashboard = Image.new('RGB', (DASH_W, DASH_H))
        ctx_dash = ImageDraw.Draw(dashboard)
        ctx_dash.rectangle((0, 0, DASH_W-1, DASH_H-1), fill=None, outline=gui_color) 
        ctx_dash.rectangle((0, 0, DASH_W-1, FONT_SIZE_2+8), fill=None, outline=gui_color) 
        ctx_dash.text((3, 3), "Dashboard", font=font2, fill=gui_color_rgba)
        time_running = str(round(time.time()-t0)) + ' sec'
        if self.mode == Worker.Mode.COLLECTING:
            current_status = 'Collecting samples'
        elif self.mode == Worker.Mode.READY_TO_UPDATE:
            current_status = 'Ready to update'
        elif self.mode == Worker.Mode.WAITING:
            current_status = 'Waiting for conductor'

        last_updated_ago_str = ('%d sec' % last_updated_ago) if last_updated_ago <= 60 else ('%d min' % int(last_updated_ago/60))
        for l, line in enumerate([
            'Current status:  %s' % current_status,
            '',
            'Time running:  %s' % time_running,
            'Number of peers:  %d' % self.num_peers,
            'My name:  %s' % (self.name),
            'My location:  %s:%d' % (self.worker_ip, self.syft_port),
            'Conductor location:  %s:%d' % (self.conductor_ip, self.conductor_port),
            '',
            'Model:  %s (%d parameters, batch size %d)' % (model_type, num_params, self.batch_size),
            'Number of local updates:  %d' % self.num_updates,
            'Last model update:  %s (%s ago)' % (last_updated_str, last_updated_ago_str),
            'Current reconstruction loss:  %0.2f' % self.reconstruction_loss,
            'Current KL-divergence loss:  %0.2f' % self.kl_loss,
            '',
            'Current batch num samples:  %d' % self.num_samples()
        ]):
            ctx_dash.text((6, 15 + FONT_SIZE_2 + l * FONT_SIZE_3 * 1.4), line, font=font3, fill=gui_color_rgba)
        gui.paste(dashboard, (DASH_X, DASH_Y))

        return gui

    def num_samples(self):
        return len(self.dataset)

    async def get_port_assignment(self):
        log('Get port assignment from conductor', self.verbose.event)
        async with websockets.connect(self.socket_uri) as websocket:
            message = json.dumps({'action': 'get_available_port'}) #, 'name': self.name, 'host': get_local_ip_address()})
            await websocket.send(message)
            result = json.loads(await websocket.recv())
            if result['success']:
                log('Got available syft port from conductor: %d' % result['syft_port'], self.verbose.event)
                self.syft_port = result['syft_port']

    async def try_register(self):
        log('Register with conductor', self.verbose.event)
        async with websockets.connect(self.socket_uri) as websocket:
            message = json.dumps({'action': 'register', 'name': self.name, 'syft_port': self.syft_port, 'host': get_local_ip_address()})
            await websocket.send(message)
            result = json.loads(await websocket.recv())
            if result['success']:
                log('Registration successful.', self.verbose.event)
                self.num_peers = result['num_peers']
                self.registered = True

    async def request_update(self):
        log('Request to make a model update', self.verbose.event)
        async with websockets.connect(self.socket_uri) as websocket:
            message = json.dumps({'action': 'request_model', 'name': self.name})
            self.waiting_for_conductor = True
            await websocket.send(message)
            self.waiting_for_conductor = False
            result = json.loads(await websocket.recv())
            if result['success']:
                log('Ready to make an update...', self.verbose.event)
    
    async def ping_conductor(self):
        async with websockets.connect(self.socket_uri) as websocket:
            message = json.dumps({'action': 'ping_conductor', 'name': self.name})
            await websocket.send(message)
            result = json.loads(await websocket.recv())
            return result



async def update_worker(loop: asyncio.AbstractEventLoop, worker, faceCascade, verbose) -> None:

    while worker.syft_port is None:
        await worker.get_port_assignment()
        time.sleep(1)

    while not worker.registered:
        await worker.try_register()
        time.sleep(1)

    while worker.active:
        log('worker mode %s' % str(worker.mode), verbose.event)
        await worker.capture()
        if worker.mode == Worker.Mode.COLLECTING:
            await worker.process_capture(faceCascade)
            if worker.ready_to_update():
                worker.set_mode(Worker.Mode.READY_TO_UPDATE)

        elif worker.mode == Worker.Mode.READY_TO_UPDATE:
            worker.get_next_batch() 
            await worker.request_update()
            worker.set_mode(Worker.Mode.WAITING)

        elif worker.mode == Worker.Mode.WAITING:
            result = await worker.ping_conductor()   
            if not result['in_queue']:
                if 'user' in result:
                    worker.reconstructed_capture = np.array(result['user']['reconstructed_image'])
                    worker.random_image = np.array(result['user']['random_image'])
                    worker.reconstruction_loss = result['user']['reconstruction_loss']
                    worker.kl_loss = result['user']['kl_loss']
                worker.num_peers = result['num_peers']
                worker.last_updated_time = time.time()
                worker.num_updates += 1
                worker.random_image = None
                worker.dataset = []
                worker.set_mode(Worker.Mode.COLLECTING)

        global gui
        gui = worker.draw()
        
        time.sleep(0.25)


def start_handler_thread(loop: asyncio.AbstractEventLoop, worker, faceCascade, verbose) -> None:
    asyncio.set_event_loop(loop)
    asyncio.run_coroutine_threadsafe(update_worker(loop, worker, faceCascade, verbose), loop)
    loop.run_forever()


def start_syft_worker_thread(loop: asyncio.AbstractEventLoop, worker) -> None:
    asyncio.set_event_loop(loop)
    #asyncio.run_coroutine_threadsafe(launch_worker(loop, worker, faceCascade), loop)
    worker.activate()
    loop.run_forever()


def main():
    args = parser.parse_args()

    if args.input == 'picam':
        import picamera
    
    verbose = CheckableQueue()
    verbose.event = args.verbose_event
    verbose.capture = args.verbose_capture
    verbose.draw = args.verbose_draw
    
    # setup worker
    hook = sy.TorchHook(torch)
    name = args.name if args.name is not None else get_random_name()
    worker_ip = get_local_ip_address() if args.worker_ip == 'auto' else args.worker_ip
    faceCascade = cv2.CascadeClassifier(args.cascadeFile)
    
    worker = Worker(name, worker_ip, args.conductor_ip, args.conductor_port, args.batch_size, hook, faceCascade, verbose)
    worker.setup_data_source(args.input, args.image_size) # * args.image_size)
    worker.set_mode(Worker.Mode.COLLECTING)
    if args.input == 'lfw':
        lfw_loader = LFWLoader("/Users/gene/Downloads/lfw-deepfunneled", True, verbose)
        worker.setup_lfw_loader(lfw_loader)
    
    # setup update thread
    handler_loop = asyncio.new_event_loop()
    update_thread = Thread(target=start_handler_thread, args=(handler_loop, worker, faceCascade, verbose,), daemon=True)
    update_thread.start()
    
    # launch syft worker in thread
    while worker.syft_port is None:
        log('No port assignment yet, communicating with conductor...', verbose.event)
        time.sleep(1)    
    syft_loop = asyncio.new_event_loop()
    worker_thread = Thread(target=start_syft_worker_thread, args=(syft_loop, worker,), daemon=True)
    worker_thread.start()
    
    # draw gui
    while True:
        if gui == None:
            continue
        if not args.hide_gui:
            cv2.imshow('Video', np.array(gui))
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        time.sleep(0.25)



if __name__== "__main__":
    main()