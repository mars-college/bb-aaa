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
import time
import json
import numpy as np
import cv2
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import syft as sy
from syft.workers.websocket_server import WebsocketServerWorker

# USE_PICAMERA = False



gui = None





# gui settings
FONT = '/Users/gene/Downloads/of_v0.11.0_linuxarmv6l_release/examples/communication/networkUdpReceiverExample/bin/data/type/verdana.ttf'
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
#CASCADE_PATH = '/Users/gene/Downloads/Webcam-Face-Detect-master/haarcascade_frontalface_default.xml'
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

# video and face capturing
#faceCascade = cv2.CascadeClassifier(CASCADE_PATH)
#log.basicConfig(filename='webcam.log',level=log.INFO)
#video_capture = cv2.VideoCapture(IDX_WEBCAM)




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
        idx_top = np.argsort(areas)[-1]
        if areas[idx_top] < MIN_FACE_AREA:
            return None

        return faces[idx_top]



class Worker:

    class Mode(Enum):
        CAPTURING = 1
        READY_TO_UPDATE = 2
        WAITING = 3

    def __init__(self, name, worker_ip, conductor_ip, conductor_port, syft_port, batch_size, hook, faceCascade):
        self.name = name
        self.worker_ip = worker_ip
        self.conductor_ip = conductor_ip
        self.conductor_port = conductor_port
        self.syft_port = syft_port
        self.batch_size = batch_size
        self.hook = hook
        self.registered = False
        self.active = False
        self.captures = []
        self.current_capture = None
        self.reconstructed_capture = None
        self.random_image = None
        self.n_updates = 0
        self.socket_uri = 'ws://%s:%d' % (conductor_ip, conductor_port)
        self.faceCascade = faceCascade

    def set_mode(self, mode):
        self.mode = mode
            
    def setup_data_source(self, data_input, image_size):
        self.image_size = image_size
        self.input = data_input
        if self.input == 'cam':
            self.camera = cv2.VideoCapture(0)

    def ready_to_update(self):
        ready = (self.num_captures() == self.batch_size)
        return ready

    def get_next_batch(self):
        print('Get next batch...')
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
                #time.sleep(1)
                camera.capture(stream, format='jpeg')
            stream.seek(0)
            image = np.array(Image.open(stream))

        elif self.input == 'lfw':
            idx_random = int(len(person) * random.random())
            print('Get lfw input %s (%d/%d)' % (person[idx_random], idx_random, len(person)))
            image = Image.open(person[idx_random]).convert('RGB')
            image = np.array(image)

        self.image = image
        face = search_for_face(image, self.faceCascade)
        if face is not None:
            ih, iw = self.image.shape[0:2]
            x, y, w, h = face

            x1, x2 = int(x - 0.33 * w), int(x + 1.33 * w)
            y1, y2 = int(y - 0.33 * h), int(y + 1.33 * h)

            if x1 < 0 or x2 >= iw or y1 < 0 or y2 >= ih:
                print('Face not fully inside frame')
                self.current_capture = None
                return False 

            print('Found face inside', y1, y2, x1, x2)
        
            # crop out face image
            image = image[y1:y2, x1:x2]
            image = cv2.resize(image, (64, 64), interpolation=cv2.INTER_CUBIC)
            image = np.array(image).astype(np.float32) / 255.
            image = np.expand_dims(image, 0)

            self.current_capture = image
            self.captures.append(image)
            return True

        else:
            print('No face found')
            self.current_capture = None
            return False






######################




    def draw(self):
        print("d1")

        # dashboard data
        time_running = '16:44'
        num_peers = 4
        name = 'worker_01'
        ip_address = '108.205.10.82'
        conductor_ip, conductor_port = '108.205.10.99', 7182
        syft_port = 8182
        num_updates = 16
        last_updated_time, last_updated_ago = '2:45pm', 56
        reconst_loss, kl_loss = 291.2923401, 5.7129323
        model_type, num_params, batch_size = 'fully-connected VAE', 48812, 8 # 'convolutional'
        #num_samples = len(self.captures)
        current_status = 'Collecting samples' # 'Ready', 'Training

        print("d2")
        # # Capture frame-by-frame
        # ret, frame = video_capture.read()
        # currTime = time.strftime('%l:%M%p')

        # face = search_for_face(frame, self.faceCascade)
        # if face is not None:
        #     x, y, w, h = face
        #     frame_face = frame[y:y+h, x:x+w]
        #     im_face = Image.fromarray(frame_face)
        #     im_face = im_face.resize((FACE_W, FACE_H), Image.BICUBIC)
        # else:
        #     im_face = im_face_empty
        #     # insert 
            
        print('d2a')
        
    



    
        # draw images
        im_frame = Image.fromarray(self.image)
        print('d2b')
        im_frame = im_frame.resize((FRAME_W, FRAME_H), Image.BICUBIC)
        print('d2c')
        ctx.rectangle((0, 0, GUI_W, GUI_H), fill='#000', outline='#000')
        print('d2d')
        gui.paste(im_frame, (FRAME_X, FRAME_Y))
        print('d2e')
        if self.current_capture is None:
            ctx.text((FRAME_X + 3, FRAME_Y + 2), "Place your face inside the red ellipse :)", font=font2, fill='#00f')
            ctx.ellipse((FRAME_X + 20, FRAME_Y + 25, FRAME_X + FRAME_W - 20, FRAME_Y + FRAME_H - 20), width=12, fill=None, outline='#0000ff44')
        else:
            ctx.rectangle((FRAME_X, FRAME_Y, FRAME_X + FRAME_W, FRAME_Y + FRAME_H), width=8, fill=None, outline='#0f0')
        print('d2f', type(self.current_capture))
        #print(self.current_capture.dtype, self.current_capture.shape)
        print('uhug')
        #gui.paste(Image.fromarray(self.current_capture[0].astype(np.uint8)), (FACE_X, FACE_Y))
        #print('dfjhksdf')
        #print(gui)
        #print(gui.paste)
        if self.current_capture is not None:
            print('in here', FACE_X)
    #        gui.paste(self.current_capture, (FACE_X, FACE_Y))
            print(self.current_capture.shape)
            print(np.min(self.current_capture[0]), np.max(self.current_capture[0]))
            img_capture = Image.fromarray((255 * self.current_capture[0]).astype(np.uint8))
            print(img_capture)
            img_capture = img_capture.resize((INSET_DIM, INSET_DIM), Image.NEAREST)
            gui.paste(img_capture, (FACE_X, FACE_Y))
        if self.reconstructed_capture is not None:
            print('d2grecons')
            #img_reconstructed0 = self.reconstructed_capture.astype(np.uint8)
            #print(img_reconstructed0.shape)
            #print(img_reconstructed0)
            img_reconstructed = Image.fromarray(self.reconstructed_capture.astype(np.uint8))
            img_reconstructed = img_reconstructed.resize((INSET_DIM, INSET_DIM), Image.NEAREST)
            gui.paste(img_reconstructed, (FACER_X, FACER_Y))
        #gui.paste(im_face, (FACEG_X, FACEG_Y))
        
        print('d2h')
        # draw text for images
        currTime = time.strftime('%l:%M%p')
        ctx.text((FRAME_X, FRAME_Y - FONT_SIZE_1 - 5), "Camera feed %s" % currTime, font=font1, fill=(0, 255, 0, 255))
        ctx.text((FACE_X, FACE_Y - FONT_SIZE_2 - 5), "detected", font=font2, fill=(0, 255, 0, 255))
        ctx.text((FACER_X, FACER_Y - FONT_SIZE_2 - 5), "reconstructed", font=font2, fill=(0, 255, 0, 255))
        ctx.text((FACEG_X, FACEG_Y - FONT_SIZE_2 - 5), "randomly generated", font=font2, fill=(0, 255, 0, 255))

        print("d8")
        
        # draw dashboard
        dashboard = Image.new('RGB', (DASH_W, DASH_H))
        ctx_dash = ImageDraw.Draw(dashboard)
        ctx_dash.rectangle((0, 0, DASH_W-1, DASH_H-1), fill=None, outline='#0f0') 
        ctx_dash.rectangle((0, 0, DASH_W-1, FONT_SIZE_2+8), fill=None, outline='#0f0') 
        ctx_dash.text((3, 3), "Dashboard", font=font2, fill=(0, 255, 0, 255))
        for l, line in enumerate([
            'Time running:  %s' % time_running,
            'Number of peers:  %d' % num_peers,
            'My name:  %s' % (name),
            'My location:  %s:%d' % (ip_address, syft_port),
            'Conductor location:  %s:%d' % (conductor_ip, conductor_port),
            '',
            'Model:  %s (%d parameters, batch size %d)' % (model_type, num_params, batch_size),
            'Number of local updates:  %d' % num_updates,
            'Last model update:  %s (%d min ago)' % (last_updated_time, last_updated_ago),
            'Current reconstruction loss:  %0.2f' % reconst_loss,
            'Current KL-divergence loss:  %0.2f' % kl_loss,
            '',
            'Current number of samples:  %d' % self.num_captures(),
            'Current status:  %s %d' % (current_status, np.random.randint(1000))
        ]):
            ctx_dash.text((6, 15 + FONT_SIZE_2 + l * FONT_SIZE_3 * 1.4), line, font=font3, fill=(0, 255, 0, 255))
        gui.paste(dashboard, (DASH_X, DASH_Y))

        print("d9")
        
        # Display the resulting frame
        #cv2.imshow('Video', np.array(gui))
        return gui
        #print("d10")
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break






##############





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
    
    async def ping_conductor(self):
        async with websockets.connect(self.socket_uri) as websocket:
            message = json.dumps({'action': 'ping_conductor', 'name': self.name})
            await websocket.send(message)
            result = json.loads(await websocket.recv())
            return result



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
            result = await worker.ping_conductor()            
            if not result['in_queue']:
                if 'user' in result:
                    worker.reconstructed_capture = np.array(result['user']['reconstructed_image'])
                    worker.random_image = np.array(result['user']['random_image'])
                worker.n_updates += 1
                worker.random_image = None
                worker.captures = []
                worker.set_mode(Worker.Mode.CAPTURING)
        
        global gui
        gui = worker.draw()
        
        #time.sleep(0.25)


def start_background_thread(loop: asyncio.AbstractEventLoop, worker, faceCascade) -> None:
    asyncio.set_event_loop(loop)
    asyncio.run_coroutine_threadsafe(update_worker(loop, worker, faceCascade), loop)
    loop.run_forever()


def start_worker_thread(loop: asyncio.AbstractEventLoop, worker) -> None:
    asyncio.set_event_loop(loop)
    #asyncio.run_coroutine_threadsafe(launch_worker(loop, worker, faceCascade), loop)
    worker.activate()
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
    worker = Worker(args.name, worker_ip, args.conductor_ip, args.conductor_port, args.syft_port, args.batch_size, hook, faceCascade)
    worker.setup_data_source(args.input, 64 * 64)
    worker.set_mode(Worker.Mode.CAPTURING)

    # setup update thread
    loop = asyncio.new_event_loop()
    update_thread = Thread(target=start_background_thread, args=(loop, worker, faceCascade,), daemon=True)
    update_thread.start()

    # launch syft worker in thread
    loop2 = asyncio.new_event_loop()
    worker_thread = Thread(target=start_worker_thread, args=(loop2, worker,), daemon=True)
    worker_thread.start()

    # draw gui
    while True:
        if gui == None:
            continue
        cv2.imshow('Video', np.array(gui))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        #time.sleep(1)



if __name__== "__main__":
    main()