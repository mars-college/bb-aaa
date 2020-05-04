from threading import Thread
from socket import gethostbyname_ex, gethostname
from enum import Enum
import queue
from json import JSONEncoder
from sys import platform
import time
import fnmatch
import os
from random import shuffle
import glob
from datetime import datetime
import time
from PIL import Image
import numpy as np



def log(msg, verbose=True):
    currTime = datetime.utcnow().strftime('%H:%M:%S.%f')[:-3]
    if verbose:
        print('%s\t%s' % (currTime, msg))


def get_random_name():
    name = 'worker_%08d' % np.random.randint(1e8)
    return name


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



class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)



class CheckableQueue(queue.Queue): 
    def __contains__(self, item):
        with self.mutex:
            return item in self.queue
    def __str__(self):
        msg = ', '.join([item for item in self.queue])
        return msg



class LFWLoader:

    def __init__(self, lfw_path, use_top=False, verbose=False, to_shuffle=True):
        self.lfw_path = lfw_path
        self.verbose = verbose
        if use_top:
            files = self.get_top_person_images()            
        else:
            files = glob.glob('%s/*/*.jpg' % self.lfw_path)
        if to_shuffle:
            shuffle(files)
        self.images = files
    
    def get_top_person_images(self):
        people = {}
        for root, dirnames, filenames in os.walk(self.lfw_path):
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
        top_person = people[top_names[0]]
        return top_person

    def get_random_image(self):
        idx_random = np.random.randint(len(self.images))
        log('Get lfw input %s (%d/%d)' % (self.images[idx_random], idx_random, len(self.images)), self.verbose.capture)
        image = Image.open(self.images[idx_random]).convert('RGB')
        image = np.array(image)
        image = image[:, :, [2, 1, 0]]        
        return image
