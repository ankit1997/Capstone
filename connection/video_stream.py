import os
import math
import time
import struct
import threading
import numpy as np
import socketserver
from pynput.keyboard import Key, Listener

carState = 0 # state of the car (stop, start) = (0, 1)
steeringVal = 90 # Steering angle
indicatorVal = 0 # Indicator value (left, none, right) = (-1, 0, 1)

stopped = True
firstTime = True
dataset = []

# def getSteeringAngle():
#     return steeringVal

# def getcarState():
#     return carState

def direction(angle):
    global steeringVal
    steeringVal = angle

def changeCarState(state):
    global carState
    carState = state

def indicatorCar(val):
    global indicatorVal
    indicatorVal = val

if 0:
    def save_data():
        # convert file paths to absolute paths
        ds = [[os.path.abspath(row[0]), row[1]] for row in dataset]

        fname = "dataset/logs.csv"
        if os.path.isfile(fname):
            previous_df = pd.read_csv(fname, header=None)
            print(previous_df.shape)
            previous_df = previous_df.append(ds, ignore_index=True)
            print(previous_df.shape)
            df = previous_df
            print("Adding to previous dataset...")
        else:
            df = pd.DataFrame(ds)
            print(df.shape)
            print("Creating new dataset file...")
        df.to_csv(fname, header=None, index=None)

        print("Data written to disk!")

class VideoStreamHandler(socketserver.StreamRequestHandler):
    def handle(self):
        print('Car connected!')

        # dataset = []
        global dataset

        stream_bytes = b''
        count = 0
        f = True
        size = 0

        # f = open('extra_data.csv', 'r')
        # values = [ float(line[:-1]) for line in f.readlines() ]
        # current = 0

        try:
            while True:
                if f:
                    s = self.rfile.read(4)
                    size = struct.unpack('>L', s)
                    size = size[0]
                    f = False

                stream_bytes += self.rfile.read(size)
                first = stream_bytes.find(b'\xff\xd8')
                last = stream_bytes.find(b'\xff\xd9')

                if first != -1 and last != -1:
                    f = True
                    jpg = stream_bytes[first:last+2]
                    stream_bytes = stream_bytes[last+2:]

                    fname = "dataset/IMG/{}.jpg".format(str(time.time()).replace('.', '_'))
                    with open(fname, 'wb') as f:
                        f.write(jpg)

                    # dataset.append([fname, getSteeringAngle()])
                    # a = int(steeringVal)

                    # dataset.append([fname, steeringVal])
                    data = struct.pack('>bbb', carState, steeringVal, indicatorVal)
                    count += 1
                    self.wfile.write(data)
                    self.wfile.flush()
        finally:
            print("Connection closed on thread 1")
            # save_data()
            self.rfile.close()

class Controller:
    def __init__(self):
        self.steer = 90
        self.indicator = 1
        self.STEP = 4
        self.STOP_VAL = 180
        self.MINVAL = 75
        self.MAXVAL = 105
        self.state = 0

    def start(self):

        def on_press(key):
            if key == Key.right:
                self.steer += self.STEP
                self.steer = min(self.MAXVAL, self.steer)
            elif key == Key.left:
                self.steer -= self.STEP
                self.steer = max(self.MINVAL, self.steer)
            elif key == Key.down:
                self.state = 0
            elif key == Key.up:
                self.state = 1
            elif key == Key.ctrl_l:
                self.indicator = -1 # left
            elif key == Key.alt_l:
                self.indicator = 1 # right
            elif key == Key.ctrl_r:
                self.indicator = 0

            if self.indicator and abs(self.steer-90) > 10:
                self.indicator = 0

            direction(self.steer)
            changeCarState(self.state)
            indicatorCar(self.indicator)
            print("> ", self.state, self.steer, self.indicator)

        def on_release(key):
            if key == Key.esc:
                # Stop listener
                return False

        print("Car started...")
        self.state = 1 # Start the car

        # Collect events until released
        with Listener(on_press=on_press, 
                    on_release=on_release) as listener:
            listener.join()

class ThreadServer(object):

    def server_thread(host, port):
        server = socketserver.TCPServer((host, port), VideoStreamHandler)
        return server.serve_forever

    video_thread = threading.Thread(target=server_thread('0.0.0.0', 8000))
    video_thread.start()

print("Starting server")
ThreadServer()

controller = Controller()
controller.start()