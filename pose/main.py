from math import pi, sin, cos

from direct.showbase.ShowBase import ShowBase
from direct.task import Task

import my_get_pose
import numpy as np

class ValueBuffer():
    def __init__(self, size, thresh=float("inf")):
        self.size = size
        self.values = []
        self.thresh = thresh
        self.prev_avg = 0
        self.prediction = None

    def add(self, value):
        self.prev_avg = self.avg()
        if (len(self.values) == self.size):
            # if abs(value - self.avg()) < self.thresh:
            self.values = self.values[1:] + [value]
        else:
            self.values += [value]
    
    def avg(self):
        return np.mean(self.values)

    def med(self):
        return np.median(self.values)

    def clear(self):
        self.values = []

    def predict(self):
        if np.isnan(self.avg()):
            return 0
        elif self.prediction is None:
            self.prediction = self.avg()*2 - self.prev_avg
        else:
            self.prediction += self.avg() - self.prev_avg

        return self.prediction

    def clear_predic(self):
        self.prediction = None


class MyApp(ShowBase):
    def __init__(self):
        ShowBase.__init__(self)

        self.cap, self.src, self.img_queue, self.box_queue, self.tm, self.mark_detector, self.pose_estimator = my_get_pose.init()
        self.x_pos_buff = ValueBuffer(10)
        self.y_pos_buff = ValueBuffer(10)
        self.z_pos_buff = ValueBuffer(10)
        self.x_ang_buff = ValueBuffer(10, 2)
        self.y_ang_buff = ValueBuffer(10)
        self.z_ang_buff = ValueBuffer(10)

        # Load the environment model.
        self.scene = self.loader.loadModel("models/environment")
        # Reparent the model to render.
        self.scene.reparentTo(self.render)
        # Apply scale and position transforms on the model.
        self.scene.setScale(0.25, 0.25, 0.25)
        self.scene.setPos(-8, 42, 0)

        # Add the spinCameraTask procedure to the task manager.
        self.taskMgr.add(self.spinCameraTask, "SpinCameraTask")

    # Define a procedure to move the camera.
    def spinCameraTask(self, task):
        pose = my_get_pose.get_pose(self.cap, self.src, self.img_queue, self.box_queue, self.tm, self.mark_detector, self.pose_estimator)

        # angleDegrees = task.time * 6.0
        # angleRadians = angleDegrees * (pi / 180.0)
        # self.camera.setPos(20 * sin(angleRadians), -20 * cos(angleRadians), 3)
        # self.camera.setHpr(angleDegrees, 0, 0)
        if pose != None:
            angle = pose[0] * -50
            print(angle[0])
            self.x_ang_buff.add(angle[0])
            x_ang = self.x_ang_buff.avg() 
            self.x_ang_buff.clear_predic()

            self.y_ang_buff.add(angle[1])
            y_ang = self.y_ang_buff.avg() 

            self.z_ang_buff.add(angle[0])
            z_ang = self.z_ang_buff.avg()

            pos = pose[1] / -10
            self.x_pos_buff.add(pos[0])
            x_pos = self.x_pos_buff.avg() 

            self.y_pos_buff.add(pos[0])
            y_pos = self.y_pos_buff.avg() 

            self.z_pos_buff.add(pos[0])
            z_pos = self.z_pos_buff.avg() 

            self.camera.setPos(0, 0, 0)
            self.camera.setHpr(x_ang, 0, 0)
        else:
            self.x_ang_buff.clear()
            self.x_pos_buff.clear()
            x_ang = self.x_ang_buff.predict()

            self.camera.setHpr(x_ang, 0, 0)
            print("LOST!!!")
        return Task.cont


app = MyApp()
app.run()