import numpy as np
import pybullet as pb

import subprocess as sp


UP = np.array((0, 0, 1))
HEIGHT = 256*4
WIDTH = 256*4
FFMPEG_BIN = "ffmpeg"


class Camera:
    def __init__(self,
                 env,
                 camera_offset=None,
                 height=HEIGHT,
                 width=WIDTH):

        self.env = env
        if camera_offset is None:
            camera_offset = [3, -3, 3]
        self.camera_offset = np.array(camera_offset)
        self.frame_step_rate = 5
        self.width = width
        self.height = height
        self.dirname = 'video'
        self.frame_num = 0

        command = [
            FFMPEG_BIN,
            '-y', '-f', 'rawvideo', '-vcodec', 'rawvideo',
            '-s', f'{height}x{width}',
            '-pix_fmt', 'rgb24', '-r', '24', '-i', '-',
            '-an', '-vcodec', 'mpeg4', self.file_path]

        self.pipe = sp.Popen(command, stdin=sp.PIPE)

    @property
    def file_path(self):
        return f'{self.dirname}/video.mp4'

    def take_image(self):
        self.frame_num += 1
        robot_loc = pb.getBasePositionAndOrientation(self.env.robot_id)[0]
        robot_loc = np.array(robot_loc)
        camera_pos = robot_loc + self.camera_offset
        cameras_vector = - self.camera_offset

        Camera_UP = UP - np.dot(UP, cameras_vector) * cameras_vector

        view_mat = pb.computeViewMatrix(
            camera_pos, robot_loc, Camera_UP)

        proj_mat = pb.computeProjectionMatrixFOV(
            fov=60,
            aspect=self.height/self.width,
            nearVal=0.01,
            farVal=100)

        img = pb.getCameraImage(
            height=HEIGHT, width=WIDTH,
            projectionMatrix=proj_mat,
            viewMatrix=view_mat)

        frame = np.array(img[2]).reshape((img[0], img[1], 4))
        try:
            self.pipe.stdin.write(frame[:, :, :3].tostring())
        except Exception as err:
            print(err)
