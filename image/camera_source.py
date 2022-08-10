import time
from datetime import datetime

from image.image_source import ImageSource
import cv2


def get_cameras(max_id):
    non_working_ports = []
    test_port = 0
    working_ports = []

    for i in range(max_id):
        camera = cv2.VideoCapture(i)
        if not camera.isOpened():
            non_working_ports.append(i)
            print("Port {0} is not working".format(i))
        else:
            is_reading, img = camera.read()
            w = camera.get(3)
            h = camera.get(4)
            if is_reading:
                print("Camera at port {0} is available with capture res {1} x {2}".format(i, w, h))
                working_ports.append(i)
                camera.release()
        test_port += 1

    return working_ports, non_working_ports


class CameraSource(ImageSource):

    def __init__(self, cam_id, save_video=False, d_show=False, is_video=False,
                 flip_horizontal=False, flip_vertical=False):

        ImageSource.__init__(self, source_name='src_' + str(cam_id), source_id=cam_id)

        self.cam_id = cam_id

        if d_show:
            self.video_cap = cv2.VideoCapture(cam_id, cv2.CAP_DSHOW)
        else:
            self.video_cap = cv2.VideoCapture(cam_id)

        self.video_writer_old = None
        self.video_writer_new = None
        self.date_init = None
        self.save_video = save_video
        self.fps = self.video_cap.get(cv2.CAP_PROP_FPS)
        if self.fps == 0:
            self.fps = 20
        self.frames_written = 0

        self.flip_horizontal = flip_horizontal
        self.flip_vertical = flip_vertical
        self.is_video = is_video

        if save_video:
            self.new_video_writer()

    def new_video_writer(self):
        h, w, c = self.get_frame()[1].shape
        fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')

        self.date_init = date = datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")

        video_name = 'Videos/video_out_' + str(self.cam_id) + '_' + date + '.mp4'
        self.frames_written = 0
        self.video_writer_new = cv2.VideoWriter(video_name, fourcc, self.fps, (w, h))

    def set_parameter(self, param, value):
        self.check_video_feed()

        if not self.is_video:  # Do not need to set parameters on videos
            self.video_cap.set(param, value)

    def check_video_feed(self):
        if self.video_cap is None:
            raise Exception("No image feed found!")

    def get_frame(self):
        self.check_video_feed()
        ret, img = self.video_cap.read()

        if ret:
            if self.flip_vertical:
                img = cv2.flip(img, flipCode=0)
            if self.flip_horizontal:
                img = cv2.flip(img, flipCode=1)

        if self.save_video and ret and self.video_writer_new is not None:
            self.video_writer_new.write(img)
            self.frames_written = self.frames_written + 1

            if self.frames_written / self.fps > 60.0:
                self.save_video_minute()

        return ret, img

    def is_valid(self):
        ret, img = self.get_frame()
        return ret

    def flush_unsaved_video(self):
        self.video_writer_new.release()
        self.video_writer_new = None

    def save_video_minute(self):

        self.video_writer_new.release()

        self.video_writer_new = None

        self.new_video_writer()
