from datetime import datetime

import cv2


class ImageSink:

    SINK_COUNT = 0

    def __init__(self, fps, sink_name='default_'+str(SINK_COUNT), interval_save_sec=60):

        self.sink_name = sink_name
        self.fps = fps
        self.width = 0
        self.height = 0

        self.video_writer = None
        self.frames_written = 0
        self.interval_save = interval_save_sec
        self.date_init = None

    def init_video_writer(self):

        fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')

        self.date_init = date = datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")

        video_name = 'Videos/video_out_' + self.sink_name + '_' + date + '.mp4'
        self.frames_written = 0
        self.video_writer = cv2.VideoWriter(video_name, fourcc, self.fps, (self.width, self.height))

    def sink_frame(self, frame):

        if self.video_writer is None:
            self.height, self.width, c = frame.shape
            self.init_video_writer()

        self.video_writer.write(frame)

        self.frames_written = self.frames_written + 1

        if self.frames_written / self.fps > self.interval_save:
            self.save_video_interval()

    def flush_unsaved_video(self):

        self.video_writer.release()
        self.video_writer = None

    def save_video_interval(self):
        self.video_writer.release()
        self.video_writer = None

        self.init_video_writer()

