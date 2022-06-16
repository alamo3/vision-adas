import json


class ModelOutputWriter:

    def __init__(self, lane_poly_deg=3, model_res=(640, 640), video_res=(1920, 1080)):
        self.lane_poly_degree = lane_poly_deg

        self.data_frame = {'model_res_x': model_res[0], 'model_res_y': model_res[1], 'video_res_x': video_res[0],
                           'video_res_y': video_res[1]}

    def append_data(self, num_lanes, coeffs_lanes, num_bboxes, bboxes, frame_num):

        frame_num_str = 'frame_num_' + str(frame_num)

        self.data_frame[frame_num_str] = {}
        self.data_frame[frame_num_str]['num_lanes'] = num_lanes
        self.data_frame[frame_num_str]['num_bboxes'] = num_bboxes

        for i in range(num_lanes):
            a, b, c, d = coeffs_lanes[i][0]
            self.data_frame[frame_num_str]['lane_' + str(i)] = {'a': str(a), 'b': str(b), 'c': str(c), 'd': str(d)}

        for i in range(num_bboxes):
            x1, y1, x2, y2, conf, label = bboxes[i]
            self.data_frame[frame_num_str]['bbox_' + str(i)] = {'x': str(x1), 'y': str(y1), 'x2': str(x2), 'y2': str(y2)}

    def save_data(self, file_name='output.json'):

        with open(file_name, 'w') as out_file:
            json.dump(self.data_frame, out_file, indent=4)
