import numpy as np
from keras.models import model_from_json
from sklearn.preprocessing import StandardScaler
import pandas as pd


class DepthEstimator:
    """
    This keras model attempts to predict depth of objects
    based on bounding box coordinates of all detected objects
    in image. Has been trained on the KITTI dataset.
    """

    def __init__(self):
        # Load json model (pre-trained)
        json_file = open('../models_pre/model@1535477330.json')
        loaded_model_json = json_file.read()
        json_file.close()

        self.model = model_from_json(loaded_model_json)

        # Load model weights
        self.model.load_weights('../models_pre/model@1535477330.h5')

        self.model.compile(loss='mean_squared_error', optimizer='adam')

        print('Successfully loaded depth estimator model')

        # Load test data for scalers
        df_test = pd.read_csv('../models_pre/test.csv')
        X_test = df_test[['xmin', 'ymin', 'xmax', 'ymax']].values
        y_test = df_test[['zloc']].values

        self.input_scaler = StandardScaler()
        self.input_scaler.fit(X_test)

        self.output_scaler = StandardScaler()
        self.output_scaler.fit(y_test)

    def predict_depth(self, bboxes):
        """
        Predicts depth of objects in meters based on detected bounding boxes in the image
        :param bboxes: numpy array of bounding boxes
        :return: numpy array with predicted depths of bounding boxes matched by index.
        """
        X = bboxes
        X = self.input_scaler.transform(X)

        Y = self.model.predict(X)

        Y = self.output_scaler.inverse_transform(Y)

        return Y
