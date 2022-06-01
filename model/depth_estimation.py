import numpy as np
from keras.models import model_from_json
from sklearn.preprocessing import StandardScaler
import pandas as pd


class DepthEstimator:

    def __init__(self):
        json_file = open('../models_pre/model@1535477330.json')
        loaded_model_json = json_file.read()
        json_file.close()

        self.model = model_from_json(loaded_model_json)

        self.model.load_weights('../models_pre/model@1535477330.h5')

        self.model.compile(loss='mean_squared_error', optimizer='adam')

        print('Successfully loaded depth estimator model')

        df_test = pd.read_csv('../models_pre/test.csv')
        X_test = df_test[['xmin', 'ymin', 'xmax', 'ymax']].values
        y_test = df_test[['zloc']].values

        self.input_scaler = StandardScaler()
        self.input_scaler.fit(X_test)

        self.output_scaler = StandardScaler()
        self.output_scaler.fit(y_test)

    def predict_depth(self, bboxes):

        X = bboxes
        X = self.input_scaler.transform(X)

        Y = self.model.predict(X)

        Y = self.output_scaler.inverse_transform(Y)

        return Y


if __name__ == "__main__":
    depth = DepthEstimator()

    depth_pred = depth.predict_depth([[676, 163, 688, 193]])
    print(depth_pred)
