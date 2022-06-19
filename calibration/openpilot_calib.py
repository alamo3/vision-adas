import copy

import numpy as np

from common.transformations.camera import get_view_frame_from_road_frame
from common.transformations.orientation import euler_from_rot, rot_from_euler

MIN_SPEED_FILTER = 24  # Minimum speed require for calibration to accept data
MAX_VEL_ANGLE_STD = np.radians(0.25)  # Maximum angular velocity allowed below which calibration data is acceptable
MAX_YAW_RATE_FILTER = np.radians(2)  # Maximum Yaw rate allowed below which calibration data is acceptable

# This is at model frequency, blocks needed for efficiency
SMOOTH_CYCLES = 400
BLOCK_SIZE = 100
INPUTS_NEEDED = 5  # Minimum blocks needed for valid calibration
INPUTS_WANTED = 50  # We want a little bit more than we need for stability
MAX_ALLOWED_SPREAD = np.radians(2)
RPY_INIT = np.array([0.0, 0.0, 0.0])

# These values are needed to accomodate biggest modelframe
PITCH_LIMITS = np.array([-0.09074112085129739, 0.14907572052989657])  # -5 to 8.6 degrees approx
YAW_LIMITS = np.array([-0.06912048084718224, 0.06912048084718235])  # -4 to 4 degrees approx


class Calibration:
    UNCALIBRATED = 0
    CALIBRATED = 1
    INVALID = 2


def is_calibration_valid(rpy):
    return (PITCH_LIMITS[0] < rpy[1] < PITCH_LIMITS[1]) and (YAW_LIMITS[0] < rpy[2] < YAW_LIMITS[1])


def sanity_clip(rpy):
    if np.isnan(rpy).any():
        rpy = RPY_INIT
    return np.array([rpy[0],
                     np.clip(rpy[1], PITCH_LIMITS[0] - .005, PITCH_LIMITS[1] + .005),
                     np.clip(rpy[2], YAW_LIMITS[0] - .005, YAW_LIMITS[1] + .005)])


class Calibrator:
    def __init__(self, param_put=False):
        self.param_put = param_put

        rpy_init = RPY_INIT
        valid_blocks = 0

        self.reset(rpy_init, valid_blocks)
        self.update_status()

    def reset(self, rpy_init=RPY_INIT, valid_blocks=0, smooth_from=None):
        if not np.isfinite(rpy_init).all():
            self.rpy = copy.copy(RPY_INIT)
        else:
            self.rpy = rpy_init
        if not np.isfinite(valid_blocks) or valid_blocks < 0:
            self.valid_blocks = 0
        else:
            self.valid_blocks = valid_blocks
        self.rpys = np.tile(self.rpy, (INPUTS_WANTED, 1))

        self.idx = 0
        self.block_idx = 0
        self.v_ego = 0

        if smooth_from is None:
            self.old_rpy = RPY_INIT
            self.old_rpy_weight = 0.0
        else:
            self.old_rpy = smooth_from
            self.old_rpy_weight = 1.0

    def update_status(self):
        if self.valid_blocks > 0:
            max_rpy_calib = np.array(np.max(self.rpys[:self.valid_blocks], axis=0))
            min_rpy_calib = np.array(np.min(self.rpys[:self.valid_blocks], axis=0))
            self.calib_spread = np.abs(max_rpy_calib - min_rpy_calib)
        else:
            self.calib_spread = np.zeros(3)

        if self.valid_blocks < INPUTS_NEEDED:
            self.cal_status = Calibration.UNCALIBRATED
        elif is_calibration_valid(self.rpy):
            self.cal_status = Calibration.CALIBRATED
        else:
            self.cal_status = Calibration.INVALID

        # If spread is too high, assume mounting was changed and reset to last block.
        # Make the transition smooth. Abrupt transistion are not good foor feedback loop through supercombo model.
        if max(self.calib_spread) > MAX_ALLOWED_SPREAD and self.cal_status == Calibration.CALIBRATED:
            self.reset(self.rpys[self.block_idx - 1], valid_blocks=INPUTS_NEEDED, smooth_from=self.rpy)

        write_this_cycle = (self.idx == 0) and (self.block_idx % (INPUTS_WANTED // 5) == 5)
        if self.param_put and write_this_cycle:
            cal_params = {"calib_radians": list(self.rpy),
                          "valid_blocks": int(self.valid_blocks)}

    def update_car_speed(self, v_ego):
        self.v_ego = v_ego

    def get_smooth_rpy(self):
        if self.old_rpy_weight > 0:
            return self.old_rpy_weight * self.old_rpy + (1.0 - self.old_rpy_weight) * self.rpy
        else:
            return self.rpy

    def update_calibration_movement(self, trans, rot, trans_std, rot_std):
        self.old_rpy_weight = min(0.0, self.old_rpy_weight - 1 / SMOOTH_CYCLES)

        straight_and_fast = ((self.v_ego > MIN_SPEED_FILTER) and (trans[0] > MIN_SPEED_FILTER) and (
                    abs(rot[2]) < MAX_YAW_RATE_FILTER))
        certain_if_calib = ((np.arctan2(trans_std[1], trans[0]) < MAX_VEL_ANGLE_STD) or
                            (self.valid_blocks < INPUTS_NEEDED))

        if not (straight_and_fast and certain_if_calib):
            return None

        observed_rpy = np.array([0,
                                 -np.arctan2(trans[2], trans[0]),
                                 np.arctan2(trans[1], trans[0])])
        new_rpy = euler_from_rot(rot_from_euler(self.get_smooth_rpy()).dot(rot_from_euler(observed_rpy)))
        new_rpy = sanity_clip(new_rpy)

        self.rpys[self.block_idx] = (self.idx * self.rpys[self.block_idx] + (BLOCK_SIZE - self.idx) * new_rpy) / float(
            BLOCK_SIZE)
        self.idx = (self.idx + 1) % BLOCK_SIZE
        if self.idx == 0:
            self.block_idx += 1
            self.valid_blocks = max(self.block_idx, self.valid_blocks)
            self.block_idx = self.block_idx % INPUTS_WANTED
        if self.valid_blocks > 0:
            self.rpy = np.mean(self.rpys[:self.valid_blocks], axis=0)

        self.update_status()

        return new_rpy

    def get_calibration(self):
        smooth_rpy = self.get_smooth_rpy()
        extrinsic_matrix = get_view_frame_from_road_frame(0, smooth_rpy[1], smooth_rpy[2], 1.22)

        extrinsic_matrix_flattened = [float(x) for x in extrinsic_matrix.flatten()]
        cal_percentage = min(100 * (self.valid_blocks * BLOCK_SIZE + self.idx) // (INPUTS_NEEDED * BLOCK_SIZE), 100)
        cal_status = self.cal_status
        rpy_calib = [float(x) for x in smooth_rpy]

        return {'extrinsic': extrinsic_matrix_flattened, 'cal_percentage': cal_percentage, 'cal_status': cal_status,
                'rot_pitch_yaw': rpy_calib}
