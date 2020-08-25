
from yaw_controller import YawController
from pid import PID
from lowpass import LowPassFilter
import rospy

GAS_DENSITY = 2.858
ONE_MPH = 0.44704
MIN_SPEED = 0.1

class Controller(object):
    def __init__(self, vehicle_mass, brake_deadband, decel_limit, accel_limit, wheel_radius,
                wheel_base, max_steer_angle, max_lat_accel, steer_ratio):
        # TODO: Implement
        # Steering
        self.yc = YawController(wheel_base, steer_ratio, MIN_SPEED, max_lat_accel, max_steer_angle)
        # Throttle
        kp = 0.3
        ki = 0.1
        kd = 0
        mn = 0 #min throttle
        mx = 0.4 #max throttle
        self.tc = PID(kp, ki, kd, mn, mx)
        # LPF
        tau = 0.5
        ts = 0.02
        self.lpf = LowPassFilter(tau, ts) 

        self.vehicle_mass = vehicle_mass
        self.brake_deadband = brake_deadband
        self.decel_limit = decel_limit
        self.accel_limit = accel_limit
        self.wheel_radius = wheel_radius

        self.last_time = rospy.get_time()

    def control(self, proposed_linear_velocity, proposed_angular_velocity,
                current_linear_velocity, current_angular_velocity, dbw_status):
        # TODO: Change the arg, kwarg list to suit your needs
        # Return throttle, brake, steer
        if not dbw_status:
            self.tc.reset()
            return 0., 0., 0.
        
        current_linear_velocity = self.lpf.filt(current_linear_velocity)

        steering = self.yc.get_steering(proposed_linear_velocity, proposed_angular_velocity, 
                                        current_linear_velocity)

        vel_error = proposed_linear_velocity - current_linear_velocity
        self.last_vel = current_linear_velocity

        current_time = rospy.get_time()
        sample_time = current_time - self.last_time
        self.last_time = current_time

        throttle = self.tc.step(vel_error, sample_time)
        brake = 0

        if proposed_linear_velocity == 0 and current_linear_velocity < 0.1:
            throttle = 0
            brake = 700
        elif throttle < 0.01 and vel_error < 0:
            throttle = 0
            decel = max(vel_error, self.decel_limit)
            brake = abs(decel)*self.vehicle_mass*self.wheel_radius #Torque in Nm

        return throttle, brake, steering

                            
