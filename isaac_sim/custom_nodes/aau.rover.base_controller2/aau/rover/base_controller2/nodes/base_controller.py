"""
This is the implementation of the OGN node defined in base_controller.ogn
"""

# Array or tuple values are accessed as numpy arrays so you probably need this import
import numpy
import math

"""
    Compute the steering angles and angular velocities for the wheels.

    lin_vel: linear velocity in meters per second
    ang_vel: angular velocity in radians per second
"""
def Ackerman_drive(lin_vel, ang_vel):
    # Define the static variables
    wheel_radius = 0.1
    wheel_diameter = 2 * math.pi * wheel_radius

    # The offset is defined as pairs of dX and dY
    wheel_offsets = [[-0.385, 0.438],   # FL
                    [0.385, 0.438],     # FR
                    [-0.447, 0.0],      # ML
                    [0.447, 0.0],       # MR
                    [-0.385, -0.411],   # RL
                    [0.385, -0.411]]    # RR

    steering_angles = [0] * len(wheel_offsets)
    angular_velocity = [0] * len(wheel_offsets)

    if ang_vel != 0:
        # Calculate the turning radius
        turn_radius = abs(lin_vel / ang_vel)

        sign = 1 
        if ang_vel < 0: 
            sign = -1

        index = 0
        for offset_pair in wheel_offsets:
            dX = turn_radius - offset_pair[0]
            steering_angles[index] = math.atan2(offset_pair[1], dX) * sign
            index = index + 1

        # Calculate the linar velocity of the wheels.

        # Calculate the amount of time to turn 45 degrees based on the anglar
        # velocity.
        t = math.pi / (2 * abs(ang_vel))

        direction = 1      # forward
        if lin_vel < 0:
            direction = -1 # backwards

        index = 0
        for offset_pair in wheel_offsets:
            radius = math.sqrt((turn_radius - offset_pair[0])**2 + (offset_pair[1])**2)
            wheel_vel = (math.pi * radius) / (2 * t)
            angular_velocity[index] = (wheel_vel / wheel_diameter) * direction
            index = index + 1
    else:
        # No angular velocity. Go stragit ahead.
        index = 0
        for offset_pair in wheel_offsets:
            angular_velocity[index] = lin_vel / wheel_diameter
            index = index + 1

    return steering_angles, angular_velocity

class base_controller:
    """
         Controller for the rover base
    """
    @staticmethod
    def compute(db) -> bool:
        """Compute the outputs from the current input"""
        
        try:
            # With the compute in a try block you can fail the compute by raising an exception
            #db.log_warn("Enter func")
            steering_angles, angular_velocity = Ackerman_drive(db.inputs.lin_vel, db.inputs.ang_vel)
            #db.log_warn(str(steering_angles))
            indexes = [0,1,4,5]
            steering_angles = list(map(lambda x: steering_angles[x], indexes))

            db.outputs.steer_command = numpy.array(steering_angles)
            db.outputs.velocity_command = numpy.array(angular_velocity)
        except Exception as error:
            # If anything causes your compute to fail report the error and return False
            db.log_error(str(error))
            return False

        # Even if inputs were edge cases like empty arrays, correct outputs mean success
        return True
