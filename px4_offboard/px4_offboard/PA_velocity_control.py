#!/usr/bin/env python
############################################################################
#
#   Copyright (C) 2022 PX4 Development Team. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
# 1. Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in
#    the documentation and/or other materials provided with the
#    distribution.
# 3. Neither the name PX4 nor the names of its contributors may be
#    used to endorse or promote products derived from this software
#    without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
# OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED
# AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
############################################################################

__author__ = "Braden Wagstaff"
__contact__ = "braden@arkelectron.com"

import rclpy
from rclpy.node import Node
import numpy as np
from rclpy.clock import Clock
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy

from px4_msgs.msg import OffboardControlMode
from px4_msgs.msg import TrajectorySetpoint
from px4_msgs.msg import VehicleStatus
from px4_msgs.msg import VehicleAttitude
from px4_msgs.msg import VehicleCommand
from px4_msgs.msg import VehicleOdometry
from geometry_msgs.msg import Twist, Vector3
from math import pi
from std_msgs.msg import Bool
from sensor_msgs.msg import LaserScan   

import torch
import torch.nn as nn
import torch.optim as optim
import os

class StudentPolicy(nn.Module):
    def __init__(self, input_size=31, hidden_size=64, output_size=4):
        super(StudentPolicy, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.dropout1 = nn.Dropout(0.05)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.dropout2 = nn.Dropout(0.05)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.dropout3 = nn.Dropout(0.05)
        self.out = nn.Linear(hidden_size, output_size)
        self.elu = nn.ELU()

    def forward(self, x):
        x1 = self.elu(self.fc1(x))
        x1 = self.dropout1(x1)
        
        x2 = self.elu(self.fc2(x1))
        x2 = self.dropout2(x2)
        
        x3 = self.elu(self.fc3(x2))
        x3 = self.dropout3(x3)
        
        out = self.out(x3)
        
        return out  # self.fc3(x)
    
    def loss(self):
        return nn.MSELoss()
    
    def loss(self):
        return nn.MSELoss()
    
class PA_inference:
    def __init__(self):
        # Load the checkpoint
        checkpoint_dir = os.path.join("../LearningPerceptionAwareness_DynamicObs/checkpoints")
        checkpoint_path = os.path.join(checkpoint_dir, "best_model.pt")
        # print("LOADDDDDDDDDDDDDDDDDDDD",checkpoint_path)
        checkpoint = torch.load(checkpoint_path)
        
        # Load the model state
        self.student_model = StudentPolicy()
        self.student_model = self.student_model.to('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.student_model.load_state_dict(checkpoint['model_state_dict'])
        self.student_model.eval()


    def predict(self, input_data):
        with torch.no_grad():
            input_tensor = torch.tensor(input_data, dtype=torch.float32).to('cuda' if torch.cuda.is_available() else 'cpu')
            output = self.student_model(input_tensor)
        # return output.cpu().detach().numpy()
        return output

class OffboardControl(Node):

    def __init__(self):
        super().__init__('PA_velocity_control')
        qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1
        )

        #Create subscriptions
        self.status_sub = self.create_subscription(
            VehicleStatus,
            '/fmu/out/vehicle_status',
            self.vehicle_status_callback,
            qos_profile)
        
        self.offboard_velocity_sub = self.create_subscription(
            Twist,
            '/offboard_velocity_cmd',
            self.offboard_velocity_callback,
            qos_profile)
        
        self.attitude_sub = self.create_subscription(
            VehicleAttitude,
            '/fmu/out/vehicle_attitude',
            self.attitude_callback,
            qos_profile)
        
        self.my_bool_sub = self.create_subscription(
            Bool,
            '/arm_message',
            self.arm_message_callback,
            qos_profile)
        
        self.odom_sub = self.create_subscription(
            VehicleOdometry,
            '/fmu/out/vehicle_odometry',
            self.odom_callback,
            qos_profile)
        
        # self.desired_pos_sub = self.create_subscription(
        #     Vector3,
        #     '/desired_position',
        #     self.desired_position_callback,
        #     qos_profile)
        
        # self.lidar_sub = self.create_subscription(
        #     LaserScan,
        #     '/lidar_scan',
        #     self.lidar_callback,
        #     qos_profile)


        #Create publishers
        self.publisher_offboard_mode = self.create_publisher(OffboardControlMode, '/fmu/in/offboard_control_mode', qos_profile)
        self.publisher_velocity = self.create_publisher(Twist, '/fmu/in/setpoint_velocity/cmd_vel_unstamped', qos_profile)
        self.publisher_trajectory = self.create_publisher(TrajectorySetpoint, '/fmu/in/trajectory_setpoint', qos_profile)
        self.vehicle_command_publisher_ = self.create_publisher(VehicleCommand, "/fmu/in/vehicle_command", 10)

        
        #creates callback function for the arm timer
        # period is arbitrary, just should be more than 2Hz
        arm_timer_period = .1 # seconds
        self.arm_timer_ = self.create_timer(arm_timer_period, self.arm_timer_callback)

        # creates callback function for the command loop
        # period is arbitrary, just should be more than 2Hz. Because live controls rely on this, a higher frequency is recommended
        # commands in cmdloop_callback won't be executed if the vehicle is not in offboard mode
        timer_period = 0.02  # seconds
        # self.timer = self.create_timer(timer_period, self.cmdloop_callback)

        self.nav_state = VehicleStatus.NAVIGATION_STATE_MAX
        self.arm_state = VehicleStatus.ARMING_STATE_ARMED
        self.velocity = Vector3()
        self.yaw = 0.0  #yaw value we send as command
        self.trueYaw = 0.0  #current yaw value of drone
        self.offboardMode = False
        self.flightCheck = False
        self.myCnt = 0
        self.arm_message = False
        self.failsafe = False
        self.current_state = "IDLE"
        self.last_state = self.current_state
        
        self.PA_inference = PA_inference()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.robot_state = torch.zeros((1, 31), dtype=torch.float32, device=self.device)
        self.Yaw = torch.zeros((1,), dtype=torch.float32, device=self.device)

        controled_loop_period = 0.05  # seconds
        self.PA_timer = self.create_timer(controled_loop_period, self.PA_controller_callback)
        self.velocity_cmd = Vector3()


    def arm_message_callback(self, msg):
        self.arm_message = msg.data
        self.get_logger().info(f"Arm Message: {self.arm_message}")

    #callback function that arms, takes off, and switches to offboard mode
    #implements a finite state machine
    def arm_timer_callback(self):

        match self.current_state:
            case "IDLE":
                if(self.flightCheck and self.arm_message == True):
                    self.current_state = "ARMING"
                    self.get_logger().info(f"Arming")

            case "ARMING":
                if(not(self.flightCheck)):
                    self.current_state = "IDLE"
                    self.get_logger().info(f"Arming, Flight Check Failed")
                elif(self.arm_state == VehicleStatus.ARMING_STATE_ARMED and self.myCnt > 10):
                    self.current_state = "TAKEOFF"
                    self.get_logger().info(f"Arming, Takeoff")
                self.arm() #send arm command

            case "TAKEOFF":
                if(not(self.flightCheck)):
                    self.current_state = "IDLE"
                    self.get_logger().info(f"Takeoff, Flight Check Failed")
                elif(self.nav_state == VehicleStatus.NAVIGATION_STATE_AUTO_TAKEOFF):
                    self.current_state = "LOITER"
                    self.get_logger().info(f"Takeoff, Loiter")
                self.arm() #send arm command
                self.take_off() #send takeoff command

            # waits in this state while taking off, and the 
            # moment VehicleStatus switches to Loiter state it will switch to offboard
            case "LOITER": 
                if(not(self.flightCheck)):
                    self.current_state = "IDLE"
                    self.get_logger().info(f"Loiter, Flight Check Failed")
                elif(self.nav_state == VehicleStatus.NAVIGATION_STATE_AUTO_LOITER):
                    self.current_state = "OFFBOARD"
                    self.get_logger().info(f"Loiter, Offboard")
                self.arm()

            case "OFFBOARD":
                if(not(self.flightCheck) or self.arm_state != VehicleStatus.ARMING_STATE_ARMED or self.failsafe == True):
                    self.current_state = "IDLE"
                    self.get_logger().info(f"Offboard, Flight Check Failed")
                self.state_offboard()

        if(self.arm_state != VehicleStatus.ARMING_STATE_ARMED):
            self.arm_message = False

        if (self.last_state != self.current_state):
            self.last_state = self.current_state
            self.get_logger().info(self.current_state)

        self.myCnt += 1

    def state_offboard(self):
        self.myCnt = 0
        self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_DO_SET_MODE, 1., 6.)
        self.offboardMode = True   

    # Arms the vehicle
    def arm(self):
        self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, 1.0)
        self.get_logger().info("Arm command send")

    # Takes off the vehicle to a user specified altitude (meters)
    def take_off(self):
        self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_NAV_TAKEOFF, param1 = 1.0, param7=2.0) # param7 is altitude in meters
        self.get_logger().info("Takeoff command send")

    #publishes command to /fmu/in/vehicle_command
    def publish_vehicle_command(self, command, param1=0.0, param2=0.0, param7=0.0):
        msg = VehicleCommand()
        msg.param1 = param1
        msg.param2 = param2
        msg.param7 = param7    # altitude value in takeoff command
        msg.command = command  # command ID
        msg.target_system = 1  # system which should execute the command
        msg.target_component = 1  # component which should execute the command, 0 for all components
        msg.source_system = 1  # system sending the command
        msg.source_component = 1  # component sending the command
        msg.from_external = True
        msg.timestamp = int(Clock().now().nanoseconds / 1000) # time in microseconds
        self.vehicle_command_publisher_.publish(msg)

    #receives and sets vehicle status values 
    def vehicle_status_callback(self, msg):

        if (msg.nav_state != self.nav_state):
            self.get_logger().info(f"NAV_STATUS: {msg.nav_state}")
        
        if (msg.arming_state != self.arm_state):
            self.get_logger().info(f"ARM STATUS: {msg.arming_state}")

        if (msg.failsafe != self.failsafe):
            self.get_logger().info(f"FAILSAFE: {msg.failsafe}")
        
        if (msg.pre_flight_checks_pass != self.flightCheck):
            self.get_logger().info(f"FlightCheck: {msg.pre_flight_checks_pass}")

        self.nav_state = msg.nav_state
        self.arm_state = msg.arming_state
        self.failsafe = msg.failsafe
        self.flightCheck = msg.pre_flight_checks_pass


    #receives Twist commands from Teleop and converts NED -> FLU
    def offboard_velocity_callback(self, msg):
        #implements NED -> FLU Transformation
        # X (FLU) is -Y (NED)
        self.velocity.x = -msg.linear.y
        # Y (FLU) is X (NED)
        self.velocity.y = msg.linear.x
        # Z (FLU) is -Z (NED)
        self.velocity.z = -msg.linear.z
        # A conversion for angular z is done in the attitude_callback function(it's the '-' in front of self.trueYaw)
        self.yaw = msg.angular.z

    #receives current trajectory values from drone and grabs the yaw value of the orientation
    def attitude_callback(self, msg):
        orientation_q = msg.q

        #trueYaw is the drones current yaw value
        self.trueYaw = -(np.arctan2(2.0*(orientation_q[3]*orientation_q[0] + orientation_q[1]*orientation_q[2]), 
                                  1.0 - 2.0*(orientation_q[0]*orientation_q[0] + orientation_q[1]*orientation_q[1])))
    

    def odom_callback(self, msg):
        lin = torch.as_tensor(msg.velocity, dtype=torch.float32, device=self.device )
        ang = torch.as_tensor(msg.angular_velocity, dtype=torch.float32, device=self.device )
        pos = torch.as_tensor(msg.position, dtype=torch.float32, device=self.device )
        quat = torch.as_tensor(msg.q, dtype=torch.float32, device=self.device )

        self.Yaw = -(torch.arctan2(2.0*(quat[3]*quat[0] + quat[1]*quat[2]), 
                                  1.0 - 2.0*(quat[0]*quat[0] + quat[1]*quat[1])))
        
        local_lin = torch.zeros_like(lin)
        local_lin[0] = (lin[0] * torch.cos(self.Yaw) + lin[1] * torch.sin(self.Yaw)) * -1.0
        local_lin[1] = -lin[0] * torch.sin(self.Yaw) + lin[1] * torch.cos(self.Yaw)
        local_lin[2] = -lin[2]

        
        # print("Position:", pos)

        # print("local_lin:", local_lin)


        self.robot_state[0, 0:3] = local_lin
        self.robot_state[0, 3:6] = ang
        self.robot_state[0, 6:9] = pos
        self.robot_state[0, 9:13] = quat 

        # self.prepare_PA_input()
    
    def lidar_callback(self, msg):
        # Process LIDAR data here
        lidar_ranges = torch.tensor(msg.ranges, dtype=torch.float32, device=self.device).unsqueeze(0)
    
    
    def quat_rotate_inverse(self, q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        Rotate vector v by the inverse of quaternion q.
        q: (..., 4)  (w, x, y, z)
        v: (..., 3)
        """
        q = q / q.norm(dim=-1, keepdim=True)

        w = q[..., 0:1]
        xyz = q[..., 1:]

        t = 2.0 * torch.cross(xyz, v, dim=-1)
        return v - w * t + torch.cross(xyz, t, dim=-1)
    
    def quat_conjugate(self, q: torch.Tensor) -> torch.Tensor:
        """Computes the conjugate of a quaternion.

        Args:
            q: The quaternion orientation in (w, x, y, z). Shape is (..., 4).

        Returns:
            The conjugate quaternion in (w, x, y, z). Shape is (..., 4).
        """
        shape = q.shape
        q = q.reshape(-1, 4)
        return torch.cat((q[..., 0:1], -q[..., 1:]), dim=-1).view(shape)
    def quat_inv(self, q: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
        """Computes the inverse of a quaternion.

        Args:
            q: The quaternion orientation in (w, x, y, z). Shape is (N, 4).
            eps: A small value to avoid division by zero. Defaults to 1e-9.

        Returns:
            The inverse quaternion in (w, x, y, z). Shape is (N, 4).
        """
        return self.quat_conjugate(q) / q.pow(2).sum(dim=-1, keepdim=True).clamp(min=eps)
    
    def quat_mul(self, q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
        """Multiply two quaternions together.

        Args:
            q1: The first quaternion in (w, x, y, z). Shape is (..., 4).
            q2: The second quaternion in (w, x, y, z). Shape is (..., 4).

        Returns:
            The product of the two quaternions in (w, x, y, z). Shape is (..., 4).

        Raises:
            ValueError: Input shapes of ``q1`` and ``q2`` are not matching.
        """
        # check input is correct
        if q1.shape != q2.shape:
            msg = f"Expected input quaternion shape mismatch: {q1.shape} != {q2.shape}."
            raise ValueError(msg)
        # reshape to (N, 4) for multiplication
        shape = q1.shape
        q1 = q1.reshape(-1, 4)
        q2 = q2.reshape(-1, 4)
        # extract components from quaternions
        w1, x1, y1, z1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
        w2, x2, y2, z2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]
        # perform multiplication
        ww = (z1 + x1) * (x2 + y2)
        yy = (w1 - y1) * (w2 + z2)
        zz = (w1 + y1) * (w2 - z2)
        xx = ww + yy + zz
        qq = 0.5 * (xx + (z1 - x1) * (x2 - y2))
        w = qq - ww + (z1 - y1) * (y2 - z2)
        x = qq - xx + (x1 + w1) * (x2 + w2)
        y = qq - yy + (w1 - x1) * (y2 + z2)
        z = qq - zz + (z1 + y1) * (w2 - x2)

        return torch.stack([w, x, y, z], dim=-1).view(shape)
    
    def quat_apply(self, quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
        """Apply a quaternion rotation to a vector.

        Args:
            quat: The quaternion in (w, x, y, z). Shape is (..., 4).
            vec: The vector in (x, y, z). Shape is (..., 3).

        Returns:
            The rotated vector in (x, y, z). Shape is (..., 3).
        """
        # store shape
        shape = vec.shape
        # reshape to (N, 3) for multiplication
        quat = quat.reshape(-1, 4)
        vec = vec.reshape(-1, 3)
        # extract components from quaternions
        xyz = quat[:, 1:]
        t = xyz.cross(vec, dim=-1) * 2
        return (vec + quat[:, 0:1] * t + xyz.cross(t, dim=-1)).view(shape)

    def subtract_frame_transforms(
        self, t01: torch.Tensor, q01: torch.Tensor, t02: torch.Tensor | None = None, q02: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        r"""Subtract transformations between two reference frames into a stationary frame.

        It performs the following transformation operation: :math:`T_{12} = T_{01}^{-1} \times T_{02}`,
        where :math:`T_{AB}` is the homogeneous transformation matrix from frame A to B.

        Args:
            t01: Position of frame 1 w.r.t. frame 0. Shape is (N, 3).
            q01: Quaternion orientation of frame 1 w.r.t. frame 0 in (w, x, y, z). Shape is (N, 4).
            t02: Position of frame 2 w.r.t. frame 0. Shape is (N, 3).
                Defaults to None, in which case the position is assumed to be zero.
            q02: Quaternion orientation of frame 2 w.r.t. frame 0 in (w, x, y, z). Shape is (N, 4).
                Defaults to None, in which case the orientation is assumed to be identity.

        Returns:
            A tuple containing the position and orientation of frame 2 w.r.t. frame 1.
            Shape of the tensors are (N, 3) and (N, 4) respectively.
        """
        # compute orientation
        q10 = self.quat_inv(q01)
        if q02 is not None:
            q12 = self.quat_mul(q10, q02)
        else:
            q12 = q10
        # compute translation
        if t02 is not None:
            t12 = self.quat_apply(q10, t02 - t01)
        else:
            t12 = self.quat_apply(q10, -t01)
        return t12, q12
    
    def prepare_PA_input(self):
        lin_vel = self.robot_state[:, 0:3]
        ang_vel = self.robot_state[:, 3:6]
        position = self.robot_state[:, 6:9]
        quaternion = self.robot_state[:, 9:13]
        desired_pos_b = torch.ones((1, 3), dtype=torch.float32, device=self.device) 
        desired_pos_b[0, 0] = 3.0
        desired_pos_b[0, 1] = 0.0
        desired_pos_b[0, 2] = 3.0
    
        dist_2d = torch.zeros((1, 1), dtype=torch.float32, device=self.device)
        dist_z = torch.zeros((1, 1), dtype=torch.float32, device=self.device)

        local_position = torch.zeros_like(position)
        local_position[0, 0] = position[0, 0] * torch.cos(self.Yaw) + position[0, 1] * torch.sin(self.Yaw)
        local_position[0, 1] = -position[0, 0] * torch.sin(self.Yaw) + position[0, 1] * torch.cos(self.Yaw)
        local_position[0, 2] = -position[0, 2]

        # delta_w = desired_pos_b - local_position
        # delta_b = self.quat_rotate_inverse(quaternion, delta_w)
        # unit_dir = delta_b / (delta_b.norm(dim=-1, keepdim=True) + 1e-6)
        edit_position = position.clone()
        edit_position[0, 0] = position[0, 1]
        edit_position[0, 1] = position[0, 0]
        edit_position[0, 2] = -position[0, 2] - 2.0  

        edit_quaternion = quaternion.clone()
        edit_quaternion[0, 0] = quaternion[0, 3]
        edit_quaternion[0, 1] = quaternion[0, 0]
        edit_quaternion[0, 2] = quaternion[0, 1]
        edit_quaternion[0, 3] = quaternion[0, 2]

        edit_ang_vel = ang_vel.clone()
        edit_ang_vel[0, 0] = -ang_vel[0, 1]
        edit_ang_vel[0, 1] = -ang_vel[0, 0]
        edit_ang_vel[0, 2] = ang_vel[0, 2]

        # print("ang_vel:", ang_vel)

        delta_w, _ = self.subtract_frame_transforms(
            edit_position,
            edit_quaternion,
            desired_pos_b,
            None
        )

        unit_dir = delta_w / (delta_w.norm(dim=-1, keepdim=True) + 1e-6)
        # desired_dist = desired_pos_b - local_position
        # unit_desired_dist = desired_dist / (torch.norm(desired_dist) + 1e-6)

        

        # print("Desired Pos B:", desired_dist)
        dist_2d[0, 0] = delta_w[0, :1].norm()
        dist_z[0, 0] = delta_w[0, 2]

        mock_lidar_data = torch.full(
            (1, 20),
            10.0,
            dtype=torch.float32,
            device=self.device
        )
        # print("lin_vel:", lin_vel)
        # print("ang_vel:", ang_vel)
        # print("edit_ang_vel:", edit_ang_vel)
        # print("unit_dir:", unit_dir)
        # print("dist_2d:", dist_2d)
        # print("dist_z:", dist_z)

        input_data = torch.cat([
            lin_vel,
            edit_ang_vel,
            unit_dir,
            dist_2d,
            dist_z,
            mock_lidar_data
        ], dim=1)

        # print("Input Data:", input_data, input_data.shape)

        return input_data
    
    def PA_controller_callback(self):

        if(self.offboardMode == True):
            # Publish offboard control modes
            input_data = self.prepare_PA_input()
            # print("Input Data:", input_data.device, input_data.shape)
            output = self.PA_inference.predict(input_data)
            model_output = output.clone().clamp(-1.0, 1.0)  # Ensure outputs are in the range [-1, 1]

            # model_output[0, 0] = 0.5
            # model_output[0, 1] = 0.0
            # model_output[0, 2] = 0.5
            # model_output[0, 3] = 0.2

            
            output[0, 0] = model_output[0, 0] * 6.28  # yaw rate
            self.velocity_cmd.x = float(model_output[0, 2] * -3.0)   # velocity x
            self.velocity_cmd.y = float(model_output[0, 1] * 3.0)   # velocity y
            self.velocity_cmd.z = float(model_output[0, 3] * -3.0)  # velocity z (negative because of NED to FLU transformation in odom callback)
            # fake output for testing
            


            # global_vel = torch.zeros((1, 3), dtype=torch.float32, device=self.device)
            global_vel = torch.zeros((1, 3), dtype=torch.float32, device=self.device)
            global_vel[0, 0] = self.velocity_cmd.x * torch.cos(self.Yaw) - self.velocity_cmd.y * torch.sin(self.Yaw)
            global_vel[0, 1] = self.velocity_cmd.x * torch.sin(self.Yaw) + self.velocity_cmd.y * torch.cos(self.Yaw)
            global_vel[0, 2] = self.velocity_cmd.z

                #         # Compute velocity in the world frame
    #         cos_yaw = np.cos(self.trueYaw)
    #         sin_yaw = np.sin(self.trueYaw)
    #         velocity_world_x = (self.velocity.x * cos_yaw - self.velocity.y * sin_yaw)
    #         velocity_world_y = (self.velocity.x * sin_yaw + self.velocity.y * cos_yaw)


            self.velocity.x = float(global_vel[0, 0]) 
            self.velocity.y = float(global_vel[0, 1]) 
            self.velocity.z = float(global_vel[0, 2]) 
            self.yaw = float(output[0, 0])

            print(f"PA Velocity Command: vx: {self.velocity.x:.2f}, vy: {self.velocity.y:.2f}, vz: {self.velocity.z:.2f}, yaw_rate: {self.yaw:.2f}")


            offboard_msg = OffboardControlMode()
            offboard_msg.timestamp = int(Clock().now().nanoseconds / 1000)
            offboard_msg.position = False
            offboard_msg.velocity = True
            offboard_msg.acceleration = False
            self.publisher_offboard_mode.publish(offboard_msg)            

            # Compute velocity in the world frame
            # cos_yaw = np.cos(self.trueYaw)
            # sin_yaw = np.sin(self.trueYaw)
            # velocity_world_x = (self.velocity.x * cos_yaw - self.velocity.y * sin_yaw)
            # velocity_world_y = (self.velocity.x * sin_yaw + self.velocity.y * cos_yaw)

            # Create and publish TrajectorySetpoint message with NaN values for position and acceleration
            trajectory_msg = TrajectorySetpoint()
            trajectory_msg.timestamp = int(Clock().now().nanoseconds / 1000)
            trajectory_msg.velocity[0] = self.velocity.y
            trajectory_msg.velocity[1] = self.velocity.x
            trajectory_msg.velocity[2] = self.velocity.z
            trajectory_msg.position[0] = float('nan')
            trajectory_msg.position[1] = float('nan')
            trajectory_msg.position[2] = float('nan')
            trajectory_msg.acceleration[0] = float('nan')
            trajectory_msg.acceleration[1] = float('nan')
            trajectory_msg.acceleration[2] = float('nan')
            trajectory_msg.yaw = float('nan')
            trajectory_msg.yawspeed = self.yaw

            self.publisher_trajectory.publish(trajectory_msg)

                                
    # def PA_controller_callback(self):
    #     if(self.offboardMode == True):
    #         # Publish offboard control modes
    #         input_data = self.prepare_PA_input()
    #         offboard_msg = OffboardControlMode()
    #         offboard_msg.timestamp = int(Clock().now().nanoseconds / 1000)
    #         offboard_msg.position = False
    #         offboard_msg.velocity = True
    #         offboard_msg.acceleration = False
    #         self.publisher_offboard_mode.publish(offboard_msg)            

    #         # Compute velocity in the world frame
    #         cos_yaw = np.cos(self.trueYaw)
    #         sin_yaw = np.sin(self.trueYaw)
    #         velocity_world_x = (self.velocity.x * cos_yaw - self.velocity.y * sin_yaw)
    #         velocity_world_y = (self.velocity.x * sin_yaw + self.velocity.y * cos_yaw)

    #         # Create and publish TrajectorySetpoint message with NaN values for position and acceleration
    #         trajectory_msg = TrajectorySetpoint()
    #         trajectory_msg.timestamp = int(Clock().now().nanoseconds / 1000)
    #         trajectory_msg.velocity[0] = velocity_world_x
    #         trajectory_msg.velocity[1] = velocity_world_y
    #         trajectory_msg.velocity[2] = self.velocity.z
    #         trajectory_msg.position[0] = float('nan')
    #         trajectory_msg.position[1] = float('nan')
    #         trajectory_msg.position[2] = float('nan')
    #         trajectory_msg.acceleration[0] = float('nan')
    #         trajectory_msg.acceleration[1] = float('nan')
    #         trajectory_msg.acceleration[2] = float('nan')
    #         trajectory_msg.yaw = float('nan')
    #         trajectory_msg.yawspeed = self.yaw

    #         self.publisher_trajectory.publish(trajectory_msg)


def main(args=None):
    rclpy.init(args=args)

    offboard_control = OffboardControl()

    rclpy.spin(offboard_control)

    offboard_control.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
