In lesson 3, you'll learn about:

- Robot arms
- Inverse kinematics
- Mobile robots

You'll make a robot arm, something like this:
Hand
●
|
Link 2
|
(Joint 2)
|
Link 1
|
(Joint 1)
|
Base

## NOTES

1. Nested Bodies
   Instead of this:
   World
   ├── Arm 1
   └── Arm 2

You have this:
Base
↓
Joint 1    >> Motor 1 → Shoulder
↓
Link 1
↓
Joint 2    >> Motor 2 → Elbow
↓
Link 2
↓
●        >> End Effector → Hand

Each link inherits the movement of its parent. This is a kinematic chain.

Suppose Link 1 rotates, Link 2 HAS to rotate too since Link 1 is the parent and Link 2 is its child. Just like when your shoulder moves, your arm and hand moves too.

Each joint moves independently so you have to include multiple actuators/motors, one for the shoulder and one for the elbow.
data.ctrl[0] = 0.5      >> Motor 1 → Shoulder
data.ctrl[1] = -0.3     >> Motor 2 → Elbow

data.qpos[0]        >> Angle position/angle 1 → Shoulder
data.qpos[1]        >> Angle position/angle 2 → Elbow

__Forward Kinematics__

Given shoulder = 30degs and elbow = 45degs, where is the hand? This is **forward kinematics**, it will calc the position of a joint given joint angles.

``data.qpos[0] = 0.5``
``data.qpos[1] = 0.3``

This line is given these joints, compute where every other body is at
``mujoco.mj_forward(model, data)``

__Inverse Kinematics__
You know where the hand is, now figure out what angles the joints are at.

__Error__
In the context of control, error is a variable that measures the gap between where the joint is right now and where you want it to be. It is the foundation of **proportional controller** (the 'p' in PID control), which is the most common way to move a robotic arm.
`` error = target - current_position``, where ``target`` is the angle in radians you want the joint to reach and ``current_position`` would be equal to ``data.qpos[0]``--it's the actual, current angle of the joint.

If you simply tell a motor to move to ``1.0`` is doesn't know how much power supply to apply, it might just jump/teleport to that location but that's not really how real robots move, they move slowly to the target 0 > 0.01 > 0.02 > etc not 0 > 1.0, this p-control will make the robot move smoothly and it won't flap around.

If the error is large, the motor recieves a large signal and pushes hard to move quickly.
If error is small, the motor gets a smaller signal and slows down as it reaches the target.
If error is zero, the motor get a 0 value and stops moving.

__Terms__
Link  >>  A rigid part of the robot
Joint >>  Connects two links and allows movement
Base  >>  The fixed part of the robot
End Effector  >>  The robot's hand or tool
Kinematic Chain  >>  A series of connected links and joints
Degrees of Freedonm (DoF)  >>  The number of independent ways a robot can move. Our arm has 2 DoF
Forward Kinematics  >>  Computing the hand position from joint angles