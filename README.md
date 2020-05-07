## Description:
    This Environment is a gym API that wraps Microsoft_airsim for unity.
    It is implemented specifically for learning mapless navigation and stability while flying,
    but it can be modified easily for other purposes.

## Observation Space:
    Type: Box(12)
        Num Observation                Min            Max
        0   Quad postion x            -Inf            Inf
        1   Quad postion y            -Inf            Inf
        2   Quad postion z            -Inf            Inf
        3   Quad Velocity x           -Inf            Inf
        4   Quad Velocity y           -Inf            Inf
        5   Quad Velocity z           -Inf            Inf
        6   Quad Orientation roll     -pi/2           pi/2
        7   Quad Orientation pitch    -pi/2           pi/2
        8   Quad Orientation yaw      -pi/2           pi/2
        9  Quad Angular_velocity x    -Inf            Inf
        10  Quad Angular_velocity y   -Inf            Inf
        11  Quad Angular_velocity z   -Inf            Inf
        
    The Orientations are originally recivied as a quaternioin vector from the airsim API 
    then transformed to roll, pitch, and yaw for thier easier interpretability.

## Action space:
    The Action vector consist of 4 continuous components
        Type: Box(4)                   
        Num Action                     Min            Max   
        0   roll rate                  -1             +1
        1   pitch rate                 -1             +1
        2   yaw rate                   -1             +1
        0   throttle                    0             +1


We choose a simple objective as a specific point in the 3D space but it can be modified in the environment source code, additionally the goal is specifid according to unity frame of reference because the user can easily see this in the screenm then it transformed to the quadrotor frame of reference via the Rotation matrix.
    
![equation](https://latex.codecogs.com/gif.latex?R%20%3D%20%5Cbegin%7Bpmatrix%7D%200%20%26%200%20%26%201%5C%5C%201%20%26%200%20%26%200%5C%5C%200%20%26-1%20%26%200%20%5Cend%7Bpmatrix%7D)
 
The reward is the negative l2 distance to goal (Again it can be modifed), this just a test reward to ensure the environment is working properly.

![equation](https://latex.codecogs.com/gif.latex?r%20%3D%20%5Cvert%20%5Cvert%20Quad.%20Position%20-%20goal%20%5Cvert%20%5Cvert_%7Bl2%7D)

## Depndensices:
    Unity game Engine and Airsim simulator should be installed  for detailed instruction see the website:
    https://microsoft.github.io/AirSim/Unity/
    To install the environment run the command pip install -e . inside the gym-airsim directory 


    

