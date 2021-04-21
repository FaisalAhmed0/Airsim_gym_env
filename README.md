
    This Environment is a gym API that wraps Microsoft_airsim for unity.
    It is implemented specifically for learning mapless navigation and stability while flying,
    but it can be modified easily for other purposes, [this video](https://www.youtube.com/watch?v=euKBlw2TQq8) shows the envinrment with hand engineered controller 

    

## Observation Space:
    Observation: 
        Type: Box(14)
        Num Observation                
        0   Quad postion x            
        1   Quad postion y            
        2   Quad postion z            
        3   Quad Velocity x           
        4   Quad Velocity y           
        5   Quad Velocity z           
        6   Quad Orientation roll     
        7   Quad Orientation pitch    
        8   Quad Orientation yaw      
        9  Quad Angular_velocity x    
        10  Quad Angular_velocity y   
        11  Quad Angular_velocity z   
        12 Distance to Goal           
        13 angle of the goal          
               
    The Orientations are originally recivied as a quaternioin vector from the airsim API 
    then transformed to roll, pitch, and yaw for thier easier interpretability.

## Action space:
    The Action vector consist of 3 continuous components
        Actions:
        Type: Box(3)                   
        Num Action                     
        0   roll rate                  
        1   pitch rate                 
        2   yaw rate


## Depndensices:
    Unity game Engine and Airsim simulator should be installed  for detailed instruction see the website:
    [https://microsoft.github.io/AirSim/Unity/]
    To install the environment run the command pip install -e . inside the gym-airsim directory 


    

