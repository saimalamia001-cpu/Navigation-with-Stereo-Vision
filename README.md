# Navigation-with-Stereo-Vision
Implemented with a wheeled robot with stereo obstacle detection, occupancy mapping, A* replanning, and wheel/gyro odometry.

<img width="1336" height="1177" alt="image" src="https://github.com/user-attachments/assets/00853eeb-dbf3-4c50-88fe-e574ba64a14e" />



## Simulation Environment
Pybullet simulation environment was used to test the model.

### Stereo navigation dashboard
<img width="1280" height="1000" alt="obstacle-view" src="https://github.com/user-attachments/assets/9e55c819-9c5e-4af3-ae90-b2cab1f41a23" />

## Trial videos

### Path search 
https://github.com/user-attachments/assets/fb6abbe8-7113-4c09-b526-15489a2e847b

The robot reached the goal in 53.6 simulated seconds.
## Hardware

| Component | Quantity | Purpose |
|---|---:|---|
| ESP32-CAM with OV2640 camera | 2 | Left and right stereo cameras |
| ESP32 development board | 1 | Motor control and communication |
| L298N dual H-bridge motor driver | 1 | Controls the left and right motor pairs |
| Yellow TT DC geared motor | 4 | Four-wheel drive |
| 3.7 V Li-ion cell | 4 | Robot power |
| Battery holder / pack | 1 | Holds and connects the cells |
| Robot chassis | 1 | Mounts the electronics and motors |
| Wheels | 4 | Matched to the TT motor shafts |
| Rigid stereo camera bracket | 1 | Maintains camera alignment and baseline |
| Wires, connectors and mounting hardware | As needed | Electrical connections and assembly |
| Laptop or onboard computer | 1 | Stereo processing, mapping and navigation |

## How it works

1. Two cameras capture left and right images.
2. Stereo matching estimates obstacle depth.
3. Depth observations update a probabilistic occupancy grid.
4. A* generates a route through the map.
5. A differential-drive controller follows the route.
6. New obstacle observations trigger replanning when necessary.

Wheel encoders and a simulated gyro estimate robot movement.



