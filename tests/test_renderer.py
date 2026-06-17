import math
import cv2

from vehicle.car import Car
from vehicle.position_detector import pd_rk4
from env.map_loader import MapLoader
from env.observation_builder import ObservationBuilder
from env.checkpoint_manager import CheckpointManager
from render.renderer import Renderer


car = Car()

car.reset(
    x=100,
    y=50,
    # theta=0
    theta=math.pi/4
)

next_state = pd_rk4(
    car.state,
    10.0,
    0.0,
    car,
    1.0
)

print(next_state)

map_loader = MapLoader(
    "maps"
)

checkpoint_manager = CheckpointManager(
    "maps"
)

renderer = Renderer(
    map_loader,
    checkpoint_manager
)


while True:
    
    renderer.show(car)