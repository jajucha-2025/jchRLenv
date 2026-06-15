import math
import cv2

from vehicle.car import Car
from vehicle.position_detector import pd_rk4
from env.map_loader import MapLoader
from env.observation_builder import ObservationBuilder
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

renderer = Renderer(
    map_loader
)

obs_builder = ObservationBuilder(
    map_loader
)

obs = obs_builder.build(
    x_cm=100,
    y_cm=50,
    theta=math.pi/4
    # theta=0
)

print(obs.shape)


while True:
    
    cv2.imshow(
        "obs",
        obs
    )
    renderer.show(car)