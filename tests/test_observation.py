import cv2
import math

from env.map_loader import MapLoader
from env.observation_builder import ObservationBuilder


map_loader = MapLoader(
    "maps/track.png"
)

obs_builder = ObservationBuilder(
    map_loader
)

obs = obs_builder.build(
    x_cm=100,
    y_cm=50,
    theta=math.pi/2
)

print(obs.shape)

cv2.imshow(
    "obs",
    obs
)

cv2.waitKey(0)