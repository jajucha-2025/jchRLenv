import cv2

from vehicle.car import Car
from env.track_judge import TrackJudge


mask = cv2.imread(
    "maps/line_mask.png",
    cv2.IMREAD_GRAYSCALE
)

judge = TrackJudge(mask)

car = Car()

car.x = 50
car.y = 75
car.theta = 0

print(
    judge.count_outside_wheels(
        car
    )
)