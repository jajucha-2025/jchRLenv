import cv2

from env.map_loader import MapLoader


def main():

    loader = MapLoader(
        "maps/track.png"
    )

    img = loader.map.copy()

    x_cm = 600.0
    y_cm = 350.0

    px, py = loader.world_to_pixel(
        x_cm,
        y_cm
    )

    cv2.circle(
        img,
        (
            int(px),
            int(py)
        ),
        5,
        255,
        -1
    )

    cv2.imshow(
        "map",
        img
    )

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()