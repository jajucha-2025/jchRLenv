import cv2


class LineMaskLoader:

    def __init__(
        self,
        image_path: str
    ):

        self.map = cv2.imread(
            image_path,
            cv2.IMREAD_GRAYSCALE
        )

        if self.map is None:
            raise RuntimeError(
                f"Cannot load line mask: {image_path}"
            )

        self.height = self.map.shape[0]
        self.width = self.map.shape[1]

        self.world_width_cm = 1200.0
        self.world_height_cm = 700.0

        self.cm_per_px_x = (
            self.world_width_cm
            / self.width
        )

        self.cm_per_px_y = (
            self.world_height_cm
            / self.height
        )

    def world_to_pixel(
        self,
        x_cm,
        y_cm
    ):

        px = (
            x_cm
            / self.cm_per_px_x
        )

        py = (
            y_cm
            / self.cm_per_px_y
        )

        return px, py