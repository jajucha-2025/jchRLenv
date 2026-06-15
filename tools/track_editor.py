import cv2
import json
import numpy as np
from pathlib import Path


class TrackEditor:

    def __init__(
        self,
        map_path,
        save_path
    ):

        self.map_path = map_path
        self.save_path = save_path

        self.map = cv2.imread(
            map_path
        )

        if self.map is None:
            raise RuntimeError(
                f"cannot load map: {map_path}"
            )

        self.checkpoints = []

        self.spawn = None
        self.finish = None

        self.mouse_x = 0
        self.mouse_y = 0

        cv2.namedWindow(
            "Track Editor"
        )

        cv2.setMouseCallback(
            "Track Editor",
            self.mouse_callback
        )

    def mouse_callback(
        self,
        event,
        x,
        y,
        flags,
        param
    ):

        self.mouse_x = x
        self.mouse_y = y

        if event == cv2.EVENT_LBUTTONDOWN:

            self.checkpoints.append(
                (x, y)
            )

            print(
                f"checkpoint {len(self.checkpoints)-1}"
                f" added: ({x}, {y})"
            )

        elif event == cv2.EVENT_RBUTTONDOWN:

            if len(self.checkpoints) == 0:
                return

            dists = []

            for px, py in self.checkpoints:

                d = (
                    (px - x) ** 2
                    +
                    (py - y) ** 2
                )

                dists.append(d)

            idx = int(
                np.argmin(dists)
            )

            removed = self.checkpoints.pop(
                idx
            )

            print(
                f"removed checkpoint {idx}: "
                f"{removed}"
            )

    def save(self):

        data = {

            "spawn": None,

            "finish": None,

            "checkpoints": []
        }

        if self.spawn is not None:

            data["spawn"] = {

                "x": float(
                    self.spawn["x"]
                ),

                "y": float(
                    self.spawn["y"]
                ),

                "theta": float(
                    self.spawn["theta"]
                )
            }

        if self.finish is not None:

            data["finish"] = {

                "x": float(
                    self.finish[0]
                ),

                "y": float(
                    self.finish[1]
                )
            }

        for idx, (x, y) in enumerate(
            self.checkpoints
        ):

            data["checkpoints"].append({

                "id": idx,

                "x": float(x),

                "y": float(y)
            })

        with open(
            self.save_path,
            "w"
        ) as f:

            json.dump(
                data,
                f,
                indent=4
            )

        print(
            f"saved -> {self.save_path}"
        )

    def draw(self):

        frame = self.map.copy()

        if self.spawn is not None:

            x = int(self.spawn["x"])
            y = int(self.spawn["y"])

            theta = self.spawn["theta"]

            cv2.circle(
                frame,
                (x, y),
                10,
                (0,255,0),
                -1
            )

            arrow_len = 40

            tip_x = int(
                x
                -
                np.sin(theta) * arrow_len
            )

            tip_y = int(
                y
                +
                np.cos(theta) * arrow_len
            )

            cv2.arrowedLine(
                frame,
                (x, y),
                (tip_x, tip_y),
                (0,255,0),
                3
            )

            cv2.putText(
                frame,
                "S",
                (x + 10, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0,255,0),
                2
            )

        if self.finish is not None:

            cv2.circle(
                frame,
                self.finish,
                10,
                (0,0,255),
                -1
            )

            cv2.putText(
                frame,
                "F",
                (
                    self.finish[0] + 10,
                    self.finish[1]
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0,0,255),
                2
            )

        for idx, (x, y) in enumerate(
            self.checkpoints
        ):

            cv2.circle(
                frame,
                (x, y),
                6,
                (255,255,0),
                -1
            )

            cv2.putText(
                frame,
                str(idx),
                (x + 8, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255,255,0),
                1
            )

        return frame

    def run(self):

        while True:

            frame = self.draw()

            cv2.imshow(
                "Track Editor",
                frame
            )

            key = cv2.waitKey(10)

            if key == ord("q"):

                print("quit editor")

                break

            elif key == ord("s"):

                self.spawn = {
                    "x": self.mouse_x,
                    "y": self.mouse_y,
                    "theta": 0.0
                }

                print(
                    f"spawn set: "
                    f"({self.mouse_x}, {self.mouse_y})"
                )

            elif key == ord("a"):

                if self.spawn is not None:

                    self.spawn["theta"] -= np.deg2rad(5)

            elif key == ord("d"):

                if self.spawn is not None:

                    self.spawn["theta"] += np.deg2rad(5)

            elif key == ord("f"):

                self.finish = (
                    self.mouse_x,
                    self.mouse_y
                )

                print(
                    f"finish set: "
                    f"{self.finish}"
                )

            elif key == ord("p"):

                self.save()

        cv2.destroyAllWindows()


if __name__ == "__main__":

    editor = TrackEditor(

        map_path=
        "maps/track.png",

        save_path=
        "maps/track.json"
    )

    editor.run()