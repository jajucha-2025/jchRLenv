import math


def normalize_angle(theta):

    return (
        (theta + math.pi)
        % (2.0 * math.pi)
        - math.pi
    )


def ackermann_steering(
    steer_rad,
    wheelbase,
    track
):
    """
    steer_rad:
        virtual steering angle

    return:
        deltaL, deltaR
    """

    if abs(steer_rad) < 1e-8:
        return 0.0, 0.0

    radius = (
        wheelbase /
        math.tan(abs(steer_rad))
    )

    inner = math.atan(
        wheelbase /
        (
            radius
            - track / 2.0
        )
    )

    outer = math.atan(
        wheelbase /
        (
            radius
            + track / 2.0
        )
    )

    if steer_rad > 0:
        return inner, outer

    return -outer, -inner


def equivalent_steer(
    deltaL,
    deltaR
):
    """
    left/right wheel steer
    -> equivalent steer
    """

    tanL = math.tan(deltaL)
    tanR = math.tan(deltaR)

    if abs(tanL + tanR) < 1e-9:
        return 0.0

    tan_eq = (
        2.0
        * tanL
        * tanR
    ) / (
        tanL + tanR
    )

    return math.atan(
        tan_eq
    )


def _deriv(
    state,
    speed_cm_s,
    deltaL,
    deltaR,
    car
):

    x, y, theta = state

    delta = equivalent_steer(
        deltaL,
        deltaR
    )

    tan_delta = math.tan(
        delta
    )

    beta = math.atan(
        (car.LR / car.L)
        * tan_delta
    )

    denom = (
        car.L
        - car.KPO * tan_delta
    )

    if abs(denom) < 1e-6:
        omega = 0.0
    else:
        omega = (
            speed_cm_s
            * tan_delta
        ) / denom

    heading = (
        theta
        + beta
        + math.pi / 2.0
    )

    dx = (
        speed_cm_s
        * math.cos(heading)
    )

    dy = (
        speed_cm_s
        * math.sin(heading)
    )

    return (
        dx,
        dy,
        omega
    )


def pd_rk4(
    state,
    speed_cm_s,
    steer_rad,
    car,
    dt
):
    """
    state:
        (x, y, theta)

    speed_cm_s:
        real vehicle speed

    steer_rad:
        virtual steering angle
    """

    deltaL, deltaR = (
        ackermann_steering(
            steer_rad,
            car.L,
            car.T
        )
    )

    x, y, theta = state

    k1 = _deriv(
        (x, y, theta),
        speed_cm_s,
        deltaL,
        deltaR,
        car
    )

    k2 = _deriv(
        (
            x + 0.5 * dt * k1[0],
            y + 0.5 * dt * k1[1],
            theta + 0.5 * dt * k1[2]
        ),
        speed_cm_s,
        deltaL,
        deltaR,
        car
    )

    k3 = _deriv(
        (
            x + 0.5 * dt * k2[0],
            y + 0.5 * dt * k2[1],
            theta + 0.5 * dt * k2[2]
        ),
        speed_cm_s,
        deltaL,
        deltaR,
        car
    )

    k4 = _deriv(
        (
            x + dt * k3[0],
            y + dt * k3[1],
            theta + dt * k3[2]
        ),
        speed_cm_s,
        deltaL,
        deltaR,
        car
    )

    x_next = x + (
        dt / 6.0
    ) * (
        k1[0]
        + 2.0 * k2[0]
        + 2.0 * k3[0]
        + k4[0]
    )

    y_next = y + (
        dt / 6.0
    ) * (
        k1[1]
        + 2.0 * k2[1]
        + 2.0 * k3[1]
        + k4[1]
    )

    theta_next = theta + (
        dt / 6.0
    ) * (
        k1[2]
        + 2.0 * k2[2]
        + 2.0 * k3[2]
        + k4[2]
    )

    theta_next = normalize_angle(
        theta_next
    )

    return (
        x_next,
        y_next,
        theta_next
    )