"""Кинематика шестизвенного манипулятора, таблица 1.1 учебника ИТМО."""

from dataclasses import dataclass
from math import atan2, cos, hypot, isfinite, pi, sin, sqrt


def _multiply(left, right):
    return [
        [sum(first * second for first, second in zip(row, column))
         for column in zip(*right)]
        for row in left
    ]


def _dh(length, twist, offset, angle):
    cosine, sine = cos(angle), sin(angle)
    ct, st = cos(twist), sin(twist)
    return [
        [cosine, -sine * ct, sine * st, length * cosine],
        [sine, cosine * ct, -cosine * st, length * sine],
        [0.0, st, ct, offset],
        [0.0, 0.0, 0.0, 1.0],
    ]


def euler_zyz(a, b, gamma):
    """Матрица ориентации Rz(a) @ Ry(b) @ Rz(gamma), радианы."""
    ca, sa, cb, sb, cg, sg = cos(a), sin(a), cos(b), sin(b), cos(gamma), sin(gamma)
    return [
        [ca * cb * cg - sa * sg, -ca * cb * sg - sa * cg, ca * sb],
        [sa * cb * cg + ca * sg, -sa * cb * sg + ca * cg, sa * sb],
        [-sb * cg, sb * sg, cb],
    ]


@dataclass(frozen=True)
class Manipulator6DOF:
    """Размеры d1, a2, d4, d6 в одинаковых единицах с x, y, z."""

    d1: float
    a2: float
    d4: float
    d6: float

    def __post_init__(self):
        if not all(isfinite(value) for value in (self.d1, self.a2, self.d4, self.d6)):
            raise ValueError("Размеры должны быть конечными числами")
        if self.a2 <= 0 or self.d4 <= 0 or self.d1 < 0 or self.d6 < 0:
            raise ValueError("Требуется a2 > 0, d4 > 0, d1 >= 0, d6 >= 0")

    def _transforms(self, joints):
        q1, q2, q3, q4, q5, q6 = joints
        return (
            _dh(0, pi / 2, self.d1, q1),
            _dh(self.a2, 0, 0, q2),
            _dh(0, pi / 2, 0, q3 + pi / 2),
            _dh(0, -pi / 2, self.d4, q4),
            _dh(0, pi / 2, 0, q5),
            _dh(0, 0, self.d6, q6),
        )

    def forward(self, joints):
        """Прямая кинематика: шесть углов в радианах -> матрица T06 4x4."""
        joints = tuple(joints)
        if len(joints) != 6 or not all(isfinite(value) for value in joints):
            raise ValueError("Нужны шесть конечных углов")
        result = [[float(row == column) for column in range(4)] for row in range(4)]
        for transform in self._transforms(joints):
            result = _multiply(result, transform)
        return result

    def inverse(self, x, y, z, a, b, gamma, *, shoulder=1, elbow=1, wrist=1):
        """Возвращает (Q1, Q2, Q3, Q4, Q5, Q6) в радианах.

        a, b, gamma — углы Эйлера ZYZ в радианах, не roll/pitch/yaw.
        shoulder, elbow, wrist: +1 или -1, выбор ветви решения.
        Недостижимая поза вызывает ValueError. Ограничения суставов
        и столкновения не учитываются. При сингулярности выбирается
        один представитель семейства решений.
        """
        if not all(isfinite(value) for value in (x, y, z, a, b, gamma)):
            raise ValueError("Координаты и углы должны быть конечными числами")
        if any(branch not in (-1, 1) for branch in (shoulder, elbow, wrist)):
            raise ValueError("Знаки ветвей должны быть +1 или -1")
        rotation = euler_zyz(a, b, gamma)
        wx = x - self.d6 * rotation[0][2]
        wy = y - self.d6 * rotation[1][2]
        wz = z - self.d6 * rotation[2][2] - self.d1
        radius = shoulder * hypot(wx, wy)
        q1 = atan2(wy, wx) if wx != 0 or wy != 0 else 0.0
        if shoulder == -1:
            q1 += pi
        cosine = ((radius / self.a2) ** 2 + (wz / self.a2) ** 2
                  - 1 - (self.d4 / self.a2) ** 2) / (2 * self.d4 / self.a2)
        if not isfinite(cosine) or abs(cosine) > 1 + 1e-12:
            raise ValueError("Поза недостижима: центр запястья вне рабочей зоны")
        cosine = max(-1.0, min(1.0, cosine))
        sine = elbow * sqrt(max(0.0, 1 - cosine * cosine))
        q3 = atan2(sine, cosine)
        q2 = atan2(wz, radius) - atan2(self.d4 * sine, self.a2 + self.d4 * cosine)
        transforms = self._transforms((q1, q2, q3, 0, 0, 0))
        base = _multiply(_multiply(transforms[0], transforms[1]), transforms[2])
        transpose = [[base[column][row] for column in range(3)] for row in range(3)]
        relative = _multiply(transpose, rotation)
        sin_q5 = hypot(relative[0][2], relative[1][2])
        if sin_q5 > 1e-12:
            q4 = atan2(wrist * relative[1][2], wrist * relative[0][2])
            q5 = atan2(wrist * sin_q5, relative[2][2])
            q6 = atan2(wrist * relative[2][1], -wrist * relative[2][0])
        else:
            q4 = 0.0
            q5 = 0.0 if relative[2][2] >= 0 else pi
            q6 = atan2(relative[1][0], relative[1][1])
        return tuple((angle + pi) % (2 * pi) - pi for angle in (q1, q2, q3, q4, q5, q6))


DEFAULT_ROBOT = Manipulator6DOF(d1=0.4, a2=0.3, d4=0.3, d6=0.1)


def inverse_kinematics(x, y, z, a, b, gamma):
    """ОЗК для демонстрационных размеров DEFAULT_ROBOT (метры, радианы)."""
    return DEFAULT_ROBOT.inverse(x, y, z, a, b, gamma)


def main():
    print("Введите x, y, z в метрах и углы a, b, gamma в радианах (Эйлер ZYZ).")
    print("Размеры модели: d1=0.4, a2=0.3, d4=0.3, d6=0.1 м.")
    values = []
    try:
        for name in ("x", "y", "z", "a", "b", "gamma"):
            while True:
                try:
                    value = float(input(f"{name} = ").strip().replace(",", "."))
                    if not isfinite(value):
                        raise ValueError
                except ValueError:
                    print("Ошибка: введите конечное число, например 0.4 или 0,4.")
                    continue
                values.append(value)
                break
        angles = inverse_kinematics(*values)
    except (EOFError, KeyboardInterrupt):
        print("\nВвод отменён.")
        return
    except ValueError as error:
        print(f"Ошибка: {error}")
        return
    print("Результат (радианы):")
    for number, angle in enumerate(angles, start=1):
        print(f"Q{number} = {angle:.10f}")


if __name__ == "__main__":
    main()
