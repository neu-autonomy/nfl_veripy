from .DiscreteQuadrotor import DiscreteQuadrotor  # noqa
from .DoubleIntegrator import DoubleIntegrator  # noqa
from .DoubleIntegratorOutputFeedback import (
    DoubleIntegratorOutputFeedback,
)  # noqa
from .DoubleIntegratorx4 import DoubleIntegratorx4  # noqa
from .Duffing import Duffing  # noqa
from .Dynamics import ContinuousTimeDynamics  # noqa
from .Dynamics import DiscreteTimeDynamics, Dynamics  # noqa
from .GroundRobotDI import GroundRobotDI  # noqa
from .GroundRobotSI import GroundRobotSI  # noqa
from .ISS import ISS  # noqa
from .Pendulum import Pendulum  # noqa
from .Quadrotor import Quadrotor  # noqa
from .Quadrotor_8D import Quadrotor_8D  # noqa
from .Quadrotor_v0 import Quadrotor_v0  # noqa
from .QuadrotorOutputFeedback import QuadrotorOutputFeedback  # noqa
from .QuadrotorOutputFeedback_v0 import QuadrotorOutputFeedback_v0  # noqa
from .Taxinet import Taxinet  # noqa
from .Unicycle import Unicycle  # noqa
from .Unity import Unity  # noqa


def get_dynamics_instance(system, state_feedback):
    if state_feedback == "FullState":
        dynamics_dict = {
            "DoubleIntegrator": DoubleIntegrator,
            "Quadrotor_v0": Quadrotor_v0,
            "Duffing": Duffing,
            "ISS": ISS,
            "GroundRobot": GroundRobotSI,
            "discrete_quadrotor": DiscreteQuadrotor,
        }
    else:
        dynamics_dict = {
            "DoubleIntegrator": DoubleIntegratorOutputFeedback,
            "Quadrotor_v0": QuadrotorOutputFeedback_v0,
        }
    return dynamics_dict[system]()
