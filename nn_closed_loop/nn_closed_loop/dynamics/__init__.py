from .Dynamics import Dynamics, ContinuousTimeDynamics, DiscreteTimeDynamics
from .DoubleIntegrator import DoubleIntegrator
from .Quadrotor import Quadrotor
from .DoubleIntegratorOutputFeedback import DoubleIntegratorOutputFeedback
from .Quadrotor_8D import Quadrotor_8D
from .QuadrotorOutputFeedback import QuadrotorOutputFeedback
from .Quadrotor_v0 import Quadrotor_v0
from .QuadrotorOutputFeedback_v0 import QuadrotorOutputFeedback_v0
from .Duffing import Duffing
from .ISS import ISS
from .Unity import Unity
from .GroundRobotSI import GroundRobotSI
from .GroundRobotDI import GroundRobotDI
from .DoubleIntegratorx4 import DoubleIntegratorx4
from .Unicycle import Unicycle
from .DiscreteQuadrotor import DiscreteQuadrotor
from .Taxinet import Taxinet
from .Pendulum import Pendulum


def get_dynamics_instance(system, state_feedback):
  if state_feedback == "FullState":
    dynamics_dict = {
        "DoubleIntegrator": DoubleIntegrator,
        "Quadrotor_v0": Quadrotor_v0,
    }
  else:   
    dynamics_dict = {
        "DoubleIntegrator": DoubleIntegratorOutputFeedback,
        "Quadrotor_v0": QuadrotorOutputFeedback_v0,
    }
  return dynamics_dict[system]()
