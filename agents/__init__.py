from agents.c51 import C51Agent
from agents.divl import DIVLAgent
from agents.floq import FloQAgent
from agents.flow_critic import FlowCriticAgent
from agents.implicit_c51 import ImplicitC51Agent
from agents.implicit_iqn import ImplicitIQNAgent
from agents.implicit_qc import ImplicitQCAgent
from agents.implicit_value_forcing import IVFAgent
from agents.iqn import IQNAgent
from agents.pcbf import PCBFAgent
from agents.qc import QCAgent
from agents.value_flows import ValueFlowsAgent
from agents.value_forcing import VFAgent

agents = dict(
    c51=C51Agent,
    divl=DIVLAgent,
    floq=FloQAgent,
    flow_critic=FlowCriticAgent,
    implicit_c51=ImplicitC51Agent,
    implicit_iqn=ImplicitIQNAgent,
    implicit_qc=ImplicitQCAgent,
    implicit_value_forcing=IVFAgent,
    iqn=IQNAgent,
    pcbf=PCBFAgent,
    qc=QCAgent,
    value_flows=ValueFlowsAgent,
    value_forcing=VFAgent,
)
