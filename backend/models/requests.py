from pydantic import BaseModel
from datetime import datetime
from typing import List

class ProcessDataRow(BaseModel):
    Timestamp: datetime
    action_feed_flow_rate: float
    action_coolant_flow_rate: float
    action_agitator_speed_rpm: float
    state_ambient_temp_effect: float
    state_reactor_temp: float
    state_reactor_pressure: float
    state_reaction_rate: float
    state_conversion_rate: float
    state_selectivity: float
    state_yield_pct: float
    state_vibration_rms: float
    state_motor_current: float
    state_power_consumption_kw: float
    state_operating_regime_A: float
    state_operating_regime_B: float
    state_reactor_id_A_R1: float
    state_reactor_id_A_R2: float
    state_reactor_id_A_R3: float
    state_reactor_id_B_R1: float
    state_reactor_id_B_R2: float
    state_reactor_id_B_R3: float
    target_temp_setpoint: float
    target_pressure_setpoint: float
    quality: float

class DBWriteRequest(BaseModel):
    data_tuples: List[ProcessDataRow]
    all_columns: List