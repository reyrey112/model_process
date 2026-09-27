CREATE EXTENSION IF NOT EXISTS timescaledb;
CREATE TABLE IF NOT EXISTS process_data (
    "Timestamp" TIMESTAMP NOT NULL,
    "action_feed_flow_rate" DOUBLE PRECISION,
    "action_coolant_flow_rate" DOUBLE PRECISION,
    "action_agitator_speed_rpm" DOUBLE PRECISION,
    "state_ambient_temp_effect" DOUBLE PRECISION,
    "state_reactor_temp" DOUBLE PRECISION,
    "state_reactor_pressure" DOUBLE PRECISION,
    "state_reaction_rate" DOUBLE PRECISION,
    "state_conversion_rate" DOUBLE PRECISION,
    "state_selectivity" DOUBLE PRECISION,
    "state_yield_pct" DOUBLE PRECISION,
    "state_vibration_rms" DOUBLE PRECISION,
    "state_motor_current" DOUBLE PRECISION,
    "state_power_consumption_kw" DOUBLE PRECISION,
    "state_operating_regime_A" DOUBLE PRECISION,
    "state_operating_regime_B" DOUBLE PRECISION,
    "state_reactor_id_A_R1" DOUBLE PRECISION,
    "state_reactor_id_A_R2" DOUBLE PRECISION,
    "state_reactor_id_A_R3" DOUBLE PRECISION,
    "state_reactor_id_B_R1" DOUBLE PRECISION,
    "state_reactor_id_B_R2" DOUBLE PRECISION,
    "state_reactor_id_B_R3" DOUBLE PRECISION,
    "target_temp_setpoint" DOUBLE PRECISION,
    "target_pressure_setpoint" DOUBLE PRECISION,
    "quality" DOUBLE PRECISION
);

-- Optional but recommended since you're using TimescaleDB:
SELECT create_hypertable('process_data', 'Timestamp', if_not_exists => TRUE);