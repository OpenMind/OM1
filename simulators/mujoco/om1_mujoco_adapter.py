import mujoco
import requests
import numpy as np
import threading
import time
from io import BytesIO
import matplotlib
matplotlib.use('Agg') # NEW: Use non-GUI backend for plotting
import matplotlib.pyplot as plt

# --- OM1 CONFIGURATION & CONSTANTS ---
OM1_INPUT_API_URL = "http://127.0.0.1:5000/api/v1/input"
SENSOR_INPUT_NAME = "MujocoSensorData"
MODEL_PATH = "pendulum.xml"

class MujocoAdapter:
    def __init__(self, model_path=MODEL_PATH):
        # 1. Initialize MuJoCo Model
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)

        self.actuator_id = self.model.actuator(0).id
        self.slide_joint_id = self.model.joint('slide').id
        self.hinge_joint_id = self.model.joint('hinge').id

        self.control_input = 0.0
        self.is_running = True

        # NEW: Matplotlib Plot Data Storage
        self.plot_history = {'time': [], 'pole_angle': []}
        self.plot_limit = 500  # Max data points to keep plot fast

        # 4. Start the background threads
        self.sim_thread = threading.Thread(target=self._run_simulation_loop)
        self.input_thread = threading.Thread(target=self._stream_sensor_data)
        
        self.sim_thread.start()
        self.input_thread.start()
        print("MuJoCo Adapter and Sensor Stream initialized.")

    # --- SIMULATION LOOP (Runs Physics) ---
    def _run_simulation_loop(self):
        while self.is_running:
            # 1. Apply control from the OM1 action API
            self.data.ctrl[self.actuator_id] = self.control_input

            # 2. Step the simulation
            mujoco.mj_step(self.model, self.data)

            # Update plot data history
            current_time = float(self.data.time)
            current_angle = float(self.data.qpos[self.hinge_joint_id])
            
            self.plot_history['time'].append(current_time)
            self.plot_history['pole_angle'].append(current_angle)
            
            # Keep the history length reasonable
            if len(self.plot_history['time']) > self.plot_limit:
                self.plot_history['time'].pop(0)
                self.plot_history['pole_angle'].pop(0)

            # 3. Sleep to regulate physics time
            time.sleep(self.model.opt.timestep / 2.0)

    # --- NEW: FRAME GENERATION METHOD (Plot PNG) ---
    def get_latest_frame(self):
        # 1. Create figure without using pyplot to prevent memory leaks in Flask
        fig, ax = plt.subplots(figsize=(6, 4))
        
        # 2. Plot the data
        ax.plot(self.plot_history['time'], self.plot_history['pole_angle'], label='Pole Angle (rad)')
        
        # 3. Format the plot
        ax.set_title('Inverted Pendulum State (Live)')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Angle (rad)')
        ax.legend(loc='lower left')
        ax.grid(True)
        ax.set_ylim([-3.5, 3.5]) # Set limits similar to the demo plot

        # 4. Save the figure to an in-memory buffer as a PNG image
        buf = BytesIO()
        fig.savefig(buf, format="png")
        
        # 5. Close the figure to free memory
        plt.close(fig) 
        
        return buf.getvalue()

    # --- SENSOR FEEDBACK (Runs Input Stream) ---
    def _get_sensor_data(self):
        # ... (rest of the class remains the same)
        sensor_data = {
            "cart_position": float(self.data.qpos[self.slide_joint_id]),
            "pole_angle": float(self.data.qpos[self.hinge_joint_id]),
            "cart_velocity": float(self.data.qvel[self.slide_joint_id]),
            "pole_angular_velocity": float(self.data.qvel[self.hinge_joint_id]),
            "time": float(self.data.time)
        }
        return sensor_data

    def _stream_sensor_data(self):
        # ... (rest of the class remains the same)
        while self.is_running:
            data = self._get_sensor_data()
            payload = {
                "input_name": SENSOR_INPUT_NAME,
                "data": data,
                "timestamp": int(time.time() * 1000)
            }
            try:
                requests.post(OM1_INPUT_API_URL, json=payload, timeout=0.5)
            except requests.exceptions.RequestException:
                pass
            time.sleep(0.1)


    # --- ACTION HANDLER (API Endpoint Logic) ---
    def handle_control_command(self, command_data):
        if 'force' in command_data:
            force = command_data['force']
            self.control_input = force
            print(f"Received control command: set force to {force} N")
            return {"status": "success", "result": f"Applied force {force}"}
        else:
            return {"status": "error", "message": "Missing 'force' parameter"}

    def close(self):
        self.is_running = False
        self.sim_thread.join()
        self.input_thread.join()
        print("MuJoCo Adapter closed.")

# --- Main Entry point ---
if __name__ == '__main__':
    adapter = MujocoAdapter()
    print("Adapter running.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        adapter.close()
