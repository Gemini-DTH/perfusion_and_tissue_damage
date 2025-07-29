import subprocess

from desist.eventhandler.api import API
from desist.isct.utilities import read_yaml, write_yaml

# Default path (inside container) for configuration files
# perfusion_config_file = '/app/perfusion/config_basic_flow_solver.yaml'
perfusion_dir = 'pf_sim'
# perfusion_config_name='perfusion_config.yaml'
perfusion_config_name = 'config_coupled_flow_solver.yaml'
OXYGEN_ROOT = "/app/oxygen"
# OXYGEN_ROOT = "./oxygen"
oxygen_config_file = OXYGEN_ROOT + '/config_oxygen_solver.yaml'

class API(API):
    def event(self):

        # output paths
        res_folder = self.result_dir.joinpath(f"{perfusion_dir}")
        perfusion_config_file = str(self.result_dir.joinpath(perfusion_config_name))

        # read properties from previous perfusion step
        perfusion_config = read_yaml(perfusion_config_file)
        bc_file = perfusion_config.get('input', {}).get('inlet_boundary_file')
        assert bc_file is not None, "no path for inlet boundary conditions"

        # read configuration for oxygen
        solver_config = read_yaml(oxygen_config_file)

        # update configuration
        solver_config['input']['para_path'] = f'{res_folder.resolve()}/'
        solver_config['input']['read_inlet_boundary'] = True
        solver_config['input']['pialBC_file'] = bc_file
        # brain_mesh = self.patient_dir.joinpath('brain_meshes/clustered.xdmf')
        brain_mesh = self.result_dir.joinpath('bf_sim/clustered_mesh.xdmf')
        solver_config['input']['mesh_file'] = str(brain_mesh)

        # write configuration to disk

        config_path = self.result_dir.joinpath('oxygen_config.yaml')
        write_yaml(config_path, solver_config)

        # setup command for oxygen model with arguments
        oxygen_cmd = [
            "python3", "oxygen_main.py", "--config_file",
            f"{str(config_path)}", "--rslt", f"{res_folder}/"
        ]

        # invoke oxygen model
        print(f"Evaluting: '{' '.join(oxygen_cmd)}'", flush=True)
        subprocess.run(oxygen_cmd, check=True, cwd=OXYGEN_ROOT)

    def example(self):
        self.event()

    def test(self):
        self.example()
