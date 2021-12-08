import subprocess

from desist.eventhandler.api import API
from desist.isct.utilities import read_yaml, write_yaml

# Default path (inside container) for configuration files
tissue_health_config_file = '/app/tissue_health/config_tissue_damage.yaml'
perfusion_dir = 'pf_sim'


class API(API):
    def event(self):

        # output paths
        res_folder = self.result_dir.joinpath(f"{perfusion_dir}")

        # read configuration for oxygen
        solver_config = read_yaml(tissue_health_config_file)

        # baseline, stroke, treatment directories
        paths = []
        for event in self.events:
            path = self.patient_dir.joinpath(event.get('event'))
            path = path.joinpath(perfusion_dir)

            if path.joinpath('perfusion_stroke.xdmf').exists():
                path = path.joinpath('perfusion_stroke.xdmf')
            else:
                path = path.joinpath('perfusion.xdmf')
            paths.append(path)

        baseline_dir, stroke_dir, treatment_dir = paths

        # update configuration
        solver_config['input']['healthyfile'] = str(baseline_dir)
        solver_config['input']['strokefile'] = str(stroke_dir)
        solver_config['input']['treatmentfile'] = str(treatment_dir)
        solver_config['output']['res_fldr'] = str(res_folder)

        # Set arrival time from onset to treatment, the sum of `dur_oer`
        # and `dur_erg`, converted from minutes to hours.
        arrival_time = self.patient['dur_onset_groin'] / 60
        solver_config['input']['arrival_time'] = arrival_time

        # The recovery time is expressed as the interval in number of days
        # after which the tissue health is re-evaluated. The default value is
        # set to a single day if no value is provided in the virtual patient.
        #
        # TODO: we might want to support arrays of recovery times, i.e.
        # `recovery_time = [1, 5, 7]`, to allow for probing the difference in
        # tissue health across various measurement intervals after treatment
        # (see issue #11 in `insist-trials` repository).
        num_recovery_days = self.patient.get('tissue_health_follow_up_days', 1)
        solver_config['input']['recovery_time'] = num_recovery_days * 24
        brain_mesh = self.patient_dir.joinpath('brain_meshes/b0000/clustered.xdmf')
        solver_config['input']['mesh_file'] = str(brain_mesh)

        # write the updated yaml to the patient directory
        config_path = self.result_dir.joinpath(
                f'{perfusion_dir}/tissue_health_config.yaml')
        write_yaml(config_path, solver_config)

        # the path where we are going to write the infarct volumes
        outcome_path = self.patient_dir.joinpath('tissue_health_outcome.yml')

        # initialise the simulation with the updated configuration files
        tissue_health_cmd = [
            "python3",
            "infarct_estimate_treatment.py",
            str(config_path),
            str(outcome_path),
        ]
        subprocess.run(tissue_health_cmd, check=True, cwd="/app/tissue_health")

    def example(self):
        pass

    def test(self):
        pass
