import os
import subprocess
import shutil

from desist.eventhandler.api import API
from desist.isct.utilities import read_yaml, write_yaml

# Default path (inside the container) pointing to the YAML configuration for
# the `basic_flow_solver` routine.
permeability_config_name = 'config_permeability_initialiser.yaml'
perfusion_config_name = 'config_basic_flow_solver.yaml'
blood_flow_dir = 'bf_sim'
perfusion_dir = 'pf_sim'
# filename used for boundary conditions in CSV format
bc_fn = 'boundary_conditions_file.csv'
# filename used for perfusion output required for simplified infarct values
pf_outfile = 'perfusion.xdmf'
# directory for where to generate and store patient brain meshes
PERFUSION_ROOT = "/app/perfusion/"
# PERFUSION_ROOT = "./perfusion/"
MAIN_ROOT = "/app/"
# MAIN_ROOT = "./"


class API(API):
    def event(self):

        # patient's brain meshes and permeability information
        brain_meshes = self.result_dir.joinpath(f'{blood_flow_dir}')
        permeability_dir = self.result_dir.joinpath('permeability')

        if not brain_meshes.exists():
            error_msg = f"""Brain meshes and permeability files are not present
            although previous events have been evaluated. This is not supported
            and requires investigation why {brain_meshes} or {permeability_dir}
            are not present anymore on the system."""
            assert self.event_id == 0, error_msg

            # generate clustering files from blood flow output
            clustered_mesh = self.result_dir.joinpath(f'{blood_flow_dir}/clustered_mesh.msh')
            clustering_result_dir = brain_meshes.joinpath('clustered')

            brain_mesh_cmd = [
                    "python3",
                    "convert_msh2hdf5.py",
                    str(clustered_mesh),
                    str(clustering_result_dir)
            ]
            print(f"Evaluating: '{' '.join(brain_mesh_cmd)}'", flush=True)
            subprocess.run(brain_mesh_cmd, check=True, cwd=PERFUSION_ROOT)

            VP_mesh_cmd = [
                "python3",
                "VP_mesh_prep.py",
                "--bsl_msh_fldr",
                str(clustering_result_dir),
                "--age",
                str(self.patient['age']),
                "--sex",
                str(self.patient['sex']),
                "--forced"
            ]
            print(f"Evaluating: '{' '.join(VP_mesh_cmd)}'", flush=True)
            subprocess.run(VP_mesh_cmd, check=True, cwd=MAIN_ROOT)

        if not permeability_dir.exists():
            # generate permeability meshes after clustering
            clustered_mesh = brain_meshes.joinpath('clustered_mesh.xdmf')

            permeability_config_file = str(self.result_dir.joinpath(permeability_config_name))
            if not self.result_dir.joinpath(permeability_config_name).exists():
                shutil.copy(PERFUSION_ROOT + "/config_permeability_initialiser.yaml", str(self.result_dir))

            perm_config = read_yaml(permeability_config_file)
            perm_config['input']['mesh_file'] = str(clustered_mesh)
            perm_config['output']['res_fldr'] = f'{permeability_dir}/'

            # config_path = str(self.result_dir.joinpath('config_permeability_initialiser.yaml'))
            write_yaml(permeability_config_file, perm_config)

            permeability_cmd = [
                "python3",
                "permeability_initialiser.py",
                "--config_file",
                str(permeability_config_file)
            ]
            print(f"Evaluating: '{' '.join(permeability_cmd)}'", flush=True)
            subprocess.run(permeability_cmd, check=True, cwd=PERFUSION_ROOT)

        assert brain_meshes.exists(), f"Brain meshes not at: '{brain_meshes}'."

        # output paths for perfusion simulation
        res_folder = self.result_dir.joinpath(f"{perfusion_dir}")
        os.makedirs(res_folder, exist_ok=True)

        # update configuration for perfusion
        perfusion_config_file = str(self.result_dir.joinpath(perfusion_config_name))
        solver_config = read_yaml(perfusion_config_file)

        # ensure boundary conditions are being read from input files
        solver_config['input']['read_inlet_boundary'] = True
        bc_file = self.result_dir.joinpath(f'{blood_flow_dir}/{bc_fn}')
        solver_config['input']['inlet_boundary_file'] = str(bc_file.resolve())
        solver_config['input']['mesh_file'] = str(self.result_dir.joinpath('bf_sim/clustered_mesh.xdmf'))
        solver_config['input']['permeability_folder'] = f"{self.result_dir.joinpath('permeability')}/"

        # cannot proceed without boundary conditions
        msg = f"Boundary conditions `1d-blood-flow` not present: `{bc_file}`"
        assert os.path.isfile(bc_file), msg

        # update output settings
        # config_path = self.result_dir.joinpath(
        #     f'{perfusion_dir}/perfusion_config.yaml')
        write_yaml(perfusion_config_file, solver_config)

        # form command to evaluate perfusion
        solve_cmd = [
            "python3", "basic_flow_solver.py", "--res_fldr", f"{res_folder}/",
            "--config_file", f"{str(perfusion_config_file)}"
        ]

        print(f"Evaluating: '{' '.join(solve_cmd)}'", flush=True)
        subprocess.run(solve_cmd, check=True, cwd=PERFUSION_ROOT)

        # terminate baseline scenario
        if self.event_id == 0:
            return

        # extract settings
        if not self.current_model.get('evaluate_infarct_estimates', False):
            return

        labels = [event.get('event') for event in self.events]

        # baseline scenario result directories
        baseline = self.patient_dir.joinpath(labels[0])
        baseline = baseline.joinpath(perfusion_dir)
        baseline = baseline.joinpath(pf_outfile)

        # occluded scenario assumed to be the current result
        occluded = self.patient_dir.joinpath(labels[1])
        occluded = occluded.joinpath(perfusion_dir)
        occluded = occluded.joinpath(pf_outfile)

        for path in [baseline, occluded]:
            assert os.path.exists(path), f"File not found: '{path}'."

        # evaluate preliminary infarct volumes at multiple thresholds
        infarct_cmd = [
            "python3",
            "infarct_calculation_thresholds.py",
            "--config_file",
            f"{str(perfusion_config_file)}",
            "--baseline",
            f"{str(baseline)}",
            "--occluded",
            f"{str(occluded)}",
            "--res_fldr",
            f"{res_folder}/",
            "--thresholds",
            f"{self.current_model.get('infarct_levels', 21)}",
            "--mesh_file",
            str(self.result_dir.joinpath(f"{perfusion_dir}", "clustered_mesh.xdmf"))
        ]
        print(f"Evaluating: '{' '.join(infarct_cmd)}'", flush=True)
        subprocess.run(infarct_cmd, check=True, cwd=PERFUSION_ROOT)

        # convert perfusion FEM result to NIFTI
        res2img_cmd = [
            "python3",
            "convert_res2img.py",
            "--config_file",
            f"{str(perfusion_config_file)}",
            "--res_fldr",
            f"{res_folder}/",
            "--variable",
            "perfusion",
            "--save_figure",
        ]
        print(f"Evaluating: '{' '.join(res2img_cmd)}'", flush=True)
        subprocess.run(res2img_cmd, check=True, cwd=PERFUSION_ROOT)


    def example(self):
        # when running the example, we need to generate some dummy input
        # for the boundary conditions, for this, use the `BC_creator.py`
        res_folder = self.result_dir.joinpath(f"{perfusion_dir}")
        os.makedirs(res_folder, exist_ok=True)

        # The current event simulation does not require the brain meshes and
        # therefore the default container does not contain the _extracted_
        # meshes by default. So, when running in CI/CD we extract the
        # compressed archive manually before running the event.
        import tarfile
        tar = tarfile.open("/app/brain_meshes.tar.xz", "r:xz")
        tar.extractall("/patient/tmp")
        tar.close()

        # move the contents to the expected brain mesh location
        import shutil
        res_bf_folder = self.result_dir.joinpath(f"{blood_flow_dir}")
        os.makedirs(res_bf_folder, exist_ok=True)
        files = os.listdir("/patient/tmp/brain_meshes/b0000")
        for f in files:
            shutil.move(os.path.join("/patient/tmp/brain_meshes/b0000", f), str(res_bf_folder))

        # Renaming the files
        os.rename(str(res_bf_folder.joinpath("clustered.xdmf")),
                  str(res_bf_folder.joinpath("clustered_mesh.xdmf")))
        os.rename(str(res_bf_folder.joinpath("clustered_facet_region.xdmf")),
                  str(res_bf_folder.joinpath("clustered_mesh_facet_region.xdmf")))
        os.rename(str(res_bf_folder.joinpath("clustered_physical_region.xdmf")),
                  str(res_bf_folder.joinpath("clustered_mesh_physical_region.xdmf")))

        shutil.copy(os.path.join(PERFUSION_ROOT, perfusion_config_name), str(self.result_dir))
        perfusion_config_file = os.path.join(str(self.result_dir), perfusion_config_name)

        bc_cmd = [
            "python3",
            "BC_creator.py",
            "--occluded",
            "--res_fldr",
            f"{res_folder}/",
            "--config_file",
            f"{perfusion_config_file}",
            "--folder",
            f"{res_folder}/",
            "--mesh_file",
            str(res_bf_folder.joinpath("clustered_mesh.xdmf"))
        ]

        print(f"Evaluating: '{' '.join(bc_cmd)}'", flush=True)
        subprocess.run(bc_cmd, check=True, cwd=PERFUSION_ROOT)

        # rename the boundary conditions file to match the trial scenario
        src = res_folder.joinpath('BCs.csv')
        dst = self.result_dir.joinpath(f"bf_sim/{bc_fn}")

        os.makedirs(dst.parent, exist_ok=True)
        os.rename(src, dst)

        # run event with example boundary conditions
        self.event()

    def test(self):
        self.example()
