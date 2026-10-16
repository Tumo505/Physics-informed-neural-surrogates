# Mechanics verification

{
  "source": "/Users/tumokgabeng/Projects/CMBBE/cardiac-lv-surrogate/data/raw/benchmark/benchmark_article_data/meshes_and_fibers/data/monoventricle_mesh/ellipsoid_0.005.h5",
  "n_nodes": 4577,
  "n_tetrahedra": 17625,
  "units": "benchmark mesh m; volume m^3 (toolkit README)",
  "total_reference_volume_m3": 0.00017769451479940395,
  "minimum_element_volume_m3": 4.419953333139041e-10,
  "affine_F_max_abs_error": 1.823888262642015e-14,
  "min_affine_detF": 1.1318999999999855,
  "interpretation": "kinematics verification on authentic cardiac mesh, manufactured affine displacement; NOT solver solution or cardiac physics validation",
  "gates": {
    "1_translation": "unit test",
    "2_identity": "neo-Hookean unit only",
    "3_affine": "actual benchmark mesh",
    "4_volume_gradients": "unit and benchmark mesh",
    "5_converged_FEM_residual": "BLOCKED: isolated FEniCS installed but cardiac probe exits139",
    "6_trusted_solver_assembly": "BLOCKED",
    "7_residual_gradient": "synthetic unit only",
    "8_Jacobians": "unit and manufactured benchmark",
    "9_perturbation": "synthetic unit only",
    "10_cardiac_BC_pressure": "NOT IMPLEMENTED"
  }
}

Cardiac equilibrium release gate incomplete; see solver_probe.md.
