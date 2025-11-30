/*
 * -------------------------------------------
 * Copyright (c) 2021 - 2025 Prashant K. Jha
 * -------------------------------------------
 * https://github.com/CEADpx/multiphysics-peridynamics
 *
 * Distributed under the Boost Software License, Version 1.0. (See accompanying
 * file LICENSE)
 */

#include "thermomechanical_model.h"

#include "heat_source.h"
#include "loading.h"
#include "fracture.h"

#include <memory>
#include <filesystem> // Added for directory creation

namespace {
  bool exodus_init = false;
  bool obs_qoi_init = false;
}

namespace model {

ThermomechanicalModel::ThermomechanicalModel(libMesh::ReplicatedMesh& mesh, 
  libMesh::EquationSystems& equation_systems,
  libMesh::TransientLinearImplicitSystem& temperature_system,
  libMesh::ExplicitSystem& theta_dot_system,
  libMesh::ExplicitSystem& mechanical_system,
  inp::MaterialDeck& deck, 
  double dt)
  : d_mesh(mesh),
    d_equation_systems(equation_systems),
    d_temperature_system(temperature_system),
    d_theta_dot_system(theta_dot_system),
    d_mechanical_system(mechanical_system),
    d_material_p(nullptr),
    d_dt(dt),
    d_time(0.0),
    d_displacement_update_method("velocity_verlet"), 
    d_use_nodal_fem(false)
{
  // Initialize local vectors
  const unsigned int n_nodes = mesh.n_nodes();
  d_displacement.resize(n_nodes);
  d_displacement_old.resize(n_nodes);
  d_velocity.resize(n_nodes);
  d_temperature.resize(n_nodes);
  d_mx.resize(n_nodes);
  d_theta.resize(n_nodes);
  d_theta_old.resize(n_nodes);
  d_theta_dot.resize(n_nodes);
  d_force.resize(n_nodes);
  d_neighbor_list.resize(n_nodes);
  d_neighbor_volume.resize(n_nodes);
  d_displacement_fixed.resize(n_nodes);
  d_force_fixed.resize(n_nodes);
  d_nodal_volume.resize(n_nodes);
  d_damage.resize(n_nodes);

  // Initialize fracture
  d_fracture_p = std::make_unique<geom::Fracture>(*this);

  // Initialize material
  d_material_p = std::make_shared<material::Material>(deck);

  // Initialize MPI communicator
  d_cm_p = std::make_unique<MPICommunicator>(d_mesh);

  // Initialize loading
  d_loading_p = std::make_unique<loading::Loading>(*this);

  // qoi
  d_qoi = {{"force", 0.0}, {"displacement", 0.0}, {"velocity", 0.0}, {"theta", 0.0}, {"theta_dot", 0.0}, {"temperature", 0.0}, {"damage", 0.0}};
}

void ThermomechanicalModel::initialize() {

  // Build neighbor list for all nodes
  setupNeighborList();

  // Setup ghost nodes for parallel computation
  setupGhostNodesAndCommunicator();

  // Compute nodal volume (done once)
  computeNeighborVolume();

  // Compute weighted volume (done once)
  computeMx();

  // Initialize fracture
  d_fracture_p->initialize();
}

void ThermomechanicalModel::secondaryInitialize() {

  // TODO: After initialization, the function creating this class should
  // apply boundary and initial conditions to the displacement and temperature fields
  // and then compute the theta, theta_dot, and fill the temperature data.

  // initialize loading
  d_loading_p->initialize();

  // apply initial conditions
  d_loading_p->applyInitialCondition();

  // apply displacement boundary conditions
  d_loading_p->applyDisplacement(d_time);

  // add cracks
  std::cout << "\n\n\nadding cracks at time = " << d_time << std::endl;
  d_fracture_p->addCrack(d_time);
  std::cout << "\n\n\n" << std::endl;

  // update theta and damage
  updateThetaAndDamage();

  // Assemble heat equation matrix (done once)
  assembleHeatMatrix();

  // copy current to old
  *d_temperature_system.old_local_solution = *d_temperature_system.current_local_solution;

  updateCoupledData();
}

void ThermomechanicalModel::advance() {

  solveHeatEquation();

  updateKinematics();

  updateCoupledData();

  // update time
  d_time += d_dt;
}

void ThermomechanicalModel::solveHeatEquation() {
  assembleHeatRHS(d_time);
  d_temperature_system.solve();

  *d_temperature_system.old_local_solution = *d_temperature_system.current_local_solution;

  // copy temperature from libmesh system
  copyTemperature();

  // sync temperature to ghost nodes
  d_cm_p->syncScalarData(d_temperature);
}

void ThermomechanicalModel::updateKinematics() {

  computeForces();

  if (d_displacement_update_method == "central_difference") {
    double factor = d_dt*d_dt / d_material_p->d_deck.d_rho;
    // Update displacement using central difference
    for (size_t loc_i = 0; loc_i < d_cm_p->d_owned_size; loc_i++) {
      const auto& i = d_cm_p->d_owned_and_ghost_ids[loc_i];  // Global node ID

      // central‐difference displacement update
      const auto& u_nm1 = d_displacement_old[i];
      const auto& u_n   = d_displacement[i];
      const auto& u_np1 = 2*u_n - u_nm1 + factor*d_force[i];

      for (unsigned int dof = 0; dof < 3; dof++) {
        if (util::isDofFree(i, dof, d_displacement_fixed)) {
          d_displacement_old[i](dof) = u_n(dof);
          d_displacement[i](dof)     = u_np1(dof);
          d_velocity[i](dof) = (u_np1(dof) - u_nm1(dof)) / (2.0 * d_dt);
        }
      }
    }

    // update displacement of fixed nodes
    d_loading_p->applyDisplacement(d_time);

  } else if (d_displacement_update_method == "velocity_verlet") {

    double factor = 0.5*d_dt / d_material_p->d_deck.d_rho;
    // first half step
    for (size_t loc_i = 0; loc_i < d_cm_p->d_owned_size; loc_i++) {
      const auto& i = d_cm_p->d_owned_and_ghost_ids[loc_i];  // Global node ID

      for (unsigned int dof = 0; dof < 3; dof++) {
        if (util::isDofFree(i, dof, d_displacement_fixed)) {
          d_velocity[i](dof) = d_velocity[i](dof) + factor * d_force[i](dof);
          d_displacement_old[i](dof) = d_displacement[i](dof);
          d_displacement[i](dof) = d_displacement[i](dof) + d_dt * d_velocity[i](dof);
        }
      }
    }

    // update displacement of fixed nodes
    d_loading_p->applyDisplacement(d_time);

    // sync displacement to ghost nodes
    d_cm_p->syncDisplacement(d_displacement);

    // update theta and damage
    updateThetaAndDamage();

    // sync volumetric strain to ghost nodes
    d_cm_p->syncScalarData(d_theta);

    // compute force at updated displacement
    computeForces();

    // second half step
    for (size_t loc_i = 0; loc_i < d_cm_p->d_owned_size; loc_i++) {
      const auto& i = d_cm_p->d_owned_and_ghost_ids[loc_i];  // Global node ID

      for (unsigned int dof = 0; dof < 3; dof++) {
        if (util::isDofFree(i, dof, d_displacement_fixed)) {
          d_velocity[i](dof) = d_velocity[i](dof) + factor * d_force[i](dof);
        }
      }
    }
  } else {
    throw std::runtime_error("Invalid displacement update method: " + d_displacement_update_method);
  }

  // update displacement of fixed nodes
  d_loading_p->applyDisplacement(d_time);
}

void ThermomechanicalModel::updateCoupledData() {

  // sync displacement to ghost nodes (no need for velocity verlet as it is done in updateKinematics)
  if (d_displacement_update_method == "central_difference") {

    d_cm_p->syncDisplacement(d_displacement);

    // update volumetric strain (needs updated displacement)
    updateThetaAndDamage();

    // sync volumetric strain to ghost nodes
    d_cm_p->syncScalarData(d_theta);
  }

  // update theta_old and theta_dot (needed by the heat equation)
  // we loop over owned + ghost nodes (we have updated values of ghost nodes after sync above)
  for (const auto& i: d_cm_p->d_owned_and_ghost_ids) {
    d_theta_dot[i] = (d_theta[i] - d_theta_old[i]) / d_dt;
    d_theta_old[i] = d_theta[i];
  }

  // update theta_dot in libmesh system
  copyThetaDot();
}

void ThermomechanicalModel::setupNeighborList() {

  d_neighbor_list.resize(d_mesh.n_nodes());
  d_neighbor_volume.resize(d_mesh.n_nodes());
  
  const unsigned int n_nodes = d_mesh.n_nodes();
  
  // loop over only nodes we own
  for (const auto& node : d_mesh.local_node_ptr_range()) {
    const auto i = node->id();  // Global node ID
    if (i > n_nodes) {
      printf("i = %d, n_nodes = %d\n", i, n_nodes);
      throw std::runtime_error("i > n_nodes");
    }
    const libMesh::Point& xi = *d_mesh.node_ptr(i);
    d_neighbor_list[i].clear();
    d_neighbor_volume[i].clear();
    
    for (libMesh::dof_id_type j = 0; j < n_nodes; j++) {
      if (i == j) continue;
      const libMesh::Point& xj = *d_mesh.node_ptr(j);
      double dist = (xj - xi).norm();
      if (dist <= d_material_p->d_deck.d_horizon) {
        d_neighbor_list[i].push_back(j);
        auto jj = d_neighbor_list[i][d_neighbor_list[i].size() - 1];
        if (jj > d_mesh.max_node_id()) {
          printf("j = %u, jj = %u, max_node_id = %u\n", j, jj, d_mesh.max_node_id());
          throw std::runtime_error("jj > max_node_id");
        }
        d_neighbor_volume[i].push_back(0.0);
      }
    }
  }
}

void ThermomechanicalModel::setupGhostNodesAndCommunicator() {
  // Clear existing ghost node data
  d_cm_p->d_ghost_ids.clear();
  d_cm_p->d_ghost_ids.resize(d_mesh.n_nodes());
  std::set<libMesh::dof_id_type> ghost_set_all;

  d_cm_p->d_owned_and_ghost_ids.clear();

  // Get processor ID
  const unsigned int proc_id = d_cm_p->d_rank;

  // Loop over nodes owned by this processor
  for (const auto& node : d_mesh.local_node_ptr_range()) {
    const auto i = node->id();  // Global node ID
    d_cm_p->d_owned_and_ghost_ids.push_back(i);

    std::set<libMesh::dof_id_type> ghost_set;
    
    // Loop over neighbors of owned node using global IDs from neighbor list
    for (const auto& j : d_neighbor_list[i]) {
      // If neighbor node is not owned by this processor, it's a ghost node
      if (d_mesh.node_ptr(j)->processor_id() != proc_id) {
        ghost_set.insert(j);
        ghost_set_all.insert(j);
      }
    }

    d_cm_p->d_ghost_ids[i].assign(ghost_set.begin(), ghost_set.end());
  }

  d_cm_p->d_owned_size = d_cm_p->d_owned_and_ghost_ids.size();

  // now we need to add ghost nodes to the owned_and_ghost_ids vector
  d_cm_p->d_owned_and_ghost_ids.insert(
    d_cm_p->d_owned_and_ghost_ids.end(),
      ghost_set_all.begin(),
      ghost_set_all.end());

  // initialize MPI communicator
  d_cm_p->initCommunication();

}


void ThermomechanicalModel::computeNeighborVolume() {
  
  // we need node to element connectivity
  std::unordered_map< libMesh::dof_id_type, std::vector< libMesh::dof_id_type >> nodes_to_elem_map;
  libMesh::MeshTools::build_nodes_to_elem_map(d_mesh, nodes_to_elem_map);

  if (d_use_nodal_fem) {

    // Element objects
    const libMesh::DofMap& dof_map = d_temperature_system.get_dof_map();
    libMesh::FEType fe_type = dof_map.variable_type(0);
    libMesh::QGauss qrule (d_mesh.mesh_dimension(), fe_type.default_quadrature_order());
    std::unique_ptr<libMesh::FEBase> fe(libMesh::FEBase::build(d_mesh.mesh_dimension(), fe_type));
    
    fe->attach_quadrature_rule(&qrule);

    // Element data
    const std::vector<libMesh::Real>& JxW = fe->get_JxW();
    const std::vector<std::vector<libMesh::Real>>& phi = fe->get_phi();
    const auto& xyz = fe->get_xyz();

    const double& horizon = d_material_p->d_deck.d_horizon;
    
    // Loop over all nodes
    for (size_t loc_i = 0; loc_i < d_cm_p->d_owned_size; loc_i++) {
      const auto& i = d_cm_p->d_owned_and_ghost_ids[loc_i];  // Global node ID
      const libMesh::Point& xi = *d_mesh.node_ptr(i);

      // Loop over nodes in the neighborhood of node i
      for (size_t loc_j = 0; loc_j < d_neighbor_list[i].size(); loc_j++) {
        const auto& j = d_neighbor_list[i][loc_j];
        const libMesh::Point& xj = *d_mesh.node_ptr(j);
        double Vij = 0.0;

        // loop over elements with node j as a vertex
        for (const auto& elem : nodes_to_elem_map[j]) {
          const libMesh::Elem* elem_ptr = d_mesh.elem_ptr(elem);
          double elem_volume = elem_ptr->volume();
          double elem_length_est = std::pow(elem_volume, 1.0/d_material_p->d_deck.d_dim);

          // we want to compute the integral of influence function times the shape function of node j over the element domain
          // we can do this by quadrature

          fe->reinit(elem_ptr);

          // identify local node id of node j in the element
          libMesh::dof_id_type local_node_j_id = 0;
          // iterate over nodes in the element
          for (unsigned int n = 0; n < elem_ptr->n_nodes(); n++) {
            if (elem_ptr->node_id(n) == j) {
              local_node_j_id = n;
              break;
            }
          }

          for (unsigned int qp = 0; qp < qrule.n_points(); qp++) {
            const auto& xq = xyz[qp] - xi;
            const auto& JxW_qp = JxW[qp];
            const auto& phi_j_qp = phi[local_node_j_id][qp];

            auto r = xq.norm();
            if (r <= horizon) {
              double correction_factor = 1.0;
              if (r <= horizon && r > horizon - elem_length_est) {
                correction_factor = (horizon - r)/elem_length_est;
              }

              const auto& infFn_r_qp = d_material_p->getInfFn(r);
              Vij += JxW_qp * infFn_r_qp * phi_j_qp * correction_factor;
            }
          }
        } // element loop

        d_neighbor_volume[i][loc_j] = Vij;
      }
    } // owned node loop

  } else {

    double elem_volume_fraction = 1 / std::pow(2, d_mesh.mesh_dimension());

    // loop over owned nodes
    for (size_t loc_i = 0; loc_i < d_cm_p->d_owned_size; loc_i++) {
      const auto& i = d_cm_p->d_owned_and_ghost_ids[loc_i];  // Global node ID
      const libMesh::Point& xi = *d_mesh.node_ptr(i);

      double Vi = 0.0;

      // we loop over all elements with node i as a vertex
      for (const auto& elem : nodes_to_elem_map[i]) {
        const libMesh::Elem* elem_ptr = d_mesh.elem_ptr(elem);
        double elem_volume = elem_ptr->volume();
        Vi += elem_volume*elem_volume_fraction;
      }

      d_nodal_volume[i] = Vi;
    }

    for (size_t loc_i = 0; loc_i < d_cm_p->d_owned_size; loc_i++) {
      const auto& i = d_cm_p->d_owned_and_ghost_ids[loc_i];  // Global node ID
      const libMesh::Point& xi = *d_mesh.node_ptr(i);

      // loop over neighbors to compute Vij
      for (size_t loc_j = 0; loc_j < d_neighbor_list[i].size(); loc_j++) {
        const auto& j = d_neighbor_list[i][loc_j];
        const libMesh::Point& xj = *d_mesh.node_ptr(j);
        auto Vj = d_nodal_volume[j];
        
        const double& elem_length_est = std::pow(Vj, 1.0/d_material_p->d_deck.d_dim);

        // compute distance between i and j
        libMesh::Point dx = xj - xi;
        double r = dx.norm();

        // upper and lower bound for volume correction
        auto check_up = d_material_p->d_deck.d_horizon + 0.5 * elem_length_est;
        auto check_low = d_material_p->d_deck.d_horizon - 0.5 * elem_length_est;

        if (r <= d_material_p->d_deck.d_horizon) {
          if (r >= check_up) {
            Vj *= (check_up - r) / elem_length_est;
          }
        }
        
        d_neighbor_volume[i][loc_j] = Vj;
      }
    } // owned node loop
  } // if !d_use_nodal_fem

  bool debug = false;
  if (debug) {
    if (d_cm_p->d_rank == 0) {
      std::cout << "Debugging neighbor volume" << std::endl;
      for (size_t loc_i = 0; loc_i < d_cm_p->d_owned_size; loc_i++) {
        const auto& i = d_cm_p->d_owned_and_ghost_ids[loc_i];
        const auto& i_neigh_list = d_neighbor_list[i];
        const auto& i_neigh_vol = d_neighbor_volume[i];

        // get max and min, and sum of the i_neigh_vol (use stl algorithms)
        double max_vol = *std::max_element(i_neigh_vol.begin(), i_neigh_vol.end());
        double min_vol = *std::min_element(i_neigh_vol.begin(), i_neigh_vol.end());
        double sum_vol = std::accumulate(i_neigh_vol.begin(), i_neigh_vol.end(), 0.0);
        double neigh_vol = d_material_p->d_horizonVolume;

        std::cout << "i = " << i << ", max_vol = " << max_vol 
                  << ", min_vol = " << min_vol 
                  << ", sum_vol = " << sum_vol 
                  << ", neigh_vol = " << neigh_vol 
                  << std::endl;
      }
    }
  }
}

void ThermomechanicalModel::computeMx() {

  // Loop over owned nodes
  for (size_t loc_i = 0; loc_i < d_cm_p->d_owned_size; loc_i++) {
    const auto& i = d_cm_p->d_owned_and_ghost_ids[loc_i];  // Global node ID
    const libMesh::Point& xi = *d_mesh.node_ptr(i);
    double mx = 0.0;

    // Loop over neighbors using neighbor list
    for (size_t loc_j = 0; loc_j < d_neighbor_list[i].size(); loc_j++) {
      const auto& j = d_neighbor_list[i][loc_j];
      const libMesh::Point& xj = *d_mesh.node_ptr(j);
      
      // Compute reference bond vector
      libMesh::Point dx = xj - xi;
      double r = dx.norm();

      // influence function
      auto J = d_material_p->getInfFn(r);
      if (d_use_nodal_fem) 
        J = 1.0;
      
      // Add contribution to weighted volume
      mx +=  J * r * r * d_neighbor_volume[i][loc_j];
    }

    d_mx[i] = mx;
  }

  // Synchronize weighted volume across processors
  if (d_cm_p) {
    d_cm_p->syncScalarData(d_mx);
  }
}

void ThermomechanicalModel::updateThetaAndDamage() {
  // Loop over owned nodes
  for (size_t loc_i = 0; loc_i < d_cm_p->d_owned_size; loc_i++) {
    const auto& i = d_cm_p->d_owned_and_ghost_ids[loc_i];  // Global node ID
    const libMesh::Point& xi = *d_mesh.node_ptr(i);
    double theta = 0.0;
    double a = double(d_neighbor_list[i].size());
    double b = 0.0;

    // Loop over neighbors using neighbor list
    for (size_t loc_j = 0; loc_j < d_neighbor_list[i].size(); loc_j++) {
      const auto& j = d_neighbor_list[i][loc_j];
      const libMesh::Point& xj = *d_mesh.node_ptr(j);
      
      // Compute reference and current bond vectors
      libMesh::Point dx = xj - xi;
      libMesh::Point du = d_displacement[j] - d_displacement[i];
      double r = dx.norm();

      // influence function
      auto J = d_material_p->getInfFn(r);
      if (d_use_nodal_fem) 
        J = 1.0;
      
      // Compute bond strain (s = (|eta| - |xi|) / |xi|)
      double s = d_material_p->getS(dx, du);

      // get bond state
      bool bond_state = d_fracture_p->getBondState(i, loc_j);
      if (s > d_material_p->d_s0 && d_material_p->d_deck.d_breakBonds && !bond_state) {
        d_fracture_p->setBondState(i, loc_j, true);
      }
      
      // Add contribution to volumetric strain
      if (!bond_state) {
        theta += J * r * r * s * d_neighbor_volume[i][loc_j];
      }

      if (!bond_state) {
        b += 1.;
      }
    }

    d_theta[i] = d_material_p->d_deck.d_dim * theta / d_mx[i];
    d_damage[i] = 1. - b/a;
  }

  for (size_t loc_i = d_cm_p->d_owned_size; loc_i < d_cm_p->d_owned_and_ghost_ids.size(); loc_i++) {
    const auto& i = d_cm_p->d_owned_and_ghost_ids[loc_i];  // Global node ID
    d_theta[i] = 0.0;
  }
}
  
void ThermomechanicalModel::copyTemperature() {
  // Use the libMesh DofMap to access the solution vector and copy values to d_temperature
  const libMesh::DofMap& dof_map = d_temperature_system.get_dof_map();
  const libMesh::NumericVector<double>& solution = *d_temperature_system.solution;

  std::vector<libMesh::dof_id_type> dof_indices;

  // Loop over owned nodes
  for (size_t loc_i = 0; loc_i < d_cm_p->d_owned_size; loc_i++) {
    const auto& i = d_cm_p->d_owned_and_ghost_ids[loc_i];  // Global node ID
    const libMesh::Node *node = d_mesh.node_ptr(i);

    // Get dof index for temperature variable (assume variable 0 is "temp")
    dof_map.dof_indices(node, dof_indices, 0); // 0 for "temp" variable

    if (!dof_indices.empty()) {
      d_temperature[i] = solution(dof_indices[0]);
    }
  }

  for (size_t loc_i = d_cm_p->d_owned_size; loc_i < d_cm_p->d_owned_and_ghost_ids.size(); loc_i++) {
    const auto& i = d_cm_p->d_owned_and_ghost_ids[loc_i];  // Global node ID
    d_temperature[i] = 0.0;
  }
}

void ThermomechanicalModel::copyThetaDot() {
  // Use the libMesh DofMap to access the solution vector and update theta_dot values in the libMesh system
  const libMesh::DofMap& dof_map = d_theta_dot_system.get_dof_map();
  libMesh::NumericVector<double>& solution = *d_theta_dot_system.solution;

  std::vector<libMesh::dof_id_type> dof_indices;

  // Loop over owned nodes
  for (size_t loc_i = 0; loc_i < d_cm_p->d_owned_size; ++loc_i) {
    const auto& i = d_cm_p->d_owned_and_ghost_ids[loc_i];  // Global node ID
    const libMesh::Node *node = d_mesh.node_ptr(i);

    // Get dof index for theta_dot variable (assume variable 0 is "theta_dot")
    dof_map.dof_indices(node, dof_indices, 0); // 0 for "theta_dot" variable

    if (!dof_indices.empty()) {
      solution.set(dof_indices[0], d_theta_dot[i]);
    }
  }
  solution.close();
}

void ThermomechanicalModel::assembleHeatMatrix() {
  // Get required objects
  const libMesh::DofMap& dof_map = d_temperature_system.get_dof_map();
  libMesh::SparseMatrix<double>& matrix = *d_temperature_system.matrix;
  
  // Element objects
  libMesh::FEType fe_type = dof_map.variable_type(0);
  libMesh::QGauss qrule (d_mesh.mesh_dimension(), fe_type.default_quadrature_order());
  std::unique_ptr<libMesh::FEBase> fe(libMesh::FEBase::build(d_mesh.mesh_dimension(), fe_type));
  
  fe->attach_quadrature_rule(&qrule);

  // Element data
  const std::vector<libMesh::Real>& JxW = fe->get_JxW();
  const std::vector<std::vector<libMesh::Real>>& phi = fe->get_phi();
  const std::vector<std::vector<libMesh::RealGradient>>& dphi = fe->get_dphi();

  // Element matrix
  libMesh::DenseMatrix<double> Ke;
  std::vector<libMesh::dof_id_type> dof_indices;
  

  // Loop over elements
  libMesh::MeshBase::const_element_iterator el = d_mesh.active_local_elements_begin();
  const libMesh::MeshBase::const_element_iterator end_el = d_mesh.active_local_elements_end();

  for (; el != end_el; ++el) {
    const libMesh::Elem* elem = *el;
    
    dof_map.dof_indices(elem, dof_indices);
    const auto n_dofs = dof_indices.size();
    
    Ke.resize(n_dofs, n_dofs);
    fe->reinit(elem);

    // Loop over quadrature points
    for (unsigned int qp = 0; qp < qrule.n_points(); qp++) {
      
      // Loop over test functions
      for (unsigned int i = 0; i < n_dofs; i++) {
        // Loop over trial functions
        for (unsigned int j = 0; j < n_dofs; j++) {
          // Add diffusion term: k * grad(v) . grad(u)
          Ke(i,j) += d_dt * JxW[qp] * d_material_p->d_deck.d_Ktherm * (dphi[i][qp] * dphi[j][qp]);
          
          // Add mass term: rho * c * v * u
          Ke(i,j) += JxW[qp] * d_material_p->d_deck.d_rho * 
                     d_material_p->d_deck.d_Cv * phi[i][qp] * phi[j][qp];
        }
      }
    }

    matrix.add_matrix(Ke, dof_indices);
  }

  if (!d_material_p->d_deck.d_robinBC) {
    matrix.close();
    return;
  }

  // --- Robin boundary condition: k * grad(T)·n + h_conv*(T - T_inf) = 0 ---> adds h_conv*phi_i*phi_j on boundary
  const auto& h_conv = d_material_p->d_deck.d_hconvect;

  // Setup boundary FE
  auto fe_face = libMesh::FEBase::build(d_mesh.mesh_dimension(), fe_type);
  libMesh::QGauss face_qrule(d_mesh.mesh_dimension()-1, fe_type.default_quadrature_order());
  fe_face->attach_quadrature_rule(&face_qrule);
  const auto& JxW_face  = fe_face->get_JxW();
  const auto& phi_face  = fe_face->get_phi();

  // Loop over boundary sides
  for (auto el_it = d_mesh.active_local_elements_begin(); el_it != d_mesh.active_local_elements_end(); ++el_it)
  {
    const libMesh::Elem* elem = *el_it;
    dof_map.dof_indices(elem, dof_indices);
    const auto n_dofs = dof_indices.size();

    for (unsigned side = 0; side < elem->n_sides(); ++side)
    {
      if (elem->neighbor_ptr(side) == nullptr) // exterior face
      {
        fe_face->reinit(elem, side);
        Ke.resize(n_dofs, n_dofs);
        for (unsigned qp = 0; qp < face_qrule.n_points(); ++qp)
          for (unsigned i = 0; i < n_dofs; ++i)
            for (unsigned j = 0; j < n_dofs; ++j)
            {
              Ke(i,j) += d_dt * h_conv * phi_face[i][qp] * phi_face[j][qp] * JxW_face[qp];
            }
        matrix.add_matrix(Ke, dof_indices);
      }
    }
  }

  matrix.close();
}

void ThermomechanicalModel::assembleHeatRHS(const double& time) {
  // Get required objects
  const libMesh::DofMap& dof_map = d_temperature_system.get_dof_map();
  libMesh::NumericVector<double>& rhs = *d_temperature_system.rhs;
  rhs.zero();

  const libMesh::DofMap& dof_map_theta_dot = d_theta_dot_system.get_dof_map();
  
  // Element objects
  libMesh::FEType fe_type = dof_map.variable_type(0);
  libMesh::QGauss qrule (d_mesh.mesh_dimension(), fe_type.default_quadrature_order());
  std::unique_ptr<libMesh::FEBase> fe(libMesh::FEBase::build(d_mesh.mesh_dimension(), fe_type));
  
  fe->attach_quadrature_rule(&qrule);

  // Element data
  const std::vector<libMesh::Real>& JxW = fe->get_JxW();
  const std::vector<std::vector<libMesh::Real>>& phi = fe->get_phi();
  const auto& xyz = fe->get_xyz();

  // Element vector
  libMesh::DenseVector<double> Fe;
  std::vector<libMesh::dof_id_type> dof_indices;
  std::vector<libMesh::dof_id_type> dof_indices_theta_dot;

  double Told = 0.;
  double theta_dot = 0.;
  bool mechToThermCoupling = d_material_p->d_deck.d_mechToThermCoupling;

  // Loop over elements
  libMesh::MeshBase::const_element_iterator el = d_mesh.active_local_elements_begin();
  const libMesh::MeshBase::const_element_iterator end_el = d_mesh.active_local_elements_end();

  for (; el != end_el; ++el) {
    const libMesh::Elem* elem = *el;
    
    dof_map.dof_indices(elem, dof_indices);
    dof_map_theta_dot.dof_indices(elem, dof_indices_theta_dot);
    const auto n_dofs = dof_indices.size();
    
    Fe.resize(n_dofs);
    fe->reinit(elem);

    // Loop over quadrature points
    for (unsigned int qp = 0; qp < qrule.n_points(); qp++) {

      Told = 0.;
      theta_dot = 0.;
      for (unsigned int l = 0; l < phi.size(); l++) {
        Told += phi[l][qp] * d_temperature_system.old_solution(dof_indices[l]);
        theta_dot += phi[l][qp] * d_theta_dot_system.current_solution(dof_indices_theta_dot[l]);
      }

      // Loop over test functions
      for (unsigned int i = 0; i < n_dofs; i++) {
        // Add source terms
        Fe(i) += d_dt * JxW[qp] * d_material_p->d_deck.d_rho * phi[i][qp] * d_heat_sources_p->get(xyz[qp], time);

        // add previous time step solution
        Fe(i) += JxW[qp] * d_material_p->d_deck.d_rho * 
        d_material_p->d_deck.d_Cv * phi[i][qp] * Told;

        // Add term due to theta_dot (explicit)
        if (mechToThermCoupling)
          Fe(i) += d_dt * JxW[qp] * d_material_p->d_thetaDotFactor * theta_dot * Told * phi[i][qp];
      }
    }

    rhs.add_vector(Fe, dof_indices);
  }

  if (!d_material_p->d_deck.d_robinBC) {
    rhs.close();
    return;
  }

  // --- Robin BC RHS term: ∫_Γ h_conv * T_inf * φ_i dΓ
  const auto& h_conv = d_material_p->d_deck.d_hconvect;
  const auto& T_ref  = d_material_p->d_deck.d_Tref;

  auto fe_face = libMesh::FEBase::build(d_mesh.mesh_dimension(), fe_type);
  libMesh::QGauss face_qrule(d_mesh.mesh_dimension()-1, fe_type.default_quadrature_order());
  fe_face->attach_quadrature_rule(&face_qrule);
  const auto& JxW_face  = fe_face->get_JxW();
  const auto& phi_face  = fe_face->get_phi();

  for (auto el_it = d_mesh.active_local_elements_begin(); el_it != d_mesh.active_local_elements_end(); ++el_it)
  {
    const libMesh::Elem* elem = *el_it;
    dof_map.dof_indices(elem, dof_indices);
    const auto n_dofs = dof_indices.size();
    for (unsigned side=0; side<elem->n_sides(); ++side)
    {
      if (elem->neighbor_ptr(side) == nullptr) // exterior
      {
        fe_face->reinit(elem, side);
        Fe.resize(n_dofs);
        for (unsigned qp=0; qp<face_qrule.n_points(); ++qp)
          for (unsigned i=0; i<n_dofs; ++i)
            Fe(i) += d_dt * h_conv * T_ref * phi_face[i][qp] * JxW_face[qp];
        
        rhs.add_vector(Fe, dof_indices);
      }
    }
  }

  rhs.close();
}

void ThermomechanicalModel::computeForces() {
  for (size_t loc_i = 0; loc_i < d_cm_p->d_owned_size; loc_i++) {
    d_force[d_cm_p->d_owned_and_ghost_ids[loc_i]] = libMesh::Point(0.0, 0.0, 0.0);
  }
  computePeriForces();
  d_loading_p->applyForce(d_time);
}

void ThermomechanicalModel::computePeriForces() {
  
  // Loop over owned nodes
  for (size_t loc_i = 0; loc_i < d_cm_p->d_owned_size; loc_i++) {
    const auto& i = d_cm_p->d_owned_and_ghost_ids[loc_i];  // Global node ID
    const libMesh::Point& xi = *d_mesh.node_ptr(i);

    libMesh::Point force_i(0.0, 0.0, 0.0);

    // Loop over neighbors using neighbor list
    for (size_t loc_j = 0; loc_j < d_neighbor_list[i].size(); loc_j++) {
      libMesh::dof_id_type j = d_neighbor_list[i][loc_j];
      const libMesh::Point& xj = *d_mesh.node_ptr(j);
      
      // Compute reference and current bond vectors
      libMesh::Point dx = xj - xi;
      libMesh::Point du = d_displacement[j] - d_displacement[i];
      double r = dx.norm();
      
      // Get bond state (true = broken bond)
      bool fs = d_fracture_p->getBondState(i, loc_j);

      // Compute bond strain
      double s = d_material_p->getS(dx, du);
      
      // Get force magnitude
      auto f_i = d_material_p->getBondForce(r, s, fs, d_mx[i], d_theta[i], d_temperature[i], d_use_nodal_fem);
      auto f_j = d_material_p->getBondForce(r, s, fs, d_mx[j], d_theta[j], d_temperature[j], d_use_nodal_fem);
      
      // Add force contribution
      force_i += (f_i + f_j) * d_neighbor_volume[i][loc_j] * d_material_p->getBondForceDirection(dx, du);
    }

    d_force[i] = force_i;
  }
}

void ThermomechanicalModel::syncGhostData() {

  // Synchronize displacement data
  d_cm_p->syncDisplacement(d_displacement);

  // Synchronize temperature data
  d_cm_p->syncScalarData(d_temperature);

  // Synchronize volumetric strain data
  d_cm_p->syncScalarData(d_theta);
}

void ThermomechanicalModel::updateMechanicalSystem() {

  // we need to update the equation systems with current mechanical data
  // Access the DofMap for the mechanical system
  const libMesh::DofMap& dof_map = d_mechanical_system.get_dof_map();

  // Get references to the solution vector
  libMesh::NumericVector<double>& solution = *d_mechanical_system.solution;

  // Variable numbers for mechanical variables
  const unsigned int ux_var = d_mechanical_system.variable_number("ux");
  const unsigned int uy_var = d_mechanical_system.variable_number("uy");
  unsigned int uz_var = libMesh::invalid_uint;
  if (d_mesh.mesh_dimension() == 3)
    uz_var = d_mechanical_system.variable_number("uz");

  const unsigned int vx_var = d_mechanical_system.variable_number("vx");
  const unsigned int vy_var = d_mechanical_system.variable_number("vy");
  unsigned int vz_var = libMesh::invalid_uint;
  if (d_mesh.mesh_dimension() == 3)
    vz_var = d_mechanical_system.variable_number("vz");

  const unsigned int fx_var = d_mechanical_system.variable_number("fx");
  const unsigned int fy_var = d_mechanical_system.variable_number("fy");
  unsigned int fz_var = libMesh::invalid_uint;
  if (d_mesh.mesh_dimension() == 3)
    fz_var = d_mechanical_system.variable_number("fz");

  const unsigned int theta_var = d_mechanical_system.variable_number("theta");
  const unsigned int damage_var = d_mechanical_system.variable_number("damage");

  std::vector<libMesh::dof_id_type> dof_indices;

  // Loop over owned nodes and set the values in the solution vector
  for (size_t loc_i = 0; loc_i < d_cm_p->d_owned_size; loc_i++) {
    const auto& i = d_cm_p->d_owned_and_ghost_ids[loc_i];  // Global node ID

    // Get dof indices for this node
    dof_map.dof_indices(d_mesh.node_ptr(i), dof_indices);

    // Set displacement
    solution.set(dof_indices[ux_var], d_displacement[i](0));
    solution.set(dof_indices[uy_var], d_displacement[i](1));
    if (d_mesh.mesh_dimension() == 3)
      solution.set(dof_indices[uz_var], d_displacement[i](2));

    // Set velocity
    solution.set(dof_indices[vx_var], d_velocity[i](0));
    solution.set(dof_indices[vy_var], d_velocity[i](1));
    if (d_mesh.mesh_dimension() == 3)
      solution.set(dof_indices[vz_var], d_velocity[i](2));

    // Set force
    solution.set(dof_indices[fx_var], d_force[i](0));
    solution.set(dof_indices[fy_var], d_force[i](1));
    if (d_mesh.mesh_dimension() == 3)
      solution.set(dof_indices[fz_var], d_force[i](2));

    // Set theta
    solution.set(dof_indices[theta_var], d_theta[i]);

    // Set damage
    solution.set(dof_indices[damage_var], d_damage[i]);
  }

  // Close the solution vector to finalize changes
  solution.close();
}

void ThermomechanicalModel::write(unsigned int file_number, bool print_debug) {

  // update the mechanical system (only when we write)
  updateMechanicalSystem();

  // update the observation data (only when we write)
  updateObsData();

  // update the qoi
  updateQoi();

  // write the exodus file
  if (!exodus_init) {
    libMesh::ExodusII_IO(d_mesh).write_equation_systems ("simulation.e", d_equation_systems);
    exodus_init = true;
  } else {
    libMesh::ExodusII_IO exo(d_mesh);
    exo.append(true);
    exo.write_timestep("simulation.e", d_equation_systems, file_number, d_time);
  }

  // append the observation data to a text file (time, node_id, T, ux, uy, uz)
  if (d_cm_p->d_rank == 0) {
    std::ofstream obs_file, qoi_file;
    if (!obs_qoi_init) {
      obs_file.open("obs_data.csv", std::ios::out);
      qoi_file.open("qoi.csv", std::ios::out);
    } else {
      obs_file.open("obs_data.csv", std::ios::app);
      qoi_file.open("qoi.csv", std::ios::app);
    }


    std::string delim = ",";
    if (!obs_qoi_init) {
      // write the header
      qoi_file << "time";
      for (const auto& [key, value]: d_qoi) {
        qoi_file << delim << key;
      }
      qoi_file << std::endl;

      // write the header for the observation data
      obs_file << "time" << delim << "id" << delim << "T" << delim << "ux" << delim << "uy" << delim << "uz";
      obs_file << std::endl;
      obs_qoi_init = true;
    }

    // write the observation data
    for (size_t i = 0; i < d_obs_nodes.size(); ++i) {
      obs_file << d_time << delim << d_obs_nodes.at(i) << delim << d_obs_T[i];
      for (size_t j = 0; j < 3; ++j) {
        obs_file << delim << d_obs_u[j][i];
      }
    }
    obs_file << std::endl;
    obs_file.close();

    // append the qoi to a text file
    qoi_file << d_time;
    for (const auto& [key, value]: d_qoi) {
      qoi_file << delim << value;
    }
    qoi_file << std::endl;
    qoi_file.close();
  }

  if (print_debug) {  

    if (d_cm_p->d_rank == 0) {
      std::cout << "Output number " << file_number << ", time = " << d_time << std::endl;
      for (size_t i = 0; i < d_obs_nodes.size(); ++i) {
        std::cout << "Obs point " << i << ", T = " << d_obs_T[i] << ", damage = " << d_obs_damage[i] << ", u = " << d_obs_u[0][i] << ", " << d_obs_u[1][i] << ", " << d_obs_u[2][i] << std::endl;
      }

      for (const auto& [key, value]: d_qoi) {
        printf("%s = %f\n", key.c_str(), value);
      }
    }
  }
}


void ThermomechanicalModel::setObservationPoints(const std::vector<libMesh::Point> &obs_points) {
  auto n_obs = obs_points.size();
  d_obs_points.clear();
  d_obs_nodes.clear();
  d_obs_points_owned.clear();
  d_obs_T.clear();
  d_obs_u.clear(); 
  d_obs_damage.clear();
  
  // find the nodes closest to the observation points
  std::vector<double> obs_dist(n_obs, 1e10);
  std::vector<libMesh::dof_id_type> obs_node(n_obs, 0);
  
  // brute force search for the closest node to the observation points
  for (libMesh::dof_id_type i = 0; i < d_mesh.n_nodes(); ++i) {
    const libMesh::Point& pt = *d_mesh.node_ptr(i);
    for (size_t j = 0; j < n_obs; ++j) {
      double dist = (pt - obs_points[j]).norm();
      if (dist < obs_dist[j]) {
        obs_dist[j] = dist;
        obs_node[j] = i;
      }
    }
  }
  
  // add the nodes to the set of observation nodes
  for (size_t loc_j = 0; loc_j < obs_node.size(); ++loc_j) {
    const auto& j = obs_node[loc_j];
    // verify that node id is valid
    if (j > d_mesh.max_node_id()) {
      printf("obs_node[%zu] = %d, max_node_id = %d\n", loc_j, j, d_mesh.max_node_id());
      throw std::runtime_error("obs_node[loc_j] > max_node_id");
    }
    if (util::addUnique(d_obs_nodes, j)) {
      d_obs_points.push_back(obs_points[loc_j]);
      d_obs_points_owned.push_back(d_mesh.node_ptr(j)->processor_id() == d_cm_p->d_rank);
    }
  }

  // set other data
  n_obs = d_obs_nodes.size();
  d_obs_T.resize(n_obs, 0.);
  d_obs_damage.resize(n_obs, 0.);
  d_obs_u.resize(3);
  for (size_t i = 0; i < 3; ++i) {
    d_obs_u[i].resize(n_obs, 0.);
  }

  updateObsData();

  // verify if only one rank owns each observation point
  bool debug = false;
  if (debug) {
    std::vector<int> obs_points_owned_count(n_obs, 0);
    for (size_t i = 0; i < n_obs; ++i) {
      obs_points_owned_count[i] = d_obs_points_owned[i] ? 1 : 0;
    }
    std::vector<int> obs_points_owned_count_sum(n_obs, 0);
    MPI_Allreduce(obs_points_owned_count.data(), obs_points_owned_count_sum.data(), n_obs, MPI_INT, MPI_SUM, d_cm_p->d_comm.get());
    for (size_t i = 0; i < n_obs; ++i) {
      if (d_cm_p->d_rank == 0) {
        printf("obs_points_owned_count_sum[%zu] = %d, rank = %d\n", i, obs_points_owned_count_sum[i], d_cm_p->d_rank);
      }
      if (obs_points_owned_count_sum[i] != 1) {
        throw std::runtime_error("obs_points_owned_count_sum[i] != 1");
      }
    }
    std::cout << "observation points set successfully" << std::endl;
  }
}

void ThermomechanicalModel::updateObsData() {

  std::vector<double> local_obs_T(d_obs_nodes.size(), 0.0);
  std::vector<double> local_obs_damage(d_obs_nodes.size(), 0.0);
  std::vector<std::vector<double>> local_obs_u(3, std::vector<double>(d_obs_nodes.size(), 0.0));

  for (size_t i = 0; i < d_obs_nodes.size(); ++i) {
    const auto& i_node = d_obs_nodes[i];
    if (d_obs_points_owned[i]) {
      local_obs_T[i] = d_temperature[i_node];
      local_obs_damage[i] = d_damage[i_node];
      for (size_t j = 0; j < 3; ++j) {
        local_obs_u[j][i] = d_displacement[i_node](j);
      }
    } else {
      local_obs_T[i] = 0.;
      local_obs_damage[i] = 0.;
      for (size_t j = 0; j < 3; ++j) {
        local_obs_u[j][i] = 0.;
      }
    }

    // reset the values 
    d_obs_T[i] = 0.;
    d_obs_damage[i] = 0.;
    for (size_t j = 0; j < 3; ++j) {
      d_obs_u[j][i] = 0.;
    }
  }

  // Sum all vectors and store result in processor 0
  MPI_Reduce(
      local_obs_T.data(),          // send buffer
      d_obs_T.data(),         // receive buffer at root
      d_obs_nodes.size(),                  // number of elements
      MPI_DOUBLE,                // data type
      MPI_SUM,                   // operation
      0,                         // root rank
      d_cm_p->d_comm.get()             // communicator
  );

  MPI_Reduce(
    local_obs_damage.data(),          // send buffer
    d_obs_damage.data(),         // receive buffer at root
    d_obs_nodes.size(),                  // number of elements
    MPI_DOUBLE,                // data type
    MPI_SUM,                   // operation
    0,                         // root rank
    d_cm_p->d_comm.get()             // communicator
  );

  for (size_t i = 0; i < 3; ++i) {
    MPI_Reduce(
      local_obs_u[i].data(),          // send buffer
      d_obs_u[i].data(),         // receive buffer at root
      d_obs_nodes.size(),                  // number of elements
      MPI_DOUBLE,                // data type
      MPI_SUM,                   // operation
      0,                         // root rank
      d_cm_p->d_comm.get()             // communicator
    );
  } 
}

void ThermomechanicalModel::updateQoi() {

  d_qoi["force"] = vectorNorm(d_force);
  d_qoi["displacement"] = vectorNorm(d_displacement);
  d_qoi["velocity"] = vectorNorm(d_velocity);
  d_qoi["theta"] = vectorNorm(d_theta);
  d_qoi["theta_dot"] = vectorNorm(d_theta_dot);
  d_qoi["temperature"] = vectorNorm(d_temperature);
  d_qoi["damage"] = vectorNorm(d_damage);
}

double ThermomechanicalModel::vectorNorm(const std::vector<libMesh::Point> &vec) {
  double norm_sq = 0.;
  for (size_t loc_i=0; loc_i<d_cm_p->d_owned_size; ++loc_i) {
    const auto& i = d_cm_p->d_owned_and_ghost_ids[loc_i];
    norm_sq += vec[i].norm_sq();
  }
  // reduce at processor 0
  d_cm_p->d_comm.sum(norm_sq);
  double norm = std::sqrt(norm_sq / d_mesh.n_nodes());
  return norm;
}

double ThermomechanicalModel::vectorNorm(const std::vector<double> &vec) {
  double norm_sq = 0.;
  for (size_t loc_i=0; loc_i<d_cm_p->d_owned_size; ++loc_i) {
    const auto& i = d_cm_p->d_owned_and_ghost_ids[loc_i];
    norm_sq += vec[i] * vec[i];
  }
  // reduce at processor 0
  d_cm_p->d_comm.sum(norm_sq);
  double norm = std::sqrt(norm_sq / d_mesh.n_nodes());
  return norm;
}

void ThermomechanicalModel::debugVector(const std::vector<double> &vec, const std::string &name) {
  std::vector<double> local_vec(d_mesh.n_nodes(), 0.0);
  std::vector<double> global_vec(d_mesh.n_nodes(), 0.0);

  for (size_t loc_i = 0; loc_i < d_cm_p->d_owned_size; loc_i++) {
    const auto& i = d_cm_p->d_owned_and_ghost_ids[loc_i];  // Global node ID
    local_vec[i] = vec[i];
  }
  // collect all local mx vectors on processor 0
  MPI_Reduce(local_vec.data(), global_vec.data(), 
      d_mesh.n_nodes(), MPI_DOUBLE, MPI_SUM, 0, d_cm_p->d_comm.get());

  if (d_cm_p->d_rank == 0) {
    for (size_t i = 0; i < d_mesh.n_nodes(); i = i + 100) {
      printf("Node = %zu, %s = %f\n", i, name.c_str(), global_vec[i]);
    }
  }
}

void ThermomechanicalModel::debugVector(const std::vector<libMesh::Point> &vec, const std::string &name) {

  std::vector<std::vector<double>> local_vec(3, std::vector<double>(d_mesh.n_nodes(), 0.0));
  std::vector<std::vector<double>> global_vec(3, std::vector<double>(d_mesh.n_nodes(), 0.0));

  for (size_t loc_i = 0; loc_i < d_cm_p->d_owned_size; loc_i++) {
    const auto& i = d_cm_p->d_owned_and_ghost_ids[loc_i];  // Global node ID
    for (size_t j = 0; j < 3; ++j) {
      local_vec[j][i] = vec[i](j);
    }
  }

  // collect all local mx vectors on processor 0
  for (size_t j = 0; j < 3; ++j) {
    MPI_Reduce(local_vec[j].data(), global_vec[j].data(), 
        d_mesh.n_nodes(), MPI_DOUBLE, MPI_SUM, 0, d_cm_p->d_comm.get());
  }

  if (d_cm_p->d_rank == 0) {
    for (size_t i = 0; i < d_mesh.n_nodes(); i = i + 100) {
      printf("Node = %zu, %s = %f, %f, %f\n", i, name.c_str(), global_vec[0][i], global_vec[1][i], global_vec[2][i]);
    }
  }

  std::cout << "debugVector: " << name << " done" << std::endl;
}

std::string ThermomechanicalModel::printStr(int nt, int lvl) const {
  auto tabS = util::io::getTabS(nt);
  std::ostringstream oss;
  oss << tabS << "------- model::ThermomechanicalModel --------" << std::endl
      << std::endl;
  oss << tabS << "time = " << d_time << std::endl;
  oss << tabS << "dt = " << d_dt << std::endl;
  oss << tabS << "mesh: " << std::endl;
  oss << d_mesh.get_info() << std::endl;
  oss << tabS << "equation_systems: " << std::endl;
  oss << d_equation_systems.get_info() << std::endl;
  oss << tabS << "temperature_system: " << std::endl;
  oss << d_temperature_system.get_info() << std::endl;
  oss << tabS << "theta_dot_system: " << std::endl;
  oss << d_theta_dot_system.get_info() << std::endl;
  oss << tabS << "mechanical_system: " << std::endl;
  oss << d_mechanical_system.get_info() << std::endl;

  oss << tabS << "material: " << std::endl;
  oss << d_material_p->printStr(nt + 1, lvl) << std::endl;
  oss << tabS << "communicator: " << std::endl;
  oss << d_cm_p->printStr(nt + 1, lvl) << std::endl;
  oss << tabS << "heat_source: " << std::endl;
  oss << d_heat_sources_p->printStr(nt + 1, lvl) << std::endl;
  oss << tabS << "loading: " << std::endl;
  oss << d_loading_p->printStr(nt + 1, lvl) << std::endl;
  oss << tabS << "number of obs_points: " << d_obs_points.size() << std::endl;
  oss << tabS << "number of obs_nodes: " << d_obs_nodes.size() << std::endl;

  oss << tabS << "number of edge cracks: " << d_edge_cracks.size() << std::endl;
  for (size_t i = 0; i < d_edge_cracks.size(); i++) {
    oss << tabS << "edge crack " << i << ": " << d_edge_cracks[i].printStr(nt + 1, lvl) << std::endl;
  }

  oss << tabS << std::endl;
  return oss.str();
}

void ThermomechanicalModel::print(int nt, int lvl) const {
  std::cout << printStr(nt, lvl);
}

} // namespace model