/*
 * -------------------------------------------
 * Copyright (c) 2021 - 2025 Prashant K. Jha
 * -------------------------------------------
 * https://github.com/CEADpx/multiphysics-peridynamics
 *
 * Distributed under the Boost Software License, Version 1.0. (See accompanying
 * file LICENSE)
 */
#pragma once

#include <memory>
#include <vector>
#include <libmesh/point.h>
#include "util.h"
#include "thermomechanical_model.h"
#include "crack.h"

namespace geom {


/*! @brief A class for fracture state of bonds
 *
 * This class provides method to read and modify fracture state of bonds
 */
class Fracture {

public:

/*! @brief Vector which stores the state of bonds
*
* Given node i, vector d_fracture[i] is the list of state of bonds of node
* i.
*
* We only use 1 bit per bond of node to store the state.
*/
std::vector<std::vector<uint8_t>> d_fracture;

model::ThermomechanicalModel &d_model;  ///< thermomechanical model

/*!
* @brief Constructor
* @param deck Input deck which contains user-specified information
* @param nodes Pointer to nodal coordinates
* @param neighbor_list Pointer to neighbor list
*/
Fracture(model::ThermomechanicalModel &model) : d_model(model) {};

void initialize() {
    d_fracture.resize(d_model.d_mesh.n_nodes());
    for (size_t loc_i = 0; loc_i < d_model.d_cm_p->d_owned_size; loc_i++) {
        const auto& i = d_model.d_cm_p->d_owned_and_ghost_ids[loc_i];
        d_fracture[i].resize(d_model.d_neighbor_list[i].size(), 0);
    }
};

/*!
* @brief Sets the bond state
*
* @param i Nodal id
* @param loc_j Local id of bond in neighbor list of i
* @param state State which is applied to the bond
*/
void setBondState(const size_t &i, const size_t &loc_j, const bool &state) {
    d_fracture[i][loc_j] = state ? 1 : 0;
};

/*!
* @brief Read bond state
*
* @param i Nodal id
* @param loc_j Local id of bond in neighbor list of i
* @return bool True if bond is fractured otherwise false
*/
bool getBondState(const size_t &i, const size_t &loc_j) const {
    return d_fracture[i][loc_j] == 1;
};

/*!
* @brief Returns the list of bonds of node i
*
* @param i Nodal id
* @return list Bonds of node i
*/
const std::vector<uint8_t> getBonds(const size_t &i) const {
    return d_fracture[i];
};

/*!
* @brief Sets state of bond which intersect the pre-crack line as fractured
*
* @param i Nodal id
* @param crack Pre-crack
*/
void computeFracturedBondFd(const libMesh::dof_id_type &i, geom::EdgeCrack &crack) {
    //
    //
    // Here [ ] represents a mesh node and o------o represents a crack.
    //
    //
    //                     pt = pr
    //
    //                      o
    //                     /
    //         [ ]-----[ ]/----[ ]
    //          |       |/      |
    //          |       /       |
    //          |      /|       |
    //         [ ]----/[ ]-----[ ]
    //          |    /  |       |
    //          |   /   |       |
    //          |  /    |       |
    //         [ ]/----[ ]-----[ ]
    //           /
    //          o
    //
    //     pb = pl
    
    //
    //
    // By design, the crack is offset a very small amount (5.0E-8) to bottom
    // and to right side.
    //
    const libMesh::Point& xi = *d_model.d_mesh.node_ptr(i);
    
    // we assume pb is below pt i.e. pb.y < pt.y
    const libMesh::Point pb = crack.d_pb;
    const libMesh::Point pt = crack.d_pt;

    // if (std::abs(xi(1) - 0.005) < 1.e-3 )
    //     printf("node = %d, xi = (%f, %f, %f), crack.ptOutside(xi, crack.d_o, pb, pt) = %d\n", i, xi(0), xi(1), xi(2), crack.ptOutside(xi, crack.d_o, pb, pt));
    
    // check if point is outside crack line
    if (crack.ptOutside(xi, crack.d_o, pb, pt)) return;
    
    // find if this node is on right side or left side of the crack line
    bool left_side = crack.ptLeftside(xi, pb, pt);
    
    //
    if (left_side) {
        // loop over neighboring nodes
        for (size_t loc_j = 0; loc_j < d_model.d_neighbor_list[i].size(); loc_j++) {
            const auto& j = d_model.d_neighbor_list[i][loc_j];
            const libMesh::Point& xj = *d_model.d_mesh.node_ptr(j);
        
            // check if j_node lies on right side of crack line
            bool modify = true;
        
            // check if point lies outside crack line
            if (crack.ptOutside(xj, crack.d_o, pb, pt)) modify = false;
        
            // modify only those nodes which lie on opposite side (in this
            // case on right side)
            if (crack.ptRightside(xj, pb, pt) and modify) {
                // std::cout << "setBondState(i, loc_j, modify) = " << i << ", " << loc_j << ", " << modify << std::endl;
                setBondState(i, loc_j, modify);
            }
        }
    
    }  // left side
    else {
        // loop over neighboring nodes
        for (size_t loc_j = 0; loc_j < d_model.d_neighbor_list[i].size(); loc_j++) {
        const auto& j = d_model.d_neighbor_list[i][loc_j];
            const libMesh::Point& xj = *d_model.d_mesh.node_ptr(j);
        
            // check if j_node lies on left side of crack line
            // As pointed out in the beginning, since we are looking for
            // nodes on left side of the crack, we need to be prepared to
            // handle the case when node is very close to crack line
            // i.e. area (pb, pt, j_node) >= 0.
        
            auto modify = true;
        
            if (crack.ptOutside(xj, crack.d_o, pb, pt)) modify = false;
        
            // modify only those nodes which lie on opposite side (in this
            // case on left side)
            if (crack.ptLeftside(xj, pb, pt) and modify)
                setBondState(i, loc_j, modify);
        }
    }
};

/*!
* @brief Sets fracture state according to the crack data
*/
void addCrack(const double &time) {

    for (auto &crack : d_model.d_edge_cracks) {

        if (!crack.d_crackAcrivated) {
            if (util::isLess(crack.d_activationTime, time)) {
                
                for (size_t loc_i = 0; loc_i < d_model.d_cm_p->d_owned_size; loc_i++) {
                    const auto& i = d_model.d_cm_p->d_owned_and_ghost_ids[loc_i];
        
                    computeFracturedBondFd(i, crack);
                }

                std::cout << "crack.d_crackAcrivated = true" << std::endl;
                crack.d_crackAcrivated = true;
            }
        }
    }
};

}; // class Fracture

} // namespace geom