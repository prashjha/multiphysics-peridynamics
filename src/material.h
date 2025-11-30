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

#include "materialDeck.h"
#include "io.h"
#include <memory>
#include <string>

namespace material {

/*! @brief A base class for computing influence function */
class BaseInfluenceFn {

  public:
    /*! @brief Constructor */
    BaseInfluenceFn() = default;
  
    /*!
     * @brief Returns the value of influence function
     *
     * @param r Reference (initial) bond length
     * @return value Influence function at r
     */
    virtual double get(const double &r) const = 0;
  
    /*!
     * @brief Returns the moment of influence function
     *
     * If \f$ J(r) \f$ is the influence function for \f$ r\in [0,1)\f$ then \f$
     * i^{th} \f$ moment is given by \f[ M_i = \int_0^1 J(r) r^i dr. \f]
     *
     * @param i ith moment
     * @return moment Moment
     */
    virtual double getMoment(const size_t &i) const = 0;
  
    /*!
     * @brief Returns the string containing printable information about the object
     *
     * @param nt Number of tabs to append before printing
     * @param lvl Information level (higher means more information)
     * @return string String containing printable information about the object
     */
    virtual std::string printStr(int nt = 0, int lvl = 0) const {
  
      auto tabS = util::io::getTabS(nt);
      std::ostringstream oss;
      oss << tabS << "------- material::BaseInfluenceFn --------" << std::endl << std::endl;
      oss << tabS << "Provides abstraction for different influence function "
                     "types" << std::endl;
      oss << tabS << std::endl;
  
      return oss.str();
    }
  
    /*!
     * @brief Prints the information about the object
     *
     * @param nt Number of tabs to append before printing
     * @param lvl Information level (higher means more information)
     */
    virtual void print(int nt = 0, int lvl = 0) const { std::cout << printStr(nt, lvl); }
  
  };
  
  /*! @brief A class to implement constant influence function */
  class ConstInfluenceFn : public BaseInfluenceFn {
  public:
    ConstInfluenceFn(const std::vector<double> &params, const size_t &dim) {
      d_a0 = params.empty() ? double(dim + 1) : params[0];
    };
  
    double get(const double &r) const override { return d_a0; }
  
    double getMoment(const size_t &i) const override { return d_a0 / double(i + 1); }
  
    std::string printStr(int nt = 0, int lvl = 0) const override {
  
      auto tabS = util::io::getTabS(nt);
      std::ostringstream oss;
      oss << tabS << "------- material::ConstInfluenceFn --------" << std::endl << std::endl;
      oss << tabS << "Constant function with constant = " << d_a0 << std::endl;
      oss << tabS << "First moment = " << getMoment(1) << std::endl;
      oss << tabS << "Second moment = " << getMoment(2) << std::endl;
      oss << tabS << "Third moment = " << getMoment(3) << std::endl;
      oss << tabS << std::endl;
  
      return oss.str();
    }
  
    void print(int nt = 0, int lvl = 0) const override {
      std::cout << printStr(nt, lvl);
    }
  
  private:
    /*! @brief Constant such that J(r) = Constant */
    double d_a0;
  };
  
  /*! @brief A class to implement linear influence function \f$ J(r) = a0 + a1 r \f$ */
  class LinearInfluenceFn : public BaseInfluenceFn {
  public:
    LinearInfluenceFn(const std::vector<double> &params, const size_t &dim)
    : BaseInfluenceFn(), d_a0(0.), d_a1(0.) {
  
      if (params.empty()) {
        // choose a0, a1 = -a0 such that \int_0^1 J(r) r^d dr = 1
        // and J(r) = a0 (1 - r)
        if (dim == 1) {
          d_a0 = 6.;
          d_a1 = -d_a0;
        } else if (dim == 2) {
          d_a0 = 12.;
          d_a1 = -d_a0;
        } else if (dim == 3) {
          d_a0 = 20.;
          d_a1 = -d_a0;
        }
      } else {
        d_a0 = params[0];
        if (params.size() < 2)
          d_a1 = -d_a0;
        else
          d_a1 = params[1];
      }
    };
  
    double get(const double &r) const override { return d_a0 + d_a1 * r; }
  
    double getMoment(const size_t &i) const override {
      return (d_a0 / double(i + 1)) + (d_a1 / double(i + 2));
    }
  
    std::string printStr(int nt = 0, int lvl = 0) const override {
  
      auto tabS = util::io::getTabS(nt);
      std::ostringstream oss;
      oss << tabS << "------- material::LinearInfluenceFn --------" << std::endl << std::endl;
      oss << tabS << "Linear function a0 + a1*r with constants: a0 = "
                  << d_a0 << ", a1 = " << d_a1 << std::endl;
      oss << tabS << "First moment = " << getMoment(1) << std::endl;
      oss << tabS << "Second moment = " << getMoment(2) << std::endl;
      oss << tabS << "Third moment = " << getMoment(3) << std::endl;
      oss << tabS << std::endl;
  
      return oss.str();
    }
  
    void print(int nt = 0, int lvl = 0) const override {
      std::cout << printStr(nt, lvl);
    }
  
  private:
    /*! @brief Constants such that J(r) = d_a0 + d_a1 * r */
    double d_a0;
    double d_a1;
  };
  
  /*! @brief A class to implement Gaussian influence function \f$ J(r) = \alpha \exp(-r^2/\beta) \f$ */
  class GaussianInfluenceFn : public BaseInfluenceFn {
  public:
    GaussianInfluenceFn(const std::vector<double> &params, const size_t &dim)
    : BaseInfluenceFn(), d_alpha(0.), d_beta(0.) {
  
      if (params.empty()) {
        // beta = 0.2 (default value)
        // choose alpha such that \int_0^1 J(r) r^d dr = 1
        d_beta = 0.2;
        if (dim == 1)
          d_alpha = 2. / (d_beta * (1. - std::exp(-1. / d_beta)));
        else if (dim == 2)
          d_alpha = (4.0 / d_beta) * 1.0 /
                    (std::sqrt(M_PI * d_beta) * std::erf(1.0 / std::sqrt(d_beta)) -
                    2.0 * std::exp(-1.0 / d_beta));
        else if (dim == 3)
          d_alpha = (2.0 / d_beta) * 1.0 /
                    (d_beta - (d_beta + 1.) * std::exp(-1.0 / d_beta));
      } else {
        d_alpha = params[0];
        d_beta = params[1];
      }
    };
  
    double get(const double &r) const override {
      return d_alpha * std::exp(-r * r / d_beta);
    }
  
    double getMoment(const size_t &i) const override  {
  
      double sq1 = std::sqrt(d_beta);
      double sq2 = std::sqrt(M_PI);
      // M_i = \int_0^1 alpha exp(-r^2/beta) r^i dr
    
      if (i == 0) {
        // M0 = 0.5 * \alpha (\beta)^(1/2) * (pi)^(1/2) * erf((1/beta)^(1/2))
    
        return 0.5 * d_alpha * sq1 * sq2 * std::erf(1. / sq1);
      } else if (i == 1) {
        // M1 = 0.5 * \alpha \beta (1 - exp(-1/beta))
    
        return 0.5 * d_alpha * d_beta * (1. - std::exp(-1. / d_beta));
      } else if (i == 2) {
        // M2 = 0.5 * \alpha (\beta)^(3/2) * [0.5 * (pi)^(1/2) erf((1/beta)^(1/2)
        // ) - (1/beta)^(1/2) * exp(-1/beta) ]
    
        return 0.5 * d_alpha * d_beta * sq1 *
               (0.5 * sq2 * std::erf(1. / sq1) -
                (1. / sq1) * std::exp(-1. / d_beta));
      } else if (i == 3) {
        // M3 = 0.5 * \alpha (\beta)^(2) * [1 - ((1/beta) + 1) * exp(-1/beta)]
    
        return 0.5 * d_alpha * d_beta * d_beta *
               (1. - (1. + 1. / d_beta) * std::exp(-1. / d_beta));
      } else {
        std::cerr << "Error: getMoment() accepts argument i from 0 to 3.\n";
        exit(1);
      }  
    };
  
    std::string printStr(int nt = 0, int lvl = 0) const override {
  
      auto tabS = util::io::getTabS(nt);
      std::ostringstream oss;
      oss << tabS << "------- material::GaussianInfluenceFn --------" << std::endl << std::endl;
      oss << tabS << "Gaussian function a0 * exp(-r*r / a1) with constants: a0 = "
                  << d_alpha << ", a1 = " << d_beta << std::endl;
      oss << tabS << "First moment = " << getMoment(1) << std::endl;
      oss << tabS << "Second moment = " << getMoment(2) << std::endl;
      oss << tabS << "Third moment = " << getMoment(3) << std::endl;
      oss << tabS << std::endl;
  
      return oss.str();
    }
  
    void print(int nt = 0, int lvl = 0) const override {
      std::cout << printStr(nt, lvl);
    }
  
  private:
    /*! @brief Constants */
    double d_alpha;
    double d_beta;
  };

/*! @brief A class providing methods to compute energy density and force of peridynamic material */
class Material {
public:
  /*! @brief Deck */
  inp::MaterialDeck d_deck;

  /*! @brief Micromodulus (for bond-based model) */
  double d_c;

  /*! @brief Critical stretch */
  double d_s0;

  /*! @brief Neighorhood volume */
  double d_horizonVolume;

  /*! @brief Factor multiplying the deviatoric strain */
  double d_deviatoricFactor;

  /*! @brief Factor multiplying theta_dot in heat equation */
  double d_thetaDotFactor;

  /*! @brief Factor multiplying temperature change in the force */
  double d_forceTchangeFactor;
  
  /*! @brief Influence function  */
  std::shared_ptr<material::BaseInfluenceFn> d_J_p;

  /*!
   * @brief Constructor
   * @param deck Input deck which contains user-specified information
   */
  Material(inp::MaterialDeck &deck)
      : d_deck(deck) {

    createInfluenceFn();
    computeParameters();
  };

  /*!
   * @brief Returns energy and force between bond due to state-based model
   *
   * @param r Reference (initial) bond length
   * @param s Bond strain
   * @param fs Bond fracture state (true = broken bond)
   * @param mx Weighted volume at node
   * @param thetax Dilation
   * @param Tx Temperature
   * @return value Pair of energy and force
   */
  double
  getBondForce(const double &r, const double &s, bool &fs, const double
  &mx, const double &thetax, const double &Tx, bool for_nodal = true) const {

    // break if above s0:
    if (fs) { return 0.0; }
    if (d_deck.d_breakBonds && s > d_s0) { fs = true; return 0.0; }

    double force = 0.0;
    double J = for_nodal ? 1.0 : getInfFn(r);

    if (d_deck.d_model_type == inp::ModelType::BondBased) {
  
      // bond force magnitude = c * stretch * influence
      force = d_c * s * J;
    } 
    else if (d_deck.d_model_type == inp::ModelType::StateBased) {
      double e = s * r;
      double ei = thetax * r / double(d_deck.d_dim);
      double ed = e - ei;
  
      // force
      force = (d_deck.d_dim * d_deck.d_Kadjust / mx) * (thetax) * J * r 
        + (d_deviatoricFactor / mx) * J * ed;
    }
 
    if (d_deck.d_thermToMechCoupling)
      return force + getThermalBondForce(r, s, fs, mx, thetax, Tx, for_nodal);
    else
      return force;
  };

  double getThermalBondForce(const double &r, const double &s, bool &fs, const double
  &mx, const double &thetax, const double &Tx,  bool for_nodal = true) const {

    // break if above s0:
    if (fs) { return 0.0; }
    if (d_deck.d_breakBonds && s > d_s0) { fs = true; return 0.0; }

    double J = !for_nodal ? getInfFn(r) : 1.0;

    double therm = 0;
    if (std::abs(Tx - d_deck.d_Tref) > 1e-6) {
      therm = d_forceTchangeFactor * (Tx - d_deck.d_Tref);
    }

    // force
    return -(d_deck.d_dim * d_deck.d_Kadjust / mx) * therm * J * r ;
  }

  /*!
   * @brief Returns the unit vector along which bond-force acts
   *
   * @param dx Reference bond vector
   * @param du Difference of displacement
   * @return vector Unit vector
   */
  libMesh::Point getBondForceDirection(const libMesh::Point &dx,
                                     const libMesh::Point &du) const {
    return (dx + du) / (dx + du).norm();
  };

  /*!
   * @brief Returns the bond strain
   * @param dx Reference bond vector
   * @param du Difference of displacement
   * @return strain Bond strain \f$ S = \frac{du \cdot dx}{|dx|^2} \f$
   */
  double getS(const libMesh::Point &dx, const libMesh::Point &du) const {
    return ((dx + du).norm() - dx.norm()) / dx.norm();
  };

  /*!
   * @brief Returns critical bond strain
   *
   * @param r Reference length of bond
   * @return strain Critical strain
   */
  double getSc(const double &r) const { return d_s0; };

  /*!
   * @brief Returns the density of the material
   * @return density Density of the material
   */
  double getDensity() const { return d_deck.d_rho; };

  /*!
   * @brief Returns the value of influence function
   *
   * @param r Reference (initial) bond length
   * @return value Influence function at r
   */
  double getInfFn(const double &r) const {
    return d_J_p->get(r / d_deck.d_horizon);
  };

  /*!
   * @brief Returns the moment of influence function
   *
   * If \f$ J(r) \f$ is the influence function for \f$ r\in [0,1)\f$ then \f$
   * i^{th}\f$ moment is given by \f[ M_i = \int_0^1 J(r) r^i dr. \f]
   *
   * @param i ith moment
   * @return value Moment
   */
  double getMoment(const size_t &i) const {
    return d_J_p->getMoment(i);
  };

  /*!
   * @brief Returns horizon
   *
   * @return horizon Horizon
   */
  double getHorizon() const { return d_deck.d_horizon; };

  /*!
   * @brief Returns the string containing printable information about the object
   *
   * @param nt Number of tabs to append before printing
   * @param lvl Information level (higher means more information)
   * @return string String containing printable information about the object
   */
  std::string printStr(int nt, int lvl) const {

    auto tabS = util::io::getTabS(nt);
    std::ostringstream oss;
    oss << tabS << "------- material::Material --------" << std::endl
        << std::endl;
    oss << tabS << "s0 = " << d_s0 << std::endl;
    oss << tabS << "deviatoric factor = " << d_deviatoricFactor << std::endl;
    oss << tabS << "theta dot factor = " << d_thetaDotFactor << std::endl;
    oss << tabS << "force T change factor = " << d_forceTchangeFactor << std::endl;
    oss << tabS << "Material deck: " << std::endl;
    oss << d_deck.printStr(nt + 1, lvl) << std::endl;
    oss << tabS << "Horizon = " << d_deck.d_horizon << std::endl;
    oss << tabS << "Influence fn address = " << d_J_p.get() << std::endl;
    oss << tabS << "Influence fn info: " << std::endl;
    oss << d_J_p->printStr(nt + 1, lvl) << std::endl;
    oss << tabS << std::endl;

    return oss.str();
  };

  /*!
   * @brief Prints the information about the object
   *
   * @param nt Number of tabs to append before printing
   * @param lvl Information level (higher means more information)
   */
  void print(int nt = 0, int lvl = 0) const {
    std::cout << printStr(nt, lvl);
  };

  void createInfluenceFn() {
      // create influence function
      if (d_deck.d_influenceFnType == 0) {
        d_J_p = std::make_shared<material::ConstInfluenceFn>(d_deck.d_influenceFnParams, d_deck.d_dim);
      }
      else if (d_deck.d_influenceFnType == 1) {
        d_J_p = std::make_shared<material::LinearInfluenceFn>(d_deck.d_influenceFnParams, d_deck.d_dim);
      }
      else if (d_deck.d_influenceFnType == 2) {
        d_J_p = std::make_shared<material::GaussianInfluenceFn>(d_deck.d_influenceFnParams, d_deck.d_dim);
      }
      else {
        std::cerr << "Error: Influence function type = "
                  << d_deck.d_influenceFnType
                  << " is invalid.\n";
        exit(1);
      }
  }

  /*! @brief Compute material model parameters */
  void computeParameters() {

    // Compute the volume of a ball of radius d_horizon in the given dimension
    if (d_deck.d_dim == 1) {
      d_horizonVolume = 2.0 * d_deck.d_horizon;
    } else if (d_deck.d_dim == 2) {
      d_horizonVolume = M_PI * d_deck.d_horizon * d_deck.d_horizon;
    } else if (d_deck.d_dim == 3) {
      d_horizonVolume = (4.0 / 3.0) * M_PI * d_deck.d_horizon * d_deck.d_horizon * d_deck.d_horizon;
    } else {
      throw std::runtime_error("Dimension not supported");
    }

    if (d_deck.d_model_type == inp::ModelType::StateBased) {
    
      // adjust bulk modulus
      d_deck.d_Kadjust = d_deck.d_K;
      if (d_deck.d_dim == 2) {
        if (d_deck.d_isPlaneStrain)
          d_deck.d_Kadjust = d_deck.d_K / (2*(1 + d_deck.d_nu) * (1 - 2*d_deck.d_nu));
        else
          d_deck.d_Kadjust = d_deck.d_K / (2*(1 - d_deck.d_nu));
      }

      d_s0 = std::sqrt(5*d_deck.d_Gc / (9.*d_deck.d_K*d_deck.d_horizon));

      // compute deviatoric factor
      d_deviatoricFactor = 15*d_deck.d_G;
      d_thetaDotFactor = d_deck.d_Kadjust*d_deck.d_alpha;
      d_forceTchangeFactor = d_deck.d_dim * d_deck.d_alpha;
      if (d_deck.d_dim == 2) {
        d_deviatoricFactor = 8*d_deck.d_G;
        d_thetaDotFactor *= 2;
        if (d_deck.d_isPlaneStrain) {
          d_thetaDotFactor *= (1 + d_deck.d_nu);
          d_forceTchangeFactor *= (1 + d_deck.d_nu);
        }
      }
    }

    if (d_deck.d_model_type == inp::ModelType::BondBased) {
      
      // 1) micromodulus c [units: force/length^2 in 3D]
      if (d_deck.d_dim == 3) {
        // common 3D formula with ω(r)=1
        d_c = 12.0 * d_deck.d_E / (M_PI * std::pow(d_deck.d_horizon, 4));
      }
      else if (d_deck.d_dim == 2) {
        // plane stress approximation
        d_c = 8.0 * d_deck.d_E / (M_PI * std::pow(d_deck.d_horizon, 3));
      }
      else /* d_deck.d_dim == 1 */ {
        d_c = d_deck.d_E / d_deck.d_horizon;
      }


      d_s0 = std::sqrt(2.0 * d_deck.d_Gc / (d_c * d_horizonVolume));
    }
  };
};

} // namespace material
