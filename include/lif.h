#ifndef LIF_H
#define LIF_H

#include <vector>
#include <cmath>

class LIFNeuron {
private:
    // State variables
    float V_;           // membrane potential (mV)
    float I_;           // synaptic current (nA) 
    
    // Parameters
    float V_rest_;      // resting potential (-70mV)
    float V_thresh_;    // threshold (-55mV)
    float V_reset_;     // reset potential (-70mV)
    float tau_mem_;     // membrane time constant (ms)
    float tau_syn_;     // synaptic time constant (ms)
    float R_;           // membrane resistance (MΩ)
    float dt_;          // time step (ms) - NOW USED
    int refractory_counter_;
    int refractory_steps_;
    
public:
    // Constructor with dt and tau_syn
    LIFNeuron(float tau_mem = 20.0f, float tau_syn = 5.0f,
              float V_thresh = -55.0f, float V_rest = -70.0f, 
              float V_reset = -70.0f, float R = 1.0f, 
              int refractory_period = 2, float dt = 1.0f);
    
    void reset();
    bool update(float I_syn);  // Input current from synapses
    
    // Getters
    float get_membrane_potential() const { return V_; }
    float get_synaptic_current() const { return I_; }
    float get_dt() const { return dt_; }
    
    // Setters
    void set_membrane_potential(float V) { V_ = V; }
    void set_synaptic_current(float I) { I_ = I; }
    void set_dt(float dt) { dt_ = dt; }
};

class LIFLayer {
private:
    std::vector<LIFNeuron> neurons_;
    int size_;
    float dt_;
    
public:
    LIFLayer(int size, float tau_mem = 20.0f, float tau_syn = 5.0f, float dt = 1.0f);
    
    std::vector<bool> forward(const std::vector<float>& input_currents);
    void reset();
    int size() const { return size_; }
    void set_dt(float dt);
    
    // Monitoring functions
    std::vector<float> get_membrane_potentials() const;
    std::vector<float> get_synaptic_currents() const;
};

#endif // LIF_H

// #ifndef LIF_H
// #define LIF_H

// #include <vector>
// #include <cmath>

// class LIFNeuron {
// private:
//     float V_;           // membrane potential
//     float V_rest_;      // resting potential (-70mV)
//     float V_thresh_;    // threshold (-55mV)
//     float V_reset_;     // reset potential
//     float tau_m_;       // membrane time constant (ms)
//     float R_;           // membrane resistance
//     float dt_;          // time step
//     int refractory_counter_;
//     int refractory_steps_;
    
// public:
//     LIFNeuron(float tau_m = 20.0f, float V_thresh = -55.0f, 
//               float V_rest = -70.0f, float V_reset = -70.0f,
//               float R = 1.0f, int refractory_period = 2);
    
//     void reset();
//     bool update(float I_syn);
//     float get_membrane_potential() const { return V_; }
//     void set_membrane_potential(float V) { V_ = V; }
// };

// class LIFLayer {
// private:
//     std::vector<LIFNeuron> neurons_;
//     int size_;
    
// public:
//     LIFLayer(int size, float tau_m = 20.0f);
    
//     std::vector<bool> forward(const std::vector<float>& input_currents);
//     void reset();
//     int size() const { return size_; }
// };

// #endif // LIF_H