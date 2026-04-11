#include "lif.h"

LIFNeuron::LIFNeuron(float tau_mem, float tau_syn, float V_thresh, 
                     float V_rest, float V_reset, float R, 
                     int refractory_period, float dt)
    : V_rest_(V_rest), V_thresh_(V_thresh), V_reset_(V_reset),
      tau_mem_(tau_mem), tau_syn_(tau_syn), R_(R), dt_(dt), 
      refractory_steps_(refractory_period) {
    reset();
}

void LIFNeuron::reset() {
    V_ = V_rest_;
    I_ = 0.0f;  // Reset synaptic current to zero
    refractory_counter_ = 0;
}

bool LIFNeuron::update(float I_syn) {
    // Refractory period: neuron can't spike
    if (refractory_counter_ > 0) {
        refractory_counter_--;
        return false;
    }
    
    // STEP 1: Add incoming synaptic current to internal current
    I_ += I_syn;
    
    // STEP 2: Update membrane voltage with dt
    // dv/dt = (V_rest - V)/tau_mem + I/tau_mem
    float dV = dt_ * ((V_rest_ - V_) / tau_mem_ + I_ / tau_mem_);
    V_ += dV;
    
    // STEP 3: Update synaptic current (exponential decay) with dt
    // di/dt = -I/tau_syn
    float dI = -dt_ * I_ / tau_syn_;
    I_ += dI;
    
    // STEP 4: Spike condition
    if (V_ >= V_thresh_) {
        V_ = V_reset_;
        refractory_counter_ = refractory_steps_;
        return true;
    }
    
    return false;
}

LIFLayer::LIFLayer(int size, float tau_mem, float tau_syn, float dt) 
    : size_(size), dt_(dt) {
    neurons_.reserve(size);
    for (int i = 0; i < size; i++) {
        neurons_.emplace_back(tau_mem, tau_syn, -55.0f, -70.0f, -70.0f, 1.0f, 2, dt);
    }
}

std::vector<bool> LIFLayer::forward(const std::vector<float>& input_currents) {
    std::vector<bool> spikes(size_, false);
    for (int i = 0; i < size_; i++) {
        spikes[i] = neurons_[i].update(input_currents[i]);
    }
    return spikes;
}

void LIFLayer::reset() {
    for (auto& neuron : neurons_) {
        neuron.reset();
    }
}

void LIFLayer::set_dt(float dt) {
    dt_ = dt;
    for (auto& neuron : neurons_) {
        neuron.set_dt(dt);
    }
}

std::vector<float> LIFLayer::get_membrane_potentials() const {
    std::vector<float> potentials(size_);
    for (int i = 0; i < size_; i++) {
        potentials[i] = neurons_[i].get_membrane_potential();
    }
    return potentials;
}

std::vector<float> LIFLayer::get_synaptic_currents() const {
    std::vector<float> currents(size_);
    for (int i = 0; i < size_; i++) {
        currents[i] = neurons_[i].get_synaptic_current();
    }
    return currents;
}


// #include "lif.h"

// LIFNeuron::LIFNeuron(float tau_m, float V_thresh, float V_rest, 
//                      float V_reset, float R, int refractory_period)
//     : V_rest_(V_rest), V_thresh_(V_thresh), V_reset_(V_reset),
//       tau_m_(tau_m), R_(R), refractory_steps_(refractory_period) {
//     reset();
// }

// void LIFNeuron::reset() {
//     V_ = V_rest_;
//     refractory_counter_ = 0;
// }

// bool LIFNeuron::update(float I_syn) {
//     // Refractory period: neuron can't spike
//     if (refractory_counter_ > 0) {
//         refractory_counter_--;
//         return false;
//     }
    
//     // Leaky integrate: dV/dt = (V_rest - V)/tau + I*R/tau
//     float dV = (V_rest_ - V_) / tau_m_ + I_syn * R_ / tau_m_;
//     V_ += dV;
    
//     // Spike condition
//     if (V_ >= V_thresh_) {
//         V_ = V_reset_;
//         refractory_counter_ = refractory_steps_;
//         return true;
//     }
    
//     return false;
// }

// LIFLayer::LIFLayer(int size, float tau_m) : size_(size) {
//     neurons_.resize(size, LIFNeuron(tau_m));
// }

// std::vector<bool> LIFLayer::forward(const std::vector<float>& input_currents) {
//     std::vector<bool> spikes(size_, false);
//     for (int i = 0; i < size_; i++) {
//         spikes[i] = neurons_[i].update(input_currents[i]);
//     }
//     return spikes;
// }

// void LIFLayer::reset() {
//     for (auto& neuron : neurons_) {
//         neuron.reset();
//     }
// }