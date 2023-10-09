/**
 * Optimised C++ implementation of the Dynamic Mean Field (DMF) model with a
 * BOLD Balloon-Windkessel model.
 *
 * Pedro Mediano, Apr 2020
 */

#ifndef DMF_H
#define DMF_H

#include<string>
#include<thread>
#include<iostream>
#include<random>
#include<map>

#include "Eigen/Dense"
#include "mex.h"

typedef std::map<std::string, const double*> ParamStruct;

/**
 * Simple utility struct to pass ranges of integer values.
 *
 * Example usage:
 *
 * int start=0, end=0;
 * Range r(start, end);
 *
 */
struct Range {
    size_t start, end;
    Range(size_t s, size_t e) : start(s), end(e) {}
};


/**
 * Clip value of double between 0 and 1.
 */
inline double clip(double x) {
    return x > 1 ? 1 : (x < 0 ? 0 : x);
}


/**
 * Retrieves a field from a ParamStruct and makes sure that result is an array
 * of size N. If the content of the field is a scalar, it constructs a constant
 * array.
 *
 * @param params
 * @param field name of field to be retrieved
 * @param N expected size of the array
 */
Eigen::ArrayXd ensureArray(const ParamStruct &params, std::string field, size_t N) {

    // Initialise array and field name string
    Eigen::ArrayXd A;
    std::string isvecf("isvector_");
    isvecf.append(field);

    if (params.at(isvecf.c_str())) {
        // Map the Eigen::Array to field if it is listed as a vector
        A = Eigen::Map<const Eigen::ArrayXd>(params.at(field.c_str()), N);

    } else {
        // Otherwise initialise and fill a constant array
        A = Eigen::ArrayXd::Zero(N);
        A.fill(params.at(field.c_str())[0]);
    }

    return A;
}


/**
 * Checks input parameter structure has all necessary fields.
 *
 * @param params
 */
void checkParams(const ParamStruct &params) {

    // Check that parameter struct has the necessary fields
    std::vector<std::string> required_fields = {"C", "receptors", "dt",
            "taon", "taog", "gamma", "sigma", "JN", "I0", "Jexte", "Jexti",
            "w", "g_e", "Ie", "de", "g_i", "Ii", "di", "wgaine", "wgaini",
            "G", "TR", "dtt", "batch_size"};
    for (const auto& field : required_fields) {
        if (!params.count(field.c_str())) {
            std::string s("Missing field in parameter struct: ");
            s.append(field);
            throw std::invalid_argument(s);
        }
    }
}


/**
 * Integrator of firing rates to simulate BOLD signals
 */
class BOLDIntegrator {


public:
    // Balloon-Windkessel equation parameters
    double taus  = 0.65;
    double tauf  = 0.41;
    double tauo  = 0.98;
    double alpha = 0.32;

    double Eo = 0.4;
    double TE = 0.04;
    double vo = 0.04;
    double k1 = 4.3*40.3*Eo*TE;
    double k2 = 25*Eo*TE;
    double k3 = 1;

    double itaus  = 1/taus;
    double itauf  = 1/tauf;
    double itauo  = 1/tauo;
    double ialpha = 1/alpha;
    double dt;

    size_t N, save_every, nb_bold_steps, rate_buffer_size;
    size_t count = 1;  // Start in 1 to match subsampling in Deco code
    size_t b_idx = 0;
    size_t i = 0;
    std::thread th;

    Eigen::Map<Eigen::MatrixXd> b;
    Eigen::Map<Eigen::ArrayXXd> r;
    Eigen::Map<Eigen::ArrayXd> s, f, v, q;
    Eigen::Map<Eigen::ArrayXd> ds, df, dv, dq;
    Eigen::ArrayXd Z;
    Eigen::ArrayXd dZ;

    /**
     * Construct integrator.
     *
     * @param params Matlab struct with all necessary parameters (see checkArguments)
     * @param nb_rate_steps
     * @param N_in
     */
    BOLDIntegrator(ParamStruct params, size_t nb_rate_steps, size_t N_in) :
              dt(params["dtt"][0]),
              N(N_in),
              save_every(params["TR"][0]/params["dtt"][0]),
              nb_bold_steps(nb_rate_steps*params["dtt"][0]/params["TR"][0]),
              rate_buffer_size(nb_rate_steps),
              b(NULL, N_in, nb_rate_steps*params["dtt"][0]/params["TR"][0]),
              r(NULL, N_in, nb_rate_steps),
              s(NULL, N_in),
              f(NULL, N_in),
              v(NULL, N_in),
              q(NULL, N_in),
              ds(NULL, N_in),
              df(NULL, N_in),
              dv(NULL, N_in),
              dq(NULL, N_in),
              Z(Eigen::ArrayXd::Ones(4*N_in)),
              dZ(4*N_in)
    {}

    /**
     * Initialise maps and arrays.
     *
     * Only two arrays are allocated, Z (value) and dZ (derivative) with the
     * state of the BW model. For readability, chunks of these arrays are split
     * and mapped to the standard variables in the BW model (s,f,v,q).
     *
     * As the integrator runs, fills a pre-allocated array with BOLD values.
     *
     * @param[in] rate_res reference to firing rate array
     * @param[out] bold_res writeable reference to BOLD array
     */
    void init(double* rate_res, double* bold_res) {

        double *Z_ptr = &Z(0), *dZ_ptr = &dZ(0);
        new (&s) Eigen::Map<Eigen::ArrayXd>(Z_ptr, N);
        new (&f) Eigen::Map<Eigen::ArrayXd>(Z_ptr + 1*N, N);
        new (&v) Eigen::Map<Eigen::ArrayXd>(Z_ptr + 2*N, N);
        new (&q) Eigen::Map<Eigen::ArrayXd>(Z_ptr + 3*N, N);
        new (&ds) Eigen::Map<Eigen::ArrayXd>(dZ_ptr, N);
        new (&df) Eigen::Map<Eigen::ArrayXd>(dZ_ptr + 1*N, N);
        new (&dv) Eigen::Map<Eigen::ArrayXd>(dZ_ptr + 2*N, N);
        new (&dq) Eigen::Map<Eigen::ArrayXd>(dZ_ptr + 3*N, N);
        
        s.fill(0);

        new (&b) Eigen::Map<Eigen::MatrixXd>(bold_res, N, nb_bold_steps);
        new (&r) Eigen::Map<Eigen::ArrayXXd>(rate_res, N, rate_buffer_size);

    }

    /**
     * Compute BW equations for the value of firing rates at position idx in
     * the shared array.
     *
     * @param idx index of firing rate to use
     */
    void compute(size_t idx) {

      ds = ( r.col(idx % rate_buffer_size) - itaus*s - itauf*(f - 1) );
      df = s;
      dv = itauo*(f - v.pow(ialpha));
      dq = itauo*(f*(1-pow(1-Eo, 1/f))/Eo - (v.pow(ialpha-1))*q);

      Z += dt*dZ;
      count++;

      if (count == save_every) {
          b.col(b_idx)  = vo*( k1*(1-q) + k2*(1-q/v) + k3*(1-v) );
          count -= save_every;
          b_idx++;
      }
    }

    /**
     * Compute a full batch of BOLD steps for firing rate values in the given
     * range of the shared array.
     *
     * @param range range of firing rate indices to use
     */
    void compute_range(Range range) {
        for (auto i = range.start; i <= range.end; i++) {
            compute(i);
        }
    }

    /**
     * Spawn a new thread and compute a batch of BOLD steps for firing rate
     * values in the given range of the shared array.
     *
     * @param range range of firing rate indices to use
     */
    void compute_async(Range r) {
        th = std::thread([=] { compute_range(r); });
    }

    /**
     * Join the thread in which the integrator is computing asynchronously,
     * if possible.
     *
     * Simple wrapper around std::thread::join.
     */
    void join() {
        if (th.joinable()) { th.join(); }
    }

    /**
     * Make sure that copy constructors are disabled (to avoid making copies
     * of this object when passing it to threads).
     */
    BOLDIntegrator(const BOLDIntegrator&) = delete;
    void operator=(const BOLDIntegrator&) = delete;

};


/**
 * Main class of the DMF simulator.
 *
 * Integrates the system of ODEs of the Dynamic Mean Field model using a
 * stochastic Euler-Maruyama method. Includes a parallelised integrator for the
 * BOLD Balloon-Windkessel model to simulate fMRI time series.
 *
 */
class DMFSimulator {

public:
    double dt;
    double I0;
    double Iext;
    double w;
    double JN;
    Eigen::MatrixXd C;
    Eigen::MatrixXd Iext_all;
    double G;
    double gamma;
    double sigma;
    double taog;
    double taon;
    double g_e;
    double g_i;
    double Ie;
    double Ii;
    double de;
    double di;
    double dtt;

    size_t nb_steps, N, batch_size, steps_per_millisec, seed;
    bool return_rate, return_bold;

    Eigen::ArrayXd sn, sg, J, receptors, Jexte, Jexti, wgaine, wgaini;

    BOLDIntegrator bold_int;

    /**
     * Constructor.
     *
     * @param params Matlab struct with all necessary parameters (see checkArguments)
     * @param nb_steps_in number of firing rate steps to simulate
     * @param N_in number of nodes/ROIs in the model
     * @param return_rate_in boolean, whether to return firing rates
     * @param return_bold_in boolean, whether to return BOLD activity
     */
    DMFSimulator(ParamStruct params, size_t nb_steps_in, size_t N_in,
                 bool return_rate_in, bool return_bold_in) :
            dt(params["dt"][0]),
            I0(params["I0"][0]),
            Iext(params["Iext"][0]),
            w(params["w"][0]),
            JN(params["JN"][0]),
            G(params["G"][0]),
            gamma(params["gamma"][0]),
            sigma(params["sigma"][0]),
            taog(params["taog"][0]),
            taon(params["taon"][0]),
            g_e(params["g_e"][0]),
            g_i(params["g_i"][0]),
            Ie(params["Ie"][0]),
            Ii(params["Ii"][0]),
            de(params["de"][0]),
            di(params["di"][0]),
            dtt(params["dtt"][0]),
            nb_steps(nb_steps_in),
            N(N_in),
            batch_size(params["batch_size"][0]),
            steps_per_millisec(1.0/params["dt"][0]),
            return_rate(return_rate_in),
            return_bold(return_bold_in),
            sn(N_in),
            sg(N_in),
            bold_int(params, nb_steps, N_in) {

              C = Eigen::Map<const Eigen::MatrixXd>(params["C"], N, N);
              Iext_all = Eigen::Map<const Eigen::MatrixXd>(params["Iext_all"], N, int(nb_steps/dt));


              receptors = ensureArray(params, "receptors", N);
              wgaine     = ensureArray(params, "wgaine", N);
              wgaini     = ensureArray(params, "wgaini", N);
              Jexte     = ensureArray(params, "Jexte", N);
              Jexti     = ensureArray(params, "Jexti", N);
              J         = ensureArray(params, "J", N);

              if (params.count("seed")) {
                seed = params["seed"][0];
              } else {
                seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
              }

              Eigen::initParallel();

    };


    inline Eigen::ArrayXd curr2rate(const Eigen::ArrayXd& x, const Eigen::ArrayXd& wgain, double g,
           double I, double d) {
        Eigen::ArrayXd y = g*(x-I)*(1+receptors*wgain);
        return y/(1-exp(-d*y));
    }


    void run(double* rate_res, double* bold_res) {
        mexPrintf("hello world");
        
        // Initialise BOLD integrator if needed
        if (return_bold) { bold_int.init(rate_res, bold_res); }
        double bold_timer = 0;
        size_t last_bold = 0;

        // Build Eigen::Map to return rates by reference
        size_t rate_size = return_rate ? nb_steps : (2*batch_size);
        Eigen::Map<Eigen::ArrayXXd> rn(rate_res, N, rate_size);

        // Initialise PRNG and arrays, and start simulation
        std::default_random_engine e(seed);
        std::normal_distribution<double> n(0, std::sqrt(dt)*sigma);
        sn.fill(0.001);
        sg.fill(0.001);
        Eigen::ArrayXd rnd = Eigen::ArrayXd::Zero(N);

        for (size_t t = 0; t < nb_steps; t++) {

            size_t rate_idx = t % rate_size;

            for (size_t dummy = 0; dummy < steps_per_millisec; dummy++) {
                //Eigen::ArrayXd xn = I0*Jexte + w*JN*sn + G*JN*(C*sn.matrix()).array() - J*sg  + Iext *Iext_all.col(t*steps_per_millisec+dummy);
                Eigen::ArrayXd Iexternal = Iext*Iext_all.col(t*steps_per_millisec+dummy);
                Eigen::ArrayXd xn = I0*Jexte + w*JN*sn + G*JN*(C*sn.matrix()).array() - J*sg + Iexternal;
                
                
                Eigen::ArrayXd xg = I0*Jexti + JN*sn - sg;

                rn.col(rate_idx) = curr2rate(xn, wgaine, g_e, Ie, de);
                Eigen::ArrayXd rg = curr2rate(xg, wgaini, g_i, Ii, di);

                rnd = rnd.unaryExpr([&n, &e](double dummy){return n(e);});
                sn += dt*(-sn/taon+(1-sn)*gamma*rn.col(rate_idx)/1000) + rnd;
                sn = sn.unaryExpr(&clip);

                rnd = rnd.unaryExpr([&n, &e](double dummy){return n(e);});
                sg += dt*(-sg/taog+rg/1000) + rnd;
                sg = sg.unaryExpr(&clip);
            }

            auto start = std::chrono::steady_clock::now();
            if (return_bold && ( ((t+1) % batch_size) == 0)) {
                bold_int.join();

                bold_int.compute_async(Range(rate_idx - batch_size + 1, rate_idx));

                last_bold = (rate_idx + 1) % batch_size;

                #ifdef NO_PARALLEL
                bold_int.join();
                #endif
            }
            auto end = std::chrono::steady_clock::now();
            auto diff = end - start;
            bold_timer += std::chrono::duration <double, std::milli> (diff).count();

        }

        if (return_bold) {
            bold_int.join();

            if  ((nb_steps%batch_size) > last_bold) {
                // Compute the remainder of BOLD samples
                // TODO: this will compute things even if they do not end up leading to
                // a new actual BOLD sample. Fix to avoid unnecessary computation.
                bold_int.compute_range(Range(last_bold, nb_steps%batch_size));
            }

          // std::cout << "Bold time: " << bold_timer << std::endl;
        }

    }

};

#endif

