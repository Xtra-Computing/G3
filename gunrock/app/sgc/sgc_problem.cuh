// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * graphsum_problem.cuh
 *
 * @brief GPU Storage management Structure for SSSP Problem Data
 */

#pragma once

#include <gunrock/app/problem_base.cuh>
#include <gunrock/app/sparseMatMul/sparseMatMul_app.cu>
#include <gunrock/app/CrossEntropyLoss/CrossEntropyLoss_app.cu>

// include cholmod from SuitSparse
#include "cholmod.h"

namespace gunrock {
namespace app {
namespace sgc {
/**
 * @brief Speciflying parameters for graphsum Problem
 * @param  parameters  The util::Parameter<...> structure holding all parameter
 * info \return cudaError_t error message(s), if any
 */
cudaError_t UseParameters_problem(util::Parameters &parameters) {
  cudaError_t retval = cudaSuccess;

  GUARD_CU(gunrock::app::UseParameters_problem(parameters));
  GUARD_CU(parameters.Use<int>("in_dim",
      util::OPTIONAL_ARGUMENT | util::SINGLE_VALUE | util::OPTIONAL_PARAMETER,
      0, "input_dimension", __FILE__, __LINE__))

  GUARD_CU(parameters.Use<int>("out_dim",
      util::OPTIONAL_ARGUMENT | util::SINGLE_VALUE | util::OPTIONAL_PARAMETER,
      0, "input_dimension", __FILE__, __LINE__))

  GUARD_CU(parameters.Use<double>("learning_rate",
      util::OPTIONAL_ARGUMENT | util::SINGLE_VALUE | util::OPTIONAL_PARAMETER,
      0.005, "learning rate", __FILE__, __LINE__))

  return retval;
}

struct module {
  util::GpuTimer timer;
  virtual ~module() = default;
  virtual void forward(bool) = 0;
  virtual void backward() = 0;
  virtual double GetLoss() {
    return 0;
  }
};

/**
 * @brief Single-Source Shortest Path Problem structure.
 * @tparam _GraphT  Type of the graph
 * @tparam _LabelT  Type of labels used in sssp
 * @tparam _ValueT  Type of per-vertex distance values
 * @tparam _FLAG    Problem flags
 */
template <typename _GraphT, typename _LabelT = typename _GraphT::VertexT,
          typename _ValueT = typename _GraphT::ValueT,
          ProblemFlag _FLAG = Problem_None>
struct Problem : ProblemBase<_GraphT, _FLAG> {
  typedef _GraphT GraphT;
  static const ProblemFlag FLAG = _FLAG;
  typedef typename GraphT::VertexT VertexT;
  typedef typename GraphT::SizeT SizeT;
  typedef typename GraphT::CsrT CsrT;
  typedef typename GraphT::GpT GpT;
  typedef _LabelT LabelT;
  typedef _ValueT ValueT;

  typedef ProblemBase<GraphT, FLAG> BaseProblem;
  typedef DataSliceBase<GraphT, FLAG> BaseDataSlice;
  typedef util::Parameters Parameters;

  typedef util::Array1D<SizeT, ValueT> Array;
  typedef typename app::TestGraph<VertexT, SizeT, ValueT, graph::HAS_CSR | graph::HAS_EDGE_VALUES>
      SpmatT;

  // Helper structures

  struct adam_var {
    Array weights, grads, m, v;
    bool decay;
    adam_var(Array &_w, Array &_g, bool _d) : decay(_d), weights(_w), grads(_g) {
      init();
    }
    cudaError_t init() {
      auto retval = cudaSuccess;
      GUARD_CU (m.Allocate (weights.GetSize (), util::DEVICE))
      GUARD_CU (m.ForEach ([]__host__ __device__(ValueT &x) { x = 0; }))
      GUARD_CU (v.Allocate (weights.GetSize (), util::DEVICE))
      GUARD_CU(v.ForEach ([]__host__ __device__(ValueT &x) { x = 0; }))
      return retval;
    }
  };

  struct sgconv : module {
    typedef app::sparseMatMul::Problem<SpmatT> ProblemT;
    typedef app::sparseMatMul::Enactor<ProblemT> EnactorT;

    SpmatT *a;
    util::Array1D<SizeT, ValueT> b, c, b_grad, c_grad;
    ProblemT *problem;
    EnactorT *enactor;
    int num_nodes, in_dim, out_dim;
    float *fw_time, *bw_time;

    sgconv(Parameters &p, SpmatT &s, int k, SpmatT &x, util::Array1D<SizeT, ValueT> &_b,
        util::Array1D<SizeT, ValueT> &_b_grad, util::Array1D<SizeT, ValueT> &_c, util::Array1D<SizeT, ValueT> &_c_grad,
        int _in_dim, int _out_dim, float *_fw_time, float *_bw_time) :
        b(_b), c(_c), b_grad(_b_grad), c_grad(_c_grad), in_dim(_in_dim), out_dim(_out_dim), fw_time(_fw_time),
        bw_time(_bw_time) {
      typedef typename SpmatT::CsrT CsrT;
      util::GpuTimer timer;
      float tot = 0;
      // calculate S^k
      cholmod_common c;
      cholmod_start(&c);
      c.useGPU = 1;
      cholmod_sparse *t = cholmod_allocate_sparse(s.nodes, s.nodes, s.edges, 0, 1, 0, CHOLMOD_REAL, &c);
      cholmod_sparse *res = t;
      t->p = s.CsrT::row_offsets.GetPointer(util::HOST);
      t->i = s.CsrT::column_indices.GetPointer(util::HOST);
      t->x = s.CsrT::edge_values.GetPointer(util::HOST);
//      cholmod_sparse *res = cholmod_speye(s.nodes, s.nodes, CHOLMOD_REAL, &c);
      timer.Start();
      for (int _ = 0; _ < k - 1; _++) {
        cholmod_sparse *tmp = cholmod_ssmult(res, t, 0, 1, 0, &c);
//        cholmod_free_sparse(&res, &c);
        res = tmp;
      }
      timer.Stop();
      tot += timer.ElapsedMillis();
      // calculate X'(S^k)'
//      cholmod_free_sparse(&t, &c);
      t = cholmod_allocate_sparse(in_dim, s.nodes, x.edges, 0, 1, 0, CHOLMOD_REAL, &c);
      t->p = x.CsrT::row_offsets.GetPointer(util::HOST);
      t->i = x.CsrT::column_indices.GetPointer(util::HOST);
      t->x = x.CsrT::edge_values.GetPointer(util::HOST);
      timer.Start();
      cholmod_sparse *tmp = cholmod_ssmult(t, res, 0, 1, 0, &c);
      timer.Stop();
      tot += timer.ElapsedMillis();
      printf("preprocess time: %fms", tot);
//      cholmod_free_sparse(&res, &c);
      res = tmp;
      s.nodes = res->ncol;
      s.edges = res->nzmax;
      s.CsrT::row_offsets.SetPointer((uint32_t *)res->p, s.nodes, util::HOST);
      s.CsrT::column_indices.SetPointer((uint32_t *)res->i, s.edges, util::HOST);
      s.CsrT::edge_values.SetPointer((ValueT *)res->x, s.edges, util::HOST);

      problem = new ProblemT(p);
      enactor = new EnactorT();

      problem->Init(s, in_dim, out_dim);
      enactor->Init(*problem);

      cholmod_free_work(&c);
      cholmod_finish(&c);
    }

    virtual void forward(bool train) override {
      timer.Start ();

      problem->Reset(1, b, c);
      enactor->Reset();
      enactor->Enact();

      timer.Stop ();
      *fw_time += timer.ElapsedMillis ();
    }

    virtual void backward() override {
      timer.Start ();
      problem->Reset(0, c_grad, b_grad);
      enactor->Reset();
      enactor->Enact();

      timer.Stop ();
      *bw_time += timer.ElapsedMillis ();
    }
  };

  struct cross_entropy : module {
    typedef app::CrossEntropyLoss::Problem<GraphT> ProblemT;
    typedef app::CrossEntropyLoss::Enactor<ProblemT> EnactorT;

    GraphT dummy;
    util::Array1D<SizeT, ValueT> logits, grad;
    util::Array1D<SizeT, int> truth;
    ProblemT *problem;
    EnactorT *enactor;
    int dim;
    float *fw_time;

    cross_entropy(Parameters &p, Array _logits, Array _grad,
                  util::Array1D<SizeT, int> _truth, int num_nodes, int num_classes, float *_fw,
                  bool training=true) :
        logits(_logits), grad(_grad), truth(_truth), fw_time(_fw) {
      problem = new ProblemT(p);
      enactor = new EnactorT();

      problem->Init(dummy, num_nodes, num_classes, logits, grad, truth, training);
      enactor->Init(*problem);
    }

    virtual void forward(bool train) override {
      timer.Start ();

      problem->Reset(train);
      enactor->Reset();
      enactor->Enact();

      timer.Stop ();
      *fw_time += timer.ElapsedMillis ();
    }

    virtual void backward() override {}

    virtual double GetLoss() override {
      double loss;
      problem->Extract (&loss);
      return loss;
    }
  };

  /**
   * @brief Data structure containing graphsum-specific data on indivual GPU.
   */
  struct DataSlice : BaseDataSlice {
    typedef util::Array1D<SizeT, ValueT> Array;

    double learning_rate = 0.2, beta1 = 0.9, beta2 = 0.999, eps = 1e-8, weight_decay = 5e-4;

    std::vector<module*> modules;
    std::vector<adam_var> vars;
    util::Array1D<SizeT, int> truth, wrong, cnt, label, split;
    Array penalty, theta, SkXtheta, SkXtheta_grad;
    Array theta_grad, in_feature;
    curandGenerator_t gen;
    util::GpuTimer timer;
    float tot_time = 0, fw_sgconv = 0, bw_sgconv = 0, fw_loss = 0, bw_loss = 0;

    int in_dim, hid_dim, out_dim, num_nodes, max_iter;
    bool training;
    /*
     * @brief Default constructor
     */
    DataSlice() : BaseDataSlice() {
//      loss.SetName ("loss");
      split.SetName ("split");
      in_feature.SetName ("in_feature");
      label.SetName ("label");
      penalty.SetName ("penalty");
      wrong.SetName ("wrong");
      cnt.SetName ("cnt");
      truth.SetName ("truth");
      theta.SetName("theta");
      SkXtheta.SetName("SkXtheta");
      SkXtheta_grad.SetName("SkXtheta_grad");
      theta_grad.SetName("theta_grad");
      penalty.SetName("penalty");
    }

    /*
     * @brief Default destructor
     */
    virtual ~DataSlice() { Release(); }

    /*
     * @brief Releasing allocated memory space
     * @param[in] target      The location to release memory from
     * \return    cudaError_t Error message(s), if any
     */
    cudaError_t Release(util::Location target = util::LOCATION_ALL) {
      cudaError_t retval = cudaSuccess;
      if (target & util::DEVICE) GUARD_CU(util::SetDevice(this->gpu_idx));

      GUARD_CU(BaseDataSlice ::Release(target));
      return retval;
    }

    /**
     * @brief initializing graphsum-specific data on each gpu
     * @param     sub_graph   Sub graph on the GPU.
     * @param[in] num_gpus    Number of GPUs
     * @param[in] gpu_idx     GPU device index
     * @param[in] target      Targeting device location
     * @param[in] flag        Problem flag containling options
     * \return    cudaError_t Error message(s), if any
     */
    cudaError_t Init(GraphT &sub_graph, util::Parameters &parameters, SpmatT &x, int *_truth,
                     int num_gpus = 1, int gpu_idx = 0,
                     util::Location target = util::DEVICE,
                     ProblemFlag flag = Problem_None) {
      cudaError_t retval = cudaSuccess;
      this->in_dim = parameters.Get<int>("in_dim");
      this->out_dim = parameters.Get<int>("out_dim");
      this->max_iter = parameters.Get<int>("max_iter");
      this->training = parameters.Get<bool>("training");
      this->learning_rate = parameters.Get<double>("learning_rate");
      this->num_nodes = sub_graph.nodes;
//      this->x_ptr = &x;

      GUARD_CU(BaseDataSlice::Init(sub_graph, num_gpus, gpu_idx, target, flag));

      GUARD_CU (in_feature.Allocate (x.edges))
      GUARD_CU (penalty.Allocate (1))
      GUARD_CU (wrong.Allocate (1))
      GUARD_CU (cnt.Allocate (1))
      GUARD_CU (label.Allocate (num_nodes))
      GUARD_CU (truth.Allocate (num_nodes))
      GUARD_CU (split.Allocate (num_nodes))
      GUARD_CU (cnt.Allocate (1))
      GUARD_CU (theta.Allocate (in_dim * out_dim))
      GUARD_CU (theta_grad.Allocate (in_dim * out_dim))
      GUARD_CU (SkXtheta.Allocate (sub_graph.nodes * out_dim))
      GUARD_CU (SkXtheta_grad.Allocate (sub_graph.nodes * out_dim))

      GUARD_CU (label.SetPointer(_truth, num_nodes, util::HOST))
      GUARD_CU (label.Move(util::HOST, util::DEVICE))
//      GUARD_CU (truth.Print ())


      curandCreateGenerator (&gen, CURAND_RNG_PSEUDO_XORWOW);
      curandGenerateUniformDouble(gen, theta.GetPointer(util::DEVICE), theta.GetSize ());

      ValueT range = sqrt (6.0 / (in_dim + hid_dim));
      GUARD_CU (theta.ForEach (
          [range]__host__ __device__(ValueT &x) {
            x = (x - 0.5) * range * 2;
          }
      ))

      Array *dummy = nullptr;
      modules.push_back(new sgconv(parameters, sub_graph, 1, x, theta, theta_grad, SkXtheta, SkXtheta_grad,
          in_dim, out_dim, &fw_sgconv, &bw_sgconv));
      modules.push_back(new cross_entropy(parameters, SkXtheta, SkXtheta_grad, truth, num_nodes, out_dim, &fw_loss));

//      x_val = static_cast<sprmul *>(modules[1])->problem->
//          data_slices[0][0].sub_graph[0].SpmatT::CsrT::edge_values;
//      static_cast<dropout*>(modules[0])->data = x_val;

//      GUARD_CU(x.edge_values.ForEach(in_feature,
//          []__host__ __device__(ValueT &src, ValueT &dst) {
//        dst = src;
//      }, x.edge_values.GetSize(), util::DEVICE))

      vars.emplace_back(theta, theta_grad, true);

      GUARD_CU(sub_graph.Move(util::HOST, target, this->stream));
      return retval;
    }  // Init

    /**
     * @brief Reset problem function. Must be called prior to each run.
     * @param[in] target      Targeting device location
     * \return    cudaError_t Error message(s), if any
     */
    cudaError_t Reset(util::Location target = util::DEVICE) {
      cudaError_t retval = cudaSuccess;
      return retval;
    }
  };  // DataSlice

  // Members
  // Set of data slices (one for each GPU)
  util::Array1D<SizeT, DataSlice> *data_slices;
  SpmatT in;
  int *_truth;

  // Methods

  /**
   * @brief graphsum default constructor
   */
  Problem(util::Parameters &_parameters, ProblemFlag _flag = Problem_None)
      : BaseProblem(_parameters, _flag), data_slices(NULL) {}

  /**
   * @brief graphsum default destructor
   */
  virtual ~Problem() { Release(); }

  /*
   * @brief Releasing allocated memory space
   * @param[in] target      The location to release memory from
   * \return    cudaError_t Error message(s), if any
   */
  cudaError_t Release(util::Location target = util::LOCATION_ALL) {
    cudaError_t retval = cudaSuccess;
    if (data_slices == NULL) return retval;
    for (int i = 0; i < this->num_gpus; i++)
      GUARD_CU(data_slices[i].Release(target));

    if ((target & util::HOST) != 0 &&
        data_slices[0].GetPointer(util::DEVICE) == NULL) {
      delete[] data_slices;
      data_slices = NULL;
    }
    GUARD_CU(BaseProblem::Release(target));
    return retval;
  }

  /**
   * \addtogroup PublicInterface
   * @{
   */


  cudaError_t Extract(ValueT *_w0, ValueT *_w1,
                      util::Location target = util::DEVICE) {
    cudaError_t retval = cudaSuccess;
    SizeT nodes = this->org_graph->nodes;

    if (this->num_gpus == 1) {
      auto &data_slice = data_slices[0][0];

      // Set device
      if (target == util::DEVICE) {
        GUARD_CU(util::SetDevice(this->gpu_idx[0]));
        GUARD_CU (data_slice.theta.SetPointer(_w0, data_slice.in_dim * data_slice.hid_dim))
        GUARD_CU (data_slice.theta.Move(util::DEVICE, util::HOST))
      }
    }

    return retval;
  }

  cudaError_t read_feature(Parameters &p, SpmatT &g, int *&truth) {
    typedef typename SpmatT::CsrT CsrT;

    auto retval = cudaSuccess;

    int n_rows = 0, nnz = 0, dim;
    static std::vector<typename GraphT::SizeT> indptr, indices;
    static std::vector<int> labels;
    static std::vector<ValueT> feature_val;
    indptr.push_back(0);
    std::ifstream svmlight_file(p.Get<std::string>("feature_file"));

    int max_idx = 0, max_label = 0;
    while(true) {
        std::string line;
        getline(svmlight_file, line);
        if (svmlight_file.eof()) break;
        indptr.push_back(indptr.back());
        std::istringstream ss(line);

        int label = -1;
        ss >> label;
        labels.push_back(label);
        if (ss.fail()) continue;
        max_label = std::max(max_label, label);

        while (true) {
            std::string kv;
            ss >> kv;
            if(ss.fail()) break;
            std::istringstream kv_ss(kv);

            int k;
            float v;
            char col;
            kv_ss >> k >> col >> v;

            feature_val.push_back(v);
            indices.push_back(k);
            indptr.back() += 1;
            max_idx = std::max(max_idx, k);
          }
      }
    n_rows = indptr.size() - 1;
    nnz = indices.size();
    dim = max_idx + 1;

    p.Set("in_dim", dim);
    p.Set("out_dim", max_label + 1);

    g.CsrT::Allocate(n_rows, nnz, gunrock::util::HOST);
    g.CsrT::row_offsets.SetPointer(indptr.data(), n_rows + 1, gunrock::util::HOST);
    g.CsrT::column_indices.SetPointer(indices.data(), nnz, gunrock::util::HOST);
    g.CsrT::edge_values.SetPointer(feature_val.data(), nnz, gunrock::util::HOST);
    g.nodes = n_rows;
//    std::cout << "n_rows: " << n_rows << '\n';
    g.edges = nnz;
    gunrock::graphio::LoadGraph(p, g);
//    g.Move(util::HOST, util::DEVICE);

//    for (auto e : labels) std::cout << e << '\n'; std::cout << '\n';
    truth = labels.data();

    return retval;
  }

  void get_split(Parameters &p, std::vector<int> &split) {
    std::ifstream split_file(p.Get<std::string>("split_file"));

    while (true) {
      std::string line;
      getline(split_file, line);
      if (split_file.eof()) break;
      int cur_split = stoi(line);
      split.push_back(cur_split);
    }
  }

  /**
   * @brief      initialization function.
   *
   * @param      graph   The graph that SSSP processes on
   * @param[in]  dim     The dimension of the feature vector
   * @param[in]  target  The target
   * @param[in]  Location  Memory location to work on
   *
   * @return     cudaError_t Error message(s), if any
   */
  cudaError_t Init(GraphT &graph, util::Location target = util::DEVICE) {
    cudaError_t retval = cudaSuccess;
    GUARD_CU(BaseProblem::Init(graph, target));
    data_slices = new util::Array1D<SizeT, DataSlice>[this->num_gpus];

    for (int gpu = 0; gpu < this->num_gpus; gpu++) {
      data_slices[gpu].SetName("data_slices[" + std::to_string(gpu) + "]");
      if (target & util::DEVICE) GUARD_CU(util::SetDevice(this->gpu_idx[gpu]));

      GUARD_CU(data_slices[gpu].Allocate(1, target | util::HOST));

      auto &data_slice = data_slices[gpu][0];

      read_feature (BaseProblem::parameters, in, _truth);
//      for (int i = 0; i < graph.nodes; i++) std::cout << _truth[i] << '\n';
//      std::cout << '\n';
      std::vector<int> split;
      get_split(BaseProblem::parameters, split);
      data_slice.Init(this->sub_graphs[gpu], BaseProblem::parameters, in, _truth, this->num_gpus,
                               this->gpu_idx[gpu], target, this->flag);
      GUARD_CU (data_slice.split.SetPointer(split.data (), graph.nodes, util::HOST))
      GUARD_CU (data_slice.split.Move(util::HOST, util::DEVICE))
    }  // end for (gpu)

    return retval;
  }

  /**
   * @brief Reset problem function. Must be called prior to each run.
   * @param[in] location Memory location to work on
   * \return cudaError_t Error message(s), if any
   */
  cudaError_t Reset(util::Location target = util::DEVICE) {
    cudaError_t retval = cudaSuccess;

    for (int gpu = 0; gpu < this->num_gpus; ++gpu) {
      // Set device
      if (target & util::DEVICE) GUARD_CU(util::SetDevice(this->gpu_idx[gpu]));
      GUARD_CU(data_slices[gpu]->Reset(target));
      GUARD_CU(data_slices[gpu].Move(util::HOST, target));
    }

    if (target & util::DEVICE) {
      GUARD_CU2(cudaDeviceSynchronize(), "cudaDeviceSynchronize failed");
    }

    return retval;
  }

  /** @} */
};

}  // namespace graphsum
}  // namespace app
}  // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
