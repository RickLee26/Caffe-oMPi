#include <glog/logging.h>
#include <stdio.h>
#include <sstream>
#include <string>
#include <vector>
#include <omp.h>

#include "caffe/caffe.hpp"
#include "caffe/parallel.hpp"
#include "caffe/sgd_solvers.hpp"
#include "caffe/util/format.hpp"

#ifdef USE_NCCL
#include <cuda_runtime.h>
#endif

#ifdef USE_OMPI
#include "mpi.h"
#endif

namespace caffe {

enum Op {
  copy,
  replace_cpu,
  replace_gpu,
  replace_cpu_diff,
  replace_gpu_diff
};

template<typename Dtype>
static void apply_buffers(const vector<Blob<Dtype>*>& blobs,
                          Dtype* buffer, size_t total_size, Op op) {
  Dtype* ptr = buffer;
  for (int i = 0; i < blobs.size(); ++i) {
    int size = blobs[i]->count();
    switch (op) {
      case copy: {
        // Init buffer to current values of blobs
        caffe_copy(size,
                   reinterpret_cast<const Dtype*>(blobs[i]->data()->cpu_data()),
                   ptr);
        break;
      }
      case replace_cpu:
        blobs[i]->data()->set_cpu_data(ptr);
        break;
      case replace_gpu:
        blobs[i]->data()->set_gpu_data(ptr);
        break;
      case replace_cpu_diff:
        blobs[i]->diff()->set_cpu_data(ptr);
        break;
      case replace_gpu_diff:
        blobs[i]->diff()->set_gpu_data(ptr);
        break;
    }
    ptr += size;
  }
  // total_size is at least one byte
  CHECK_EQ(total_size, (ptr == buffer ? 1 : ptr - buffer));
}

// Buffer size necessary to store given blobs
template<typename Dtype>
static size_t total_size(const vector<Blob<Dtype>*>& params) {
  size_t size = 0;
  for (int i = 0; i < params.size(); ++i)
    size += params[i]->count();
  // Size have at least one byte, otherwise cudaMalloc fails if net has no
  // learnable parameters.
  return (size > 0) ? size : 1;
}

template<typename Dtype>
Params<Dtype>::Params(shared_ptr<Solver<Dtype> > root_solver)
  : size_(total_size<Dtype>(root_solver->net()->learnable_params())),
    data_(),
    diff_() {
}
INSTANTIATE_CLASS(Params);

template<typename Dtype>
ThreadLocal<Dtype>::ThreadLocal(shared_ptr<Solver<Dtype> > solver)
  : Params<Dtype>(solver), solver_(solver) {
  data_ = (Dtype*)malloc(size_ * sizeof(Dtype));
  diff_ = (Dtype*)malloc(size_ * sizeof(Dtype));
  if(Caffe::root_thread())buffer_ = (Dtype*)malloc(size_ * sizeof(Dtype));

  const vector<Blob<Dtype>*>& net = solver->net()->learnable_params();
  apply_buffers(net, data_, size_, copy);
  apply_buffers(net, data_, size_, replace_cpu);
  apply_buffers(net, diff_, size_, replace_cpu_diff);

  solver_->add_callback(this);
  if (solver_->param().layer_wise_reduce()) {
    solver_->net()->add_after_backward(this);
  }
}

template<typename Dtype>
void ThreadLocal<Dtype>::avgDiff(int offset, int size) {
  int count = Caffe::thread_count();
  int rank = Caffe::thread_rank();

  int step = size / count;
  if(step * count < size) step++;

  Dtype *src = (*locals_)[0]->diff_ + offset + rank * step;
  int len = size - rank * step;
  len = len > step ? step : len;

#pragma omp barrier
  for(int k = 1; k < count; k++){
    Dtype *target = (*locals_)[k]->diff_ + offset + rank * step;
    caffe_add(len, target, src, src);
  }
#pragma omp barrier

#ifdef USE_OMPI
  if(Caffe::root_thread() && Caffe::mpi_count() > 1){
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Datatype type = sizeof(Dtype) == 4 ? MPI_FLOAT : MPI_DOUBLE;
    MPI_Allreduce(src, buffer_, size, type, MPI_SUM, MPI_COMM_WORLD);
    caffe_copy(size, buffer_, src);
  }
#endif

#pragma omp barrier
  caffe_scal(len, (Dtype) 1.0 / Caffe::solver_count(), src);
  for(int k = 1; k < count; k++){
    Dtype *target = (*locals_)[k]->diff_ + offset + rank * step;
    caffe_copy(len, src, target);
  }
#pragma omp barrier
}


template<typename Dtype>
void ThreadLocal<Dtype>::syncData() {
  caffe_copy(size_, (*locals_)[0]->data_, data_);
}

template<typename Dtype>
void ThreadLocal<Dtype>::syncNodeData() {
#ifdef USE_OMPI
  if(Caffe::mpi_count() < 2)return;
  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Datatype type = sizeof(Dtype) == 4 ? MPI_FLOAT : MPI_DOUBLE;
  MPI_Bcast(data_, size_, type, 0, MPI_COMM_WORLD);
#endif
}

template<typename Dtype>
void ThreadLocal<Dtype>::run(int layer) {
  vector<shared_ptr<Blob<Dtype> > >& blobs = solver_->net()->layers()[layer]->blobs();
  if (blobs.size()){
    int size = 0;
    for(int i = 0; i < blobs.size(); i++) {
      size += blobs[i]->count();
    }
    int offset = blobs[0]->mutable_cpu_diff() - diff_;
    avgDiff(offset, size);
  }
}

template<typename Dtype>
void ThreadLocal<Dtype>::on_gradients_ready() {
  if(solver_->param().layer_wise_reduce()) {
      CHECK_EQ(solver_->net()->params().size(),solver_->net()->learnable_params().size())
        << "Layer-wise reduce is not supported for nets with shared weights.";
  } else {
    avgDiff(0, size_);
  }
}

template<typename Dtype>
void ThreadLocal<Dtype>::on_test(Dtype &loss) {
  int rank = Caffe::thread_rank();
  int count = Caffe::thread_count();
  loss_ = loss;
  vector<Dtype> &svec = (*locals_)[0]->solver_->score_vec();
  Dtype &sumloss = (*locals_)[0]->loss_;

#pragma omp barrier
  int step = svec.size() / count;
  if(step * count < svec.size())step++;

  int len = svec.size() - rank * step;
  len = len > step ? step : len;

  for(int k = 1; k < count; k++){
    if (rank + 1 == count) sumloss += (*locals_)[k]->loss_;
    vector<Dtype> &vec = (*locals_)[k]->solver_->score_vec();
    for(int i = 0; i < len; i++){
      svec[rank * step + i] += vec[rank * step + i];
    }
  }
#pragma omp barrier
#ifdef USE_OMPI
  if(Caffe::root_thread() && Caffe::mpi_count() > 1){
    int data_size = svec.size();
    svec.push_back(sumloss);
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Datatype type = sizeof(Dtype) == 4 ? MPI_FLOAT : MPI_DOUBLE;
    MPI_Allreduce(svec.data(), buffer_, svec.size(), type, MPI_SUM, MPI_COMM_WORLD);
    svec.erase(svec.begin() + data_size);
    sumloss = buffer_[data_size];
    caffe_copy(data_size, buffer_, svec.data());
  }
#endif
  loss = sumloss;
}

template<typename Dtype>
void ThreadLocal<Dtype>::Run() {
  Caffe::OMPI ompi = Caffe::ompi();
  int thread_count = ompi.thread_count;
  vector<ThreadLocal<Dtype>* > locals(thread_count);
  shared_ptr<Solver<Dtype> > rank0 = solver_;

#pragma omp parallel num_threads(thread_count)
  {
      Caffe::set_ompi(ompi);
      int rank = omp_get_thread_num();
      Caffe::set_solver_rank(rank + ompi.mpi_rank * thread_count);
      Caffe::set_solver_count(ompi.mpi_count * thread_count);
      ThreadLocal<Dtype> *local = this;

      if(rank){
        SolverParameter param(rank0->param());
        param.set_type(rank0->type());
        shared_ptr<Solver<Dtype> > s(SolverRegistry<Dtype>::CreateSolver(param));
        local = new ThreadLocal<Dtype>(s);
      }
      locals[rank] = local;
      local->locals_ = &locals;
#pragma omp barrier
      if(!rank){
        local->syncNodeData();
      }
#pragma omp barrier
      if(rank){
        local->syncData();
      }
#pragma omp barrier
      local->solver_->Solve();
  }

}

INSTANTIATE_CLASS(ThreadLocal);


#ifdef USE_NCCL
  template<typename Dtype>
  GPUParams<Dtype>::GPUParams(shared_ptr<Solver<Dtype> > root_solver, int device)
    : Params<Dtype>(root_solver) {
    int initial_device;
    CUDA_CHECK(cudaGetDevice(&initial_device));

    // Allocate device buffers
    CUDA_CHECK(cudaSetDevice(device));
    CUDA_CHECK(cudaMalloc(&data_, size_ * sizeof(Dtype)));

    // Copy blob values
    const vector<Blob<Dtype>*>& net =
      root_solver->net()->learnable_params();
    apply_buffers(net, data_, size_, copy);

    CUDA_CHECK(cudaMalloc(&diff_, size_ * sizeof(Dtype)));
    caffe_gpu_set(size_, Dtype(0), diff_);

    CUDA_CHECK(cudaSetDevice(initial_device));
  }

  template<typename Dtype>
  GPUParams<Dtype>::~GPUParams() {
    CUDA_CHECK(cudaFree(data_));
    CUDA_CHECK(cudaFree(diff_));
  }

  template<typename Dtype>
  void GPUParams<Dtype>::Configure(Solver<Dtype>* solver) const {
    const vector<Blob<Dtype>*>& net =
      solver->net()->learnable_params();
    apply_buffers(net, data_, size_, replace_gpu);
    apply_buffers(net, diff_, size_, replace_gpu_diff);
  }

  static int getDevice() {
    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    return device;
  }

  template<typename Dtype>
  NCCL<Dtype>::NCCL(shared_ptr<Solver<Dtype> > solver)
    : GPUParams<Dtype>(solver, getDevice()),
      comm_(), solver_(solver), barrier_() {
    this->Configure(solver.get());
    Init();
  }

  template<typename Dtype>
  NCCL<Dtype>::NCCL(shared_ptr<Solver<Dtype> > solver, const string& uid)
    : GPUParams<Dtype>(solver, getDevice()),
      solver_(solver), barrier_() {
    this->Configure(solver.get());
    Caffe::set_multiprocess(true);
    ncclUniqueId nccl_uid;
    memcpy(&nccl_uid, &uid[0], NCCL_UNIQUE_ID_BYTES);  // NOLINT(caffe/alt_fn)
    NCCL_CHECK(ncclCommInitRank(&comm_,
                                Caffe::solver_count(),
                                nccl_uid,
                                Caffe::solver_rank()));
    Init();
  }

  template<typename Dtype>
  void NCCL<Dtype>::Init() {
    if (solver_->param().layer_wise_reduce()) {
      CUDA_CHECK(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking));
    }
  }

  template<typename Dtype>
  NCCL<Dtype>::~NCCL() {
    if (solver_->param().layer_wise_reduce()) {
      CUDA_CHECK(cudaStreamDestroy(stream_));
    }
    if (comm_) {
      ncclCommDestroy(comm_);
    }
  }

  template<typename Dtype>
  boost::barrier* NCCL<Dtype>::barrier() {
    return barrier_;
  }
  template<typename Dtype>
  void NCCL<Dtype>::set_barrier(boost::barrier* value) {
    barrier_ = value;
  }

  template<typename Dtype>
  void NCCL<Dtype>::InitSingleProcess(vector<NCCL<Dtype>*>* nccls) {
    ncclComm_t* comms = new ncclComm_t[nccls->size()];
    int* gpu_list = new int[nccls->size()];
    for (int i = 0; i < nccls->size(); ++i) {
      gpu_list[i] = (*nccls)[i]->solver_->param().device_id();
    }
    NCCL_CHECK(ncclCommInitAll(comms, static_cast<int>(nccls->size()), gpu_list));
    for (int i = 0; i < nccls->size(); ++i) {
      (*nccls)[i]->comm_ = comms[i];
    }
  }

  template<typename Dtype>
  string NCCL<Dtype>::new_uid() {
    string uid;
    uid.resize(NCCL_UNIQUE_ID_BYTES);
    ncclUniqueId nccl_uid;
    NCCL_CHECK(ncclGetUniqueId(&nccl_uid));
    memcpy(&uid[0], &nccl_uid, NCCL_UNIQUE_ID_BYTES);  // NOLINT(caffe/alt_fn)
    return uid;
  }

  template<typename Dtype>
  void NCCL<Dtype>::Broadcast() {
    if (barrier_) {  // NULL in multi process case
      barrier_->wait();
    }
    NCCL_CHECK(ncclBcast(data_, static_cast<int>(size_),
                         nccl::dataType<Dtype>::type, 0,
                         comm_, cudaStreamDefault));
    if (barrier_) {
      barrier_->wait();
    }
  }

  template<typename Dtype>
  void NCCL<Dtype>::run(int layer) {
    CHECK(solver_->param().layer_wise_reduce());
    vector<shared_ptr<Blob<Dtype> > >& blobs =
      solver_->net()->layers()[layer]->blobs();
  #ifdef DEBUG
    // Assert blobs are contiguous to reduce in one step (e.g. bias often small)
    for (int i = 1; i < blobs.size(); ++i) {
      CHECK_EQ(blobs[i - 1]->gpu_diff() + blobs[i - 1]->count(),
               blobs[i + 0]->gpu_diff());
    }
  #endif
    if (blobs.size() > 0) {
      // Make sure default stream is done computing gradients. Could be
      // replaced by cudaEventRecord+cudaStreamWaitEvent to avoid
      // blocking the default stream, but it's actually slower.
      CUDA_CHECK(cudaStreamSynchronize(cudaStreamDefault));

      // Reduce asynchronously
      int size = 0;
      for (int i = 0; i < blobs.size(); ++i) {
        size += blobs[i]->count();
      }
      if (barrier_) {  // NULL in multi process case
        barrier_->wait();
      }
      NCCL_CHECK(ncclAllReduce(blobs[0]->mutable_gpu_diff(),
                               blobs[0]->mutable_gpu_diff(),
                               size,
                               nccl::dataType<Dtype>::type,
                               ncclSum, comm_, stream_));
      caffe_gpu_scal(size, (Dtype) 1.0 / Caffe::solver_count(),
                     blobs[0]->mutable_gpu_diff(), stream_);
    }
  }

  template<typename Dtype>
  void NCCL<Dtype>::on_gradients_ready() {
    if (solver_->param().layer_wise_reduce()) {
      CHECK_EQ(solver_->net()->params().size(),
               solver_->net()->learnable_params().size())
        << "Layer-wise reduce is not supported for nets with shared weights.";

      // Make sure reduction is done before applying gradients
      CUDA_CHECK(cudaStreamSynchronize(stream_));
    } else {
      if (barrier_) {  // NULL in multi process case
        barrier_->wait();
      }
      NCCL_CHECK(ncclAllReduce(diff_, diff_, static_cast<int>(size_),
                               nccl::dataType<Dtype>::type, ncclSum, comm_,
                               cudaStreamDefault));
      caffe_gpu_scal(static_cast<int>(size_),
                     (Dtype) 1.0 / Caffe::solver_count(), diff_);
    }
  }

  template<typename Dtype>
  class Worker : public InternalThread {
   public:
    explicit Worker(shared_ptr<Solver<Dtype> > rank0, int device,
                    boost::barrier* barrier, vector<NCCL<Dtype>*>* nccls,
                    const char* restore)
      : rank0_(rank0), device_(device), barrier_(barrier),
        nccls_(nccls), restore_(restore) {
    }
    virtual ~Worker() {}

   protected:
    void InternalThreadEntry() {
      // Create solver and install callbacks
      SolverParameter param(rank0_->param());
      param.set_device_id(device_);
  #ifdef DEBUG
      int device;
      CUDA_CHECK(cudaGetDevice(&device));
      CHECK_EQ(device, device_);
  #endif
      param.set_type(rank0_->type());
      shared_ptr<Solver<Dtype> > s(SolverRegistry<Dtype>::CreateSolver(param));
      CHECK_EQ(s->type(), rank0_->type());
      if (restore_) {
        // Could not make NCCL broadcast solver state, it seems to crash
        // if called in a tight loop, regardless of barriers etc. so
        // restore all solvers from file.
        s->Restore(restore_);
      }
      NCCL<Dtype> nccl(s);
      nccl.set_barrier(barrier_);
      s->add_callback(&nccl);
      if (s->param().layer_wise_reduce()) {
        s->net()->add_after_backward(&nccl);
      }
      (*nccls_)[Caffe::solver_rank()] = &nccl;
      // Wait for other threads
      barrier_->wait();
      // Wait for NCCL init
      barrier_->wait();
      // Broadcast rank 0 state
      nccl.Broadcast();
      // Solve
      s->Step(param.max_iter() - s->iter());
      barrier_->wait();
  #ifdef DEBUG
      // Check all solvers have same state
      SGDSolver<Dtype>* sa = static_cast<SGDSolver<Dtype>*>(rank0_.get());
      SGDSolver<Dtype>* sb = static_cast<SGDSolver<Dtype>*>(s.get());
      for (int h = 0; h < sa->history().size(); ++h) {
        CUDA_CHECK(cudaSetDevice(sa->param().device_id()));
        const Dtype* a = sa->history()[h]->cpu_data();
        CUDA_CHECK(cudaSetDevice(sb->param().device_id()));
        const Dtype* b = sb->history()[h]->cpu_data();
        for (int v = 0; v < sa->history()[h]->count(); ++v) {
          CHECK_DOUBLE_EQ(a[v], b[v]);
        }
      }
  #endif
    }

    shared_ptr<Solver<Dtype> > rank0_;
    int device_;
    boost::barrier* barrier_;
    vector<NCCL<Dtype>*>* nccls_;
    const char* restore_;
  };

  template<typename Dtype>
  void NCCL<Dtype>::Run(const vector<int>& gpus, const char* restore) {
    boost::barrier barrier(static_cast<int>(gpus.size()));
    vector<NCCL<Dtype>*> nccls(gpus.size());
    // Create workers
    vector<shared_ptr<Worker<Dtype> > > workers(gpus.size());
    for (int i = 1; i < gpus.size(); ++i) {
      CUDA_CHECK(cudaSetDevice(gpus[i]));
      Caffe::set_solver_rank(i);
      Worker<Dtype>* w = new Worker<Dtype>(solver_, gpus[i], &barrier,
                                           &nccls, restore);
      w->StartInternalThread();
      workers[i].reset(w);
    }
    CUDA_CHECK(cudaSetDevice(gpus[0]));
    Caffe::set_solver_rank(0);
    barrier_ = &barrier;
    solver_->add_callback(this);
    if (solver_->param().layer_wise_reduce()) {
      solver_->net()->add_after_backward(this);
    }
    nccls[0] = this;
    // Wait for workers
    barrier.wait();
    // Init NCCL
    InitSingleProcess(&nccls);
    barrier.wait();
    // Run first solver on current thread
    Broadcast();
    solver_->Solve();
    barrier.wait();  // Hangs without it when running tests
    // Wait for shutdown
    for (int i = 1; i < gpus.size(); ++i) {
      workers[i]->StopInternalThread();
    }
  }

  INSTANTIATE_CLASS(GPUParams);
  INSTANTIATE_CLASS(Worker);
  INSTANTIATE_CLASS(NCCL);
#endif  // USE_NCCL


}  // namespace caffe

