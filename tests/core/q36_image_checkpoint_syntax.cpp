// Declaration-only host scaffold for actual new Engine methods/native oracle.
// This does NOT compile the real Engine/Serve/CUDA translation unit or link CUDA.
#include "strata/core/q36_vision_preallocation.hpp"
#include "strata/core/q36_image_checkpoint.hpp"
#include "strata/core/q36_checkpoint_schedule.hpp"
#include <atomic>
#include <chrono>
#include <functional>
#include <cstdio>
#include <sstream>
using Clock=std::chrono::steady_clock;
double ms_since(Clock::time_point);
int cudaMemcpyAsync(void*,const void*,size_t,int,void*);int cudaStreamSynchronize(void*);
constexpr int cudaMemcpyDeviceToHost=1,cudaMemcpyHostToDevice=2,cudaMemcpyDeviceToDevice=3;
#define CK(value) ((void)(value))
namespace strata::core {uint64_t unified_kv_available();struct DriverStats {uint64_t allocations,attempts,releases;};DriverStats vmm_allocation_stats();}
struct ImgKey {int64_t cell,n;uint64_t hash;};
struct Cache {int8_t *k,*v;uint16_t *ks,*vs;};
namespace q36kv8 {int copy_prefix(Cache&,const Cache&,int,uint64_t,int64_t,void*);}
struct Geometry {int64_t n_embd,ssm_conv_channels,ssm_d_conv,ssm_v_heads,ssm_state_size,n_head_kv,head_dim;};
struct Model {Geometry g;bool hybrid;std::vector<int> layers;};
struct Bank {strata::core::KvBackingInfo backing_info();void checkpoint_copy(uint64_t,uint64_t);};
struct Adapter {bool covers(uint64_t)const;};
struct Engine {
 struct Sess {
  int64_t pos;int32_t rdelta;bool active;std::vector<int32_t> live;std::vector<ImgKey> imgs;float* hlast;
  std::vector<float*> conv,S,kc,vc;std::vector<Cache> kv8;strata::core::Q36PrefixCheckpoints checkpoints;std::vector<float*> checkpoint_data;
  strata::core::Q36ImageSourcePtr image_source;std::array<strata::core::Q36ImagePoint,4> image_points{};
  strata::core::Q36ImagePlan image_plan;uint64_t image_ready_end;
  void clear_image_points(){image_source.reset();image_points={};image_plan={};image_ready_end=0;}
 };
 Model* m;std::vector<Sess> sess;int64_t pos;int cur,ctx,n_sess,mtp_d,tmax;bool image_checkpoints,vision,kv_int8,kv_protocol,img_active,map_on;void* st;
 mutable uint64_t image_metadata_us=0;
 uint64_t checkpoint_payload,checkpoint_last_reused,checkpoint_interval,vision_max_rows;size_t checkpoint_count,img_cap;
 strata::core::Q36PrefixCheckpoints::Policy checkpoint_policy;
 strata::core::Q36ScheduleBook* schedule_book;std::function<void(int)> schedule_restored;std::function<void(int,int)> schedule_before_restore;
 std::vector<uint64_t> kv_required;std::vector<strata::core::KvIdentity> kv_ids;
 std::unique_ptr<strata::core::UnifiedKvPool> kv_pool;std::unique_ptr<Bank> kv_bank;std::vector<std::unique_ptr<Adapter>> kv_adapters;
 float* d_img;int32_t *d_delta,*d_mtab,*d_map;const int32_t* mtab_now;int64_t img_prompt_cells;std::vector<int32_t> img_map;strata::core::Q36VisionLease vision_lease;
 void set_delta(int32_t);void select(int);void reset();void clear_images();void host_logits(std::vector<float>&);void kv_ensure(int,uint64_t);void kv_progress(int,uint64_t);void kv_begin(int,uint64_t,bool);void kv_end(int);
 void set_images(const std::vector<float>&,const std::vector<int32_t>&,const std::vector<int32_t>&,int64_t,int32_t);
 template<class F> int64_t process(const int32_t*,int64_t,F,bool=false){return 0;}
#include "q36_restore_observation.inc"
#include "q36_vision_staging.inc"
#include IMAGE_ENGINE_INC
};
struct ImgReq {std::vector<float> rows;std::vector<ImgKey> imgs;std::vector<int32_t> cell_row,mrope;int32_t delta;};
std::vector<int32_t> parse_ids(const char*);
std::string build_images(const std::string&,const std::vector<int32_t>&,int64_t,const std::vector<uint64_t>&,ImgReq&);
uint64_t fnv1a(const void*,size_t,uint64_t);
struct Serve {int n_active;double schedule_prefill_ms;std::atomic<bool> stop;std::function<void(int64_t)> vision_test_on_chunk;explicit Serve(Engine&);void request_body(const std::string&);};
struct GateDump {explicit GateDump(const std::string&);template<class T> void vector(const std::vector<T>&) {}};
struct SchedulePlane {std::string name;std::vector<uint8_t> bytes;};using ScheduleState=std::vector<SchedulePlane>;
ScheduleState schedule_state(Engine&,int,const strata::core::Q36Schedule&);bool schedule_compare(const ScheduleState&,const ScheduleState&,const char*);bool schedule_logits(const std::vector<float>&,const std::vector<float>&,const char*);
namespace vision_gate {bool views_idle(Engine&);}
#include "q36_image_checkpoint_regression.inc"
