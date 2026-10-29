// Declaration-only stand-ins: NOT an actual CUDA/Engine/Serve TU build.
#include "strata/core/q36_vision_preallocation.hpp"
#include "strata/core/q36_prefix_checkpoint.hpp"
#include <atomic>
#include <functional>
#include <memory>
#include <cstring>
#include <cstdio>
int cudaMemcpyAsync(void*,const void*,size_t,int,void*);int cudaStreamSynchronize(void*);
constexpr int cudaMemcpyDeviceToHost=1,cudaMemcpyHostToDevice=2;
#define CK(value) ((void)(value))
namespace strata::core {
uint64_t unified_kv_available();
struct DriverStats {uint64_t allocations,attempts,releases;};DriverStats vmm_allocation_stats();
}
struct Geometry {int64_t n_embd,ssm_conv_channels,ssm_d_conv,ssm_v_heads,ssm_state_size,n_head_kv,head_dim;};
struct Model {Geometry g;bool hybrid;};
struct ImgKey {int64_t cell,n;uint64_t hash;};
struct Bank {strata::core::KvBackingInfo backing_info();};
struct Cache {int8_t *k,*v;uint16_t *ks,*vs;};
struct Engine {
    struct Sess {int64_t pos;int32_t rdelta;std::vector<int32_t> live;std::vector<ImgKey> imgs;float* hlast;
        std::vector<float*> conv,S,kc,vc;std::vector<Cache> kv8;strata::core::Q36PrefixCheckpoints checkpoints;};
    Model* m;int64_t pos;int cur,ctx,n_sess,mtp_d,tmax;bool vision,kv_int8,img_active,map_on;void* st;
    std::vector<Sess> sess;std::vector<uint64_t> kv_required;std::vector<strata::core::KvIdentity> kv_ids;
    std::unique_ptr<strata::core::UnifiedKvPool> kv_pool;std::unique_ptr<Bank> kv_bank;
    float *d_img,*logits_mr;int32_t *d_mtab,*d_map,*pick_mr;const int32_t* mtab_now;
    size_t img_cap,checkpoint_count;int64_t img_prompt_cells;std::vector<int32_t> img_map;strata::core::Q36VisionLease vision_lease;
    void set_delta(int32_t);void select(int);void reset();void clear_images();void host_logits(std::vector<float>&);
#include "q36_vision_staging.inc"
};
struct ImgReq {std::vector<float> rows;std::vector<ImgKey> imgs;std::vector<int32_t> cell_row,mrope;int32_t delta;};
std::vector<int32_t> parse_ids(const char*);
std::string build_images(const std::string&,const std::vector<int32_t>&,int64_t,const std::vector<uint64_t>&,ImgReq&);
struct Serve {int n_active,read_windows;struct Slot{bool stop;};std::vector<Slot> slots;std::atomic<bool> stop;
    std::function<void(int64_t)> vision_test_on_chunk;explicit Serve(Engine&);void request_body(const std::string&);void window();};
struct GateDump {explicit GateDump(const std::string&);template<class T> void vector(const std::vector<T>&) {}};
#define STRATA_Q36_VISION_PREALLOCATION_IMPL 1
#include "q36_vision_regression.inc"
