// Inside Engine; only strict --kv-preallocate --vision enters these methods.
void release_vision_views() noexcept {
    if(!kv_bank || !vision) return;
    img_map.clear();img_active=false;img_prompt_cells=0;mtab_now=nullptr;map_on=false;vision_lease.release();
    // Do NOT reset rdelta: each slot's decode continues after its own images.
    // Buffers stay allocated even if T=1 work is still queued. Errors quiesce/
    // reset the admission, and set_images always drains before any new upload.
}
struct VisionPrefillScope {
    Engine& engine;bool enabled;
    VisionPrefillScope(Engine& e,bool on):engine(e),enabled(on) {}
    ~VisionPrefillScope(){if(enabled) engine.release_vision_views();}
    void finish(){if(enabled) engine.release_vision_views();enabled=false;}
};
void strict_set_images(const std::vector<float>& rows,const std::vector<int32_t>& cell_row,
                       const std::vector<int32_t>& mrope,int64_t prompt,int32_t delta) {
    if(!vision || !d_img || !d_mtab || !d_map || prompt<=0 || prompt>ctx ||
       rows.empty() || rows.size()%(size_t)m->g.n_embd || rows.size()>img_cap ||
       cell_row.size()!=(size_t)prompt || mrope.size()!=(size_t)prompt*3)
        throw strata::core::KvFailure({strata::core::KvCode::context_limit,"vision input exceeds fixed staging geometry"});
    if(!strata::core::unified_kv_available())
        throw strata::core::KvFailure({strata::core::KvCode::pool_full,"vision request has no fresh memory headroom (encoder is external to KV bank)"});
    const auto owner=kv_pool->session((uint32_t)cur);
    if(owner.state!=strata::core::KvResidency::active)
        throw strata::core::KvFailure({strata::core::KvCode::backend,"vision prefill has no active KV lease"});
    vision_lease.acquire(cur,owner.id.generation,owner.epoch);
    CK(cudaStreamSynchronize(st));
    CK(cudaMemcpyAsync(d_img,rows.data(),rows.size()*4,cudaMemcpyHostToDevice,st));
    CK(cudaMemcpyAsync(d_mtab,mrope.data(),mrope.size()*4,cudaMemcpyHostToDevice,st));
    CK(cudaStreamSynchronize(st));
    img_map=cell_row;img_prompt_cells=prompt;img_active=true;set_delta(delta);
}
