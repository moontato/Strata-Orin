// Inside Serve. Read-only optional observations, never a matching decision.
struct CacheImageObservation {
    uint64_t generation=0,epoch=0;
    std::vector<ImgKey> images;
    std::vector<strata::core::Q36CacheImageDetail> detail;
};
std::array<CacheImageObservation,Engine::kVerify> cache_observed;
std::vector<strata::core::Q36CacheImageDetail> cache_incoming;
uint64_t cache_request=0;
void cache_diag_before(bool adm,bool geni,int destination,const std::vector<int32_t>& ids,const ImgReq& ir,
                       const std::string& emb,const std::vector<uint64_t>& keys) noexcept {
    try {
        ++cache_request;cache_incoming.clear();
        const bool detail=geni && strata::core::q36_cache_image_details(emb,ir.imgs,ir.rows,(uint64_t)E.m->g.n_embd,keys,cache_incoming);
        if(!detail) cache_incoming.clear();
        int eligible=0,matched=0;
        for(int i=0;i<E.n_sess && i<Engine::kVerify;++i) {
            const auto& ss=E.sess[(size_t)i];
            const auto owner=E.kv_pool ? E.kv_pool->session((uint32_t)i) : strata::core::KvSessionInfo{};
            const bool backed=!E.kv_pool || E.kv_adapters[(size_t)i]->covers(ss.live.size());
            const bool pick=!ss.active && (!E.kv_pool || (owner.state!=strata::core::KvResidency::active && owner.state!=strata::core::KvResidency::faulted));
            const bool idle=!ss.active && !ss.live.empty() && backed && (!E.kv_pool || owner.state==strata::core::KvResidency::idle);
            const auto& observed=cache_observed[(size_t)i];
            const bool known=observed.generation==owner.id.generation && observed.epoch==owner.epoch && observed.images==ss.imgs && observed.detail.size()==ss.imgs.size();
            const auto d=strata::core::q36_cache_reason(ss.live,ids,ss.imgs,ir.imgs,known?&observed.detail:nullptr,detail?&cache_incoming:nullptr);
            const size_t actual=match_len(ss,ids,ir.imgs);eligible+=idle?1:0;matched+=(pick && actual)?1:0;
            int32_t incoming_delta=0;
            const bool delta_known=!ir.mrope.empty() && ss.live.size()<=ir.mrope.size()/3 && d.complete;
            if(delta_known) incoming_delta=strata::core::q36_vision_prefix_delta(ir.mrope,ss.live.size());
            const auto text=strata::core::q36_cache_line(cache_request,i,d,actual,E.kv_pool?strata::core::kv_residency_name(owner.state):"unmanaged",
                owner.id.generation,owner.epoch,ss.pos,ss.active,backed,pick,idle,ss.rdelta,incoming_delta,delta_known);
            std::fputs(text.c_str(),stderr);
        }
        std::fprintf(stderr,"Q36CACHE search request=%llu mode=%s explicit_destination=%d incoming=%zu images=%zu observed_slots=%d total_slots=%d idle_backed_candidates=%d exact_candidates=%d checkpoint_fallback=%s detail_source=%s\n",
            (unsigned long long)cache_request,adm?"explicit_batch":"engine_selected",destination,ids.size(),ir.imgs.size(),std::min(E.n_sess,Engine::kVerify),E.n_sess,eligible,matched,
            geni?(E.image_checkpoints?"image_optin_enabled":"image_request_bypass"):E.checkpoint_count?"text_enabled":"disabled",detail?"validated_rows_header_reread":"unavailable");
    } catch(...) {cache_incoming.clear();std::fprintf(stderr,"Q36CACHE unavailable stage=search\n");}
}
void cache_diag_selected(int slot) const noexcept {
    std::fprintf(stderr,"Q36CACHE selected request=%llu slot=%d\n",(unsigned long long)cache_request,slot);
}
void cache_diag_resolved(int slot,size_t reuse,int checkpoint_source,bool restored) const noexcept {
    std::fprintf(stderr,"Q36CACHE resolved request=%llu slot=%d reused=%zu checkpoint_source=%d checkpoint_restored=%d\n",
        (unsigned long long)cache_request,slot,reuse,checkpoint_source,restored?1:0);
}
void cache_diag_retained(int slot) noexcept {
    try {
        if(slot<0 || slot>=Engine::kVerify || E.sess[(size_t)slot].imgs.size()>1024) {
            std::fprintf(stderr,"Q36CACHE unavailable stage=retained reason=diagnostic_bound\n");return;
        }
        const auto& ss=E.sess[(size_t)slot];const auto owner=E.kv_pool?E.kv_pool->session((uint32_t)slot):strata::core::KvSessionInfo{};
        auto& dst=cache_observed[(size_t)slot];dst.generation=owner.id.generation;dst.epoch=owner.epoch;
        dst.images=ss.imgs;dst.detail.clear();
        for(const auto& k:ss.imgs) {
            auto it=std::find_if(cache_incoming.begin(),cache_incoming.end(),[&](const auto& v){return v.cell==k.cell && v.n==k.n && v.identity==k.hash;});
            if(it==cache_incoming.end()) {dst.detail.clear();break;}
            dst.detail.push_back(*it);
        }
        std::fprintf(stderr,"Q36CACHE retained request=%llu slot=%d tokens=%zu position=%lld images=%zu delta=%d generation=%llu epoch=%llu\n",
            (unsigned long long)cache_request,slot,ss.live.size(),(long long)ss.pos,ss.imgs.size(),ss.rdelta,
            (unsigned long long)owner.id.generation,(unsigned long long)owner.epoch);
    } catch(...) {std::fprintf(stderr,"Q36CACHE unavailable stage=retained\n");}
}
