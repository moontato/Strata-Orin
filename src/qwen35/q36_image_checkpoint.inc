// Inside Engine. Identity proves only input semantics; availability still owns
// residency/generation/epoch/backing/donor-pin/destination-lease authorization.
bool image_prefix_equal(const strata::core::Q36ImageSource& held,const strata::core::Q36ImageSource& incoming,uint64_t end) const {
    strata::core::Q36ImageHostTimer timer(image_metadata_us);
    return strata::core::q36_image_prefix(held,incoming,end);
}
int image_checkpoint_match(int s,const std::vector<int32_t>& ids,const strata::core::Q36ImageSource& incoming) const {
    if(!image_checkpoints || !checkpoint_count) return -1;
    strata::core::Q36ImageHostTimer timer(image_metadata_us);
    const auto& ss=sess[(size_t)s];const auto owner=kv_pool->session((uint32_t)s);
    const auto common=strata::core::q36_image_common(strata::core::q36_image_source(ss.image_source),incoming,checkpoint_common(s,ids));
    return ss.checkpoints.best_available(common,ids.size(),owner.id.generation,owner.epoch,[&](int i) {
        const auto p=ss.checkpoints.points[(size_t)i];const auto& proof=ss.image_points[(size_t)i];
        return checkpoint_available(s,i,nullptr,nullptr,true) &&
            strata::core::q36_image_prefix(strata::core::q36_image_source(proof.source),incoming,p.pos);
    });
}
strata::core::Q36PrefixCheckpoints::Learning image_checkpoint_learning(const size_t* common_tokens) {
    strata::core::Q36PrefixCheckpoints::Learning result;
    if(checkpoint_policy!=strata::core::Q36PrefixCheckpoints::Policy::adaptive) return result;
    size_t longest=0;int source=-1;
    for(int i=0;i<n_sess;++i) {
        const auto& ss=sess[(size_t)i];const auto owner=kv_pool->session((uint32_t)i);
        if(ss.active || owner.state!=strata::core::KvResidency::idle || !kv_adapters[(size_t)i]->covers(ss.live.size())) continue;
        const auto common=common_tokens[i];
        if(common && common==ss.live.size()) return ss.checkpoints.learning;
        if(common>longest) {longest=common;source=i;}
    }
    if(source>=0) {result=sess[(size_t)source].checkpoints.learning;result.observe(longest,sess[(size_t)source].live.size(),(uint64_t)tmax);}
    return result;
}
void image_checkpoint_capture() {
    auto& ss=sess[(size_t)cur];const auto& source=strata::core::q36_image_source(ss.image_source);
    if(!image_checkpoints || ss.image_ready_end!=(uint64_t)pos || !source.boundary((uint64_t)pos)) return;
    const auto owner=kv_pool->session((uint32_t)cur);
    if(owner.state!=strata::core::KvResidency::active || !kv_adapters[(size_t)cur]->covers((uint64_t)pos))
        throw std::logic_error("image checkpoint capture without ready active backing");
    vision_lease.check(cur,owner.id.generation,owner.epoch);
    const int index=ss.image_plan.placement(ss.checkpoints,(uint64_t)pos,source);
    if(index>=0) {
        const strata::core::Q36ImagePoint proof{ss.image_source,(uint64_t)pos,source.delta((uint64_t)pos),true};
        checkpoint_capture(index,&proof);
    }
}
