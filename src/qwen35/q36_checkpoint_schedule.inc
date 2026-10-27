// Developer-only independent native oracle. Host snapshots are comparison data;
// they are NEVER uploaded into the reference. No fifth GPU checkpoint payload.
struct SchedulePlane {
    std::string name;int layer=-1,head=-1;uint64_t cell=0;bool fp32=false;
    std::vector<uint8_t> bytes;
};
using ScheduleState=std::vector<SchedulePlane>;
ScheduleState schedule_state(Engine& E,int slot,const strata::core::Q36Schedule& recipe) {
    const auto cells=strata::core::q36_schedule_size(recipe);
    const auto valid=strata::core::q36_schedule_mtp(recipe);
    const auto& a=E.sess.at((size_t)slot);const auto& g=E.m->g;ScheduleState out;
    auto plane=[&](const char* name,int l,int h,uint64_t cell,const void* p,size_t n,bool fp32) {
        SchedulePlane v;v.name=name;v.layer=l;v.head=h;v.cell=cell;v.fp32=fp32;v.bytes.resize(n);
        CK(cudaMemcpyAsync(v.bytes.data(),p,n,cudaMemcpyDeviceToHost,E.st));
        CK(cudaStreamSynchronize(E.st));out.push_back(std::move(v));
    };
    plane("hlast",-1,-1,0,a.hlast,(size_t)g.n_embd*4,true);
    for(size_t l=0;l<a.conv.size();++l) if(a.conv[l]) {
        plane("conv",(int)l,-1,0,a.conv[l],(size_t)g.ssm_conv_channels*(g.ssm_d_conv-1)*4,true);
        plane("gdn",(int)l,-1,0,a.S[l],(size_t)g.ssm_v_heads*g.ssm_state_size*g.ssm_state_size*4,true);
    }
    for(size_t l=0;l<a.kc.size();++l) for(int h=0;h<g.n_head_kv;++h) {
        const bool mtp=l==E.m->layers.size();
        uint64_t begin=0;
        while(begin<cells) {
            if(mtp && !valid[(size_t)begin]) {++begin;continue;}
            uint64_t end=begin+1;while(end<cells && (!mtp || valid[(size_t)end])) ++end;
            const size_t off=((size_t)h*E.ctx+begin)*g.head_dim,n=(end-begin)*g.head_dim;
            if(a.kc[l]) {
                plane(mtp?"mtp_k":"main_k",(int)l,h,begin,a.kc[l]+off,n*4,true);
                plane(mtp?"mtp_v":"main_v",(int)l,h,begin,a.vc[l]+off,n*4,true);
            }
            if(E.kv_int8 && a.kv8[l].k) {
                const auto& x=a.kv8[l];
                plane(mtp?"mtp_k8":"main_k8",(int)l,h,begin,x.k+off,n,false);
                plane(mtp?"mtp_v8":"main_v8",(int)l,h,begin,x.v+off,n,false);
                plane(mtp?"mtp_ks":"main_ks",(int)l,h,begin,x.ks+off/64,n/64*2,false);
                plane(mtp?"mtp_vs":"main_vs",(int)l,h,begin,x.vs+off/64,n/64*2,false);
            }
            begin=end;
        }
    }
    return out;
}
bool schedule_compare(const ScheduleState& a,const ScheduleState& b,const char* kind) {
    bool equal=a.size()==b.size();uint64_t differing=0,nonfinite=0;bool first=true;
    for(size_t p=0;p<std::min(a.size(),b.size());++p) {
        const auto& x=a[p];const auto& y=b[p];
        const bool shape=x.name==y.name && x.layer==y.layer && x.head==y.head && x.cell==y.cell && x.bytes.size()==y.bytes.size();
        equal&=shape;
        for(size_t j=0;j<std::min(x.bytes.size(),y.bytes.size());++j) if(x.bytes[j]!=y.bytes[j]) {
            ++differing;
            if(first) {
                std::printf("SCHEDULE_FIRST comparison=%s category=%s layer=%d head=%d start_cell=%llu byte=%zu actual=%u reference=%u\n",
                    kind,x.name.c_str(),x.layer,x.head,(unsigned long long)x.cell,j,(unsigned)x.bytes[j],(unsigned)y.bytes[j]);first=false;
            }
        }
        if(x.fp32 && shape) for(size_t j=0;j+4<=x.bytes.size();j+=4) {
            float u,v;std::memcpy(&u,x.bytes.data()+j,4);std::memcpy(&v,y.bytes.data()+j,4);
            if(!std::isfinite(u) || !std::isfinite(v)) ++nonfinite;
        }
    }
    equal=equal && !differing && !nonfinite;
    std::printf("SCHEDULE_STATE comparison=%s planes=%zu reference_planes=%zu differing_bytes=%llu nonfinite=%llu exact=%d\n",
        kind,a.size(),b.size(),(unsigned long long)differing,(unsigned long long)nonfinite,equal?1:0);
    return equal;
}
bool schedule_logits(const std::vector<float>& a,const std::vector<float>& b,const char* kind) {
    size_t differing=0,nonfinite=0,first=SIZE_MAX;double max_abs=0;
    for(size_t i=0;i<std::min(a.size(),b.size());++i) {
        if(std::memcmp(&a[i],&b[i],4)) {++differing;if(first==SIZE_MAX) first=i;}
        if(!std::isfinite(a[i]) || !std::isfinite(b[i])) ++nonfinite;
        else max_abs=std::max(max_abs,std::abs((double)a[i]-b[i]));
    }
    const bool exact=a.size()==b.size() && !a.empty() && !differing && !nonfinite;
    std::printf("SCHEDULE_LOGITS comparison=%s differing=%zu first=%zu nonfinite=%zu max_abs=%.17g exact=%d\n",
        kind,differing,first,nonfinite,max_abs,exact?1:0);
    return exact;
}
int checkpoint_schedule_regression(Engine& E,const std::string& file,const std::string& dump,const std::string& corrupt) {
    using namespace strata::core;
    auto require=[](bool ok,const char* why) {if(!ok) {std::fflush(stdout);throw std::runtime_error(why);}};
    require(E.n_sess==4 && E.checkpoint_count && E.mtp_d==2 && !E.m->hybrid,"schedule gate requires four slots, checkpoints, GPU experts and MTP2");
    require(corrupt.empty() || corrupt=="conv" || corrupt=="gdn" || corrupt=="hlast","unknown schedule corruption control");
    Q36ScheduleBook book;Serve sv(E);E.schedule_book=&book;
    struct Cleanup {Engine& e;~Cleanup(){e.schedule_book=nullptr;e.schedule_restored={};e.schedule_before_restore={};}} cleanup{E};
    auto reset=[&](int s) {kv_require(E.kv_pool->reset(E.kv_ids[(size_t)s],E.kv_ids[(size_t)s]));E.kv_required[(size_t)s]=0;};
    for(int s=0;s<4;++s) reset(s);
    ScheduleState restored;uint64_t restored_pos=0;bool injected=false;
    E.schedule_restored=[&](int dst) {
        restored_pos=q36_schedule_size(book.live.at(dst));
        restored=schedule_state(E,dst,book.live.at(dst));
    };
    E.schedule_before_restore=[&](int src,int index) {
        if(corrupt.empty() || injected) return;
        const auto& g=E.m->g;size_t off=0;
        if(corrupt=="hlast") off=E.checkpoint_payload/4-(size_t)g.n_embd;
        else if(corrupt=="gdn") off=(size_t)g.ssm_conv_channels*(g.ssm_d_conv-1);
        const float poison=12345.25f;
        CK(cudaMemcpyAsync(E.sess[(size_t)src].checkpoint_data[(size_t)index]+off,&poison,4,cudaMemcpyHostToDevice,E.st));
        CK(cudaStreamSynchronize(E.st));injected=true;
        std::printf("SCHEDULE_CORRUPT category=%s source=%d record=%d float_offset=%zu\n",corrupt.c_str(),src,index,off);
    };
    std::unique_ptr<GateDump> artifact;if(!dump.empty()) artifact=std::make_unique<GateDump>(dump);
    std::ifstream input(file);require((bool)input,"cannot read schedule fixture");
    std::string line,group="unnamed",phase="warmup";size_t turn=0;std::array<std::vector<int32_t>,3> previous;
    const auto bank=E.kv_bank ? E.kv_bank->backing_info() : KvBackingInfo{};
    while(std::getline(input,line)) {
        if(line.empty()) continue;
        if(line.rfind("# case ",0)==0) {
            for(int s=0;s<4;++s) reset(s);
            previous={};group=line.substr(7);continue;
        }
        if(line.rfind("# phase ",0)==0) {phase=line.substr(8);require(phase=="warmup" || phase=="steady","invalid phase");continue;}
        require(line[0]!='#',"unknown schedule fixture directive");
        const auto split=line.find(' ');require(split!=std::string::npos,"expected SLOT CSV");
        const auto slot_text=line.substr(0,split);require(slot_text=="0" || slot_text=="1" || slot_text=="2","slot 3 is reference-only");
        const int slot=slot_text[0]-'0';const auto csv=line.substr(split+1);
        const auto ids=parse_ids(csv.c_str());require(!ids.empty() && ids.size()+16<(size_t)E.ctx,"invalid fixture length");
        size_t lcp=0;const auto& old=previous[(size_t)slot];while(lcp<old.size() && lcp<ids.size() && old[lcp]==ids[lcp]) ++lcp;
        std::printf("SCHEDULE_TURN case=%s turn=%zu phase=%s slot=%d tokens=%zu previous=%zu lcp=%zu class=%s\n",
            group.c_str(),turn+1,phase.c_str(),slot,ids.size(),old.size(),lcp,
            old.empty()?"new":lcp==old.size()?(ids.size()==old.size()?"identical":"append"):"rewrite");
        restored.clear();restored_pos=0;
        const auto before=E.kv_bank ? E.kv_bank->backing_info() : KvBackingInfo{};
        const auto start=Clock::now();sv.request_body("BGEN "+slot_text+" 1 "+csv);const double latency=ms_since(start);
        const auto after=E.kv_bank ? E.kv_bank->backing_info() : KvBackingInfo{};
        require(E.cur==slot && E.sess[(size_t)slot].live==ids && E.sess[(size_t)slot].pos==(int64_t)ids.size(),
            "tested path did not process complete prompt at expected position");
        const auto recipe=book.live.at(slot);q36_schedule_check(recipe,ids,(uint64_t)E.tmax);
        const auto read=recipe.back().take,reused=recipe.back().start;
        // The Serve timer encloses actual prefill, including checkpoint copies;
        // reference computation below is deliberately OUTSIDE every measurement.
        // Detailed prefill timing is supplied by the optional Serve gate callback.
        const auto actual=schedule_state(E,slot,recipe);std::vector<float> logits;E.host_logits(logits);const int pick=E.greedy();
        std::vector<std::unique_ptr<KvIdlePin>> pins;
        for(int s=0;s<3;++s) if(E.kv_pool->session((uint32_t)s).state==KvResidency::idle)
            pins.push_back(std::make_unique<KvIdlePin>(E.kv_pool.get(),E.kv_ids[(size_t)s]));
        bool restored_checked=restored.empty();
        struct NativeReference {
            Engine& e;decltype(reset)& clear;const std::vector<int32_t>& ids;
            void empty() {clear(3);e.kv_begin(3,ids.size()+16,true);e.select(3);e.reset();e.clear_images();}
            void call(const Q36ScheduleCall& c,size_t index) {
                if(e.pos!=(int64_t)c.start) throw std::runtime_error("reference call start mismatch");
                size_t boundary=0;
                const auto got=e.process(c.ids.data(),(int64_t)c.ids.size(),[&](int64_t done) {
                    const uint64_t end=c.start+(uint64_t)done;
                    if(boundary>=c.ends.size() || end!=c.ends[boundary++]) throw std::runtime_error("reference callback schedule mismatch");
                    const bool head=c.ids.size()==1 && !c.checkpoint_suffix ? true : (uint64_t)done==c.ids.size();
                    std::printf("SCHEDULE_CALL index=%zu start=%llu declared=%zu take=%llu boundary=%llu head=%d mtp=%d suffix=%d\n",
                        index,(unsigned long long)c.start,c.ids.size(),(unsigned long long)c.take,(unsigned long long)end,
                        head?1:0,(c.ids.size()!=1 || c.checkpoint_suffix)?1:0,c.checkpoint_suffix?1:0);
                    return (uint64_t)done<c.take;
                },c.checkpoint_suffix);
                if((uint64_t)got!=c.take || boundary!=c.ends.size()) throw std::runtime_error("reference consumed length mismatch");
                e.sess[3].pos=e.pos;
            }
        } ref{E,reset,ids};
        // Replay from empty using real process(), NEVER restore/copy_session.
        ref.empty();Q36Schedule prefix;
        for(size_t i=0;i<recipe.size();++i) {
            ref.call(recipe[i],i);prefix.push_back(recipe[i]);
            if(!restored_checked && q36_schedule_size(prefix)==restored_pos) {
                const bool same=schedule_compare(restored,schedule_state(E,3,prefix),"restored");
                require(same,"SCHEDULE_FAILURE comparison=restored (exact state required)");restored_checked=true;
            }
        }
        require(restored_checked,"restored boundary absent from independent recipe");
        std::vector<float> fresh;E.host_logits(fresh);
        const bool same_state=schedule_compare(actual,schedule_state(E,3,recipe),"matching");
        const bool same_logits=schedule_logits(logits,fresh,"matching");const bool same_pick=pick==E.greedy();
        std::printf("SCHEDULE_PICK comparison=matching actual=%d reference=%d exact=%d\n",pick,E.greedy(),same_pick?1:0);
        require(same_state && same_logits && same_pick,"SCHEDULE_FAILURE comparison=matching (exact state/logits required)");
        E.sess[3].pos=E.pos;E.kv_end(3);reset(3);
        // A DIFFERENT schedule remains visible, but is not the restoration oracle.
        E.kv_begin(3,ids.size()+16,true);E.select(3);E.reset();E.clear_images();
        E.process(ids.data(),(int64_t)ids.size(),[](int64_t){});E.host_logits(fresh);
        schedule_logits(logits,fresh,"different_schedule");
        Q36ScheduleCall full;full.ids=ids;full.take=ids.size();Q36Schedule full_recipe{full};
        schedule_compare(actual,schedule_state(E,3,full_recipe),"different_schedule");
        E.sess[3].pos=E.pos;E.kv_end(3);reset(3);E.select(slot);
        if(artifact) {
            artifact->vector(std::vector<uint64_t>{(uint64_t)recipe.size()});
            for(const auto& c:recipe) {
                artifact->vector(std::vector<uint64_t>{c.start,c.take,c.checkpoint_suffix?1ull:0ull});
                artifact->vector(c.ids);artifact->vector(c.ends);
            }
            artifact->vector(logits);artifact->vector(std::vector<int>{pick});
        }
        std::printf("SCHEDULE_MEASURE case=%s turn=%zu phase=%s reused=%llu replayed=%llu copy_us=%llu copy_bytes=%llu prefill_ms=%.6f latency_ms=%.6f\n",
            group.c_str(),turn+1,phase.c_str(),(unsigned long long)reused,(unsigned long long)read,
            (unsigned long long)(after.checkpoint_copy_us-before.checkpoint_copy_us),
            (unsigned long long)(after.checkpoint_copy_bytes-before.checkpoint_copy_bytes),sv.schedule_prefill_ms,latency);
        if(E.kv_bank) {
            const auto now=E.kv_bank->backing_info();
            require(now.driver_allocations==bank.driver_allocations && now.driver_allocation_attempts==bank.driver_allocation_attempts &&
                now.allocations==bank.allocations && now.bytes==bank.bytes,"physical allocation after readiness");
        }
        previous[(size_t)slot]=ids;++turn;std::fflush(stdout);
    }
    require(turn>1,"schedule fixture needs multiple turns");require(corrupt.empty() || injected,"negative control never reached checkpoint restore");
    for(int s=0;s<4;++s) reset(s);
    require(!E.kv_pool->info().mapped_bytes,"schedule reset left KV mapped");
    std::printf("CHECKPOINT_SCHEDULE_TEST PASS turns=%zu independent=1 exact=1\n",turn);return 0;
}
