// Independent IMAGE oracle. Original declared IDs + saved SVE files only.
// Never uploads candidate snapshots, copies checkpoints, or borrows restored KV.
ScheduleState image_schedule_state(Engine& e,int slot,const strata::core::Q36Schedule& recipe) {
    auto out=schedule_state(e,slot,recipe);const auto& s=e.sess[(size_t)slot];
    auto cpu=[&](const char* name,const void* data,size_t n) {SchedulePlane p;p.name=name;p.bytes.resize(n);if(n) std::memcpy(p.bytes.data(),data,n);out.push_back(std::move(p));};
    cpu("position",&s.pos,sizeof s.pos);cpu("rotary_delta",&s.rdelta,sizeof s.rdelta);
    cpu("tokens",s.live.data(),s.live.size()*4);
    std::vector<int64_t> images;for(const auto& x:s.imgs) {images.push_back(x.cell);images.push_back(x.n);uint64_t h=x.hash;int64_t value;std::memcpy(&value,&h,8);images.push_back(value);}
    cpu("complete_image_metadata",images.data(),images.size()*8);
    int32_t device_delta=0;CK(cudaMemcpyAsync(&device_delta,e.d_delta,4,cudaMemcpyDeviceToHost,e.st));CK(cudaStreamSynchronize(e.st));
    cpu("device_rotary_delta",&device_delta,4);return out;
}
int image_checkpoint_regression(Engine& e,const std::string& fixture,const std::string& path,const std::string& corrupt) {
    using namespace strata::core;
    auto require=[](bool ok,const char* why){if(!ok){std::fflush(stdout);throw std::runtime_error(why);}};
    require(e.vision && e.kv_bank && e.kv_pool && e.checkpoint_count==4 && e.checkpoint_interval==1024 && e.n_sess==4 &&
        e.tmax==512 && e.vision_max_rows==1024 && e.mtp_d==2 && !e.m->hybrid,"image checkpoint gate requires strict qualified geometry");
    require(corrupt.empty() || corrupt=="conv","unsupported image corruption control");
    Q36ScheduleBook book;e.schedule_book=&book;Serve server(e);bool injected=false;
    struct Observation {uint64_t pos=0;ScheduleState state;bool verified=false;};
    std::vector<Observation> observations;
    struct Cleanup {Engine& e;~Cleanup(){e.schedule_book=nullptr;e.schedule_restored={};e.schedule_before_restore={};}} cleanup{e};
    auto reset=[&](int s) {kv_require(e.kv_pool->reset(e.kv_ids[(size_t)s],e.kv_ids[(size_t)s]));e.kv_required[(size_t)s]=0;};
    auto reset_all=[&](){require(!server.n_active,"active image gate slot");for(int s=0;s<4;++s) reset(s);};
    e.schedule_restored=[&](int dst) {
        // Boundary contract, checked at EVERY observation: the observed slot IS the
        // current one (so its CPU/device rotary pair belongs to it), the engine sits
        // exactly at the restored boundary, and no new suffix has been consumed.
        require(e.cur==dst,"image gate observed restored state for a non-current destination");
        const auto& recipe=book.live.at(dst);const auto end=q36_schedule_size(recipe);
        require(e.pos==(int64_t)end && e.sess[(size_t)dst].live.size()==end && e.sess[(size_t)dst].pos==(int64_t)end,
            "image gate observed restored state outside its boundary");
        const auto seeded=q36_schedule_mtp(recipe);require(std::all_of(seeded.begin(),seeded.end(),[](bool v){return v;}),"restored image prefix has unseeded MTP cells");
        observations.push_back({end,image_schedule_state(e,dst,recipe),false});
        require(observations.size()<=2,"image gate observed more than two restored states per request");
    };
    e.schedule_before_restore=[&](int src,int index) {
        if(corrupt.empty() || injected) return;
        const float poison=12345.25f;
        CK(cudaMemcpyAsync(e.sess[(size_t)src].checkpoint_data[(size_t)index],&poison,4,cudaMemcpyHostToDevice,e.st));CK(cudaStreamSynchronize(e.st));
        injected=true;std::printf("IMAGE_CHECKPOINT_CORRUPT category=conv\n");
    };
    std::unique_ptr<GateDump> dump;if(!path.empty()) dump=std::make_unique<GateDump>(path);
    const auto bank=e.kv_bank->backing_info();const auto driver=vmm_allocation_stats();const auto staging=e.d_img;
    std::ifstream input(fixture);require((bool)input,"cannot open image checkpoint fixture");std::string line,group="unnamed";
    size_t turns=0,positive=0,zero=0,cross=0;std::vector<std::string> control;reset_all();
    while(std::getline(input,line)) {
        if(line.empty()) continue;
        if(line.rfind("# case ",0)==0) {group=line.substr(7);reset_all();continue;}
        std::vector<std::string> fields;std::istringstream row(line);std::string f;while(std::getline(row,f,'\t')) fields.push_back(f);
        require(fields.size()==6,"image gate row needs slot/phase/expect/key/SVE/CSV");
        require((fields[0]=="0" || fields[0]=="1" || fields[0]=="2") && (fields[1]=="warmup" || fields[1]=="steady"),"invalid image gate slot/phase");
        if(group=="native_boundaries" && control.empty()) control=fields;
        const int slot=fields[0][0]-'0';const auto ids=parse_ids(fields[5].c_str());
        require(ids.size()>2 && ids.size()<=6144 && turns<24,"bounded image gate input exceeded");
        std::printf("IMAGE_CHECKPOINT_PROGRESS case=%s turn=%zu phase=%s slot=%d tokens=%zu\n",group.c_str(),turns+1,fields[1].c_str(),slot,ids.size());std::fflush(stdout);
        observations.clear();
        const auto fires_before=e.restore_snapshot_fires,deferred_before=e.restore_snapshot_deferred,
            immediate_before=e.restore_snapshot_immediate,refused_before=e.restore_snapshot_refused;
        const auto before=e.kv_bank->backing_info();const auto metadata_before=e.image_metadata_us;
        const auto begin=Clock::now();server.request_body("BGENI "+fields[0]+" 1 img_keys="+fields[3]+" "+fields[4]+" "+fields[5]);const auto elapsed=ms_since(begin);
        require(e.cur==slot && e.sess[(size_t)slot].live==ids && e.sess[(size_t)slot].pos==(int64_t)ids.size(),"image gate request incomplete");
        const auto after=e.kv_bank->backing_info();const auto recipe=book.live.at(slot);q36_schedule_check(recipe,ids,(uint64_t)e.tmax);
        // Exactly one observation per producer, and every observation is checked: a
        // complete-state copy defers to the caller's select, checkpoint_restore observes
        // in place after it has selected its destination.
        const auto fires=e.restore_snapshot_fires-fires_before,deferred=e.restore_snapshot_deferred-deferred_before,
            immediate=e.restore_snapshot_immediate-immediate_before,refused=e.restore_snapshot_refused-refused_before;
        require(refused==0 && fires==deferred+immediate && fires==observations.size() && deferred<=1 && immediate<=1,
            "image gate restored-state observation count mismatch");
        const auto reused=recipe.back().start,replayed=recipe.back().take;const auto cp=e.checkpoint_last_reused;
        if(cp) require(immediate==1,"checkpoint reuse without an in-place restored-state observation");
        if(!reused && !cp) require(fires==0,"restored-state observation without any restore or donor copy");
        ImgReq original;require(build_images(fields[4],ids,e.m->g.n_embd,{},original).empty(),"reference cannot reconstruct original image input");
        const auto image_begin=(uint64_t)original.imgs.front().cell,image_end=(uint64_t)(original.imgs.back().cell+original.imgs.back().n);
        if(fields[2]=="zero") {require(!reused && !cp,"legitimate zero-reuse control unexpectedly reused");++zero;}
        else if(fields[2]=="before") require(e.image_checkpoints ? (cp && cp<=image_begin) : !cp,"expected valid checkpoint BEFORE image (or disabled fallback)");
        else if(fields[2]=="after") {require(e.image_checkpoints ? (cp>=image_end && cp<ids.size()) : !cp,"expected valid checkpoint AFTER complete image (or disabled fallback)");if(e.image_checkpoints && fields[1]=="steady") ++positive;}
        else if(fields[2]=="exact") require(reused && !cp,"exact-prefix path did not remain preferred");
        else require(fields[2]=="observe","unknown image reuse expectation");
        if(cp && slot==1) ++cross;
        const auto actual=image_schedule_state(e,slot,recipe);std::vector<float> logits;e.host_logits(logits);
        // Pins protect ALL candidate owners from the reference's admission/reset.
        std::vector<std::unique_ptr<KvIdlePin>> pins;
        for(int s=0;s<3;++s) if(e.kv_pool->session((uint32_t)s).state==KvResidency::idle)
            pins.push_back(std::make_unique<KvIdlePin>(e.kv_pool.get(),e.kv_ids[(size_t)s]));
        reset(3);e.kv_begin(3,ids.size()+16,true);e.select(3);e.reset();e.clear_images();
        Q36Schedule prefix;
        for(const auto& call:recipe) {
            require(!call.image_file.empty() && call.start==(uint64_t)e.pos && call.image_ids.size()==call.start+call.ids.size(),"missing original image call context");
            require(std::equal(call.ids.begin(),call.ids.end(),call.image_ids.begin()+(size_t)call.start),"declared suffix does not match original input");
            ImgReq fresh;require(build_images(call.image_file,call.image_ids,e.m->g.n_embd,{},fresh).empty(),"independent legacy image reconstruction failed");
            require(call.image_keys.empty() || call.image_keys.size()==fresh.imgs.size(),"original image key count differs");
            // Legacy positions/rows, but strict-mode identity includes ACTUAL rows
            // even with provided keys. Derive it independently, not from proof metadata.
            if(!call.image_keys.empty()) for(size_t j=0;j<fresh.imgs.size();++j) fresh.imgs[j].hash=fnv1a(&call.image_keys[j],8,fresh.imgs[j].hash);
            require(e.sess[3].live.size()==call.start && std::equal(e.sess[3].live.begin(),e.sess[3].live.end(),call.image_ids.begin()),"reference original token lineage diverged");
            Engine::VisionPrefillScope scope(e,true);e.set_images(fresh.rows,fresh.cell_row,fresh.mrope,(int64_t)call.image_ids.size(),fresh.delta);
            size_t boundary=0;const auto got=e.process(call.ids.data(),(int64_t)call.ids.size(),[&](int64_t done) {
                require(boundary<call.ends.size() && call.start+(uint64_t)done==call.ends[boundary++],"independent image callback schedule differs");
                return (uint64_t)done<call.take;
            },call.checkpoint_suffix);
            require((uint64_t)got==call.take && boundary==call.ends.size(),"independent image declared call/take mismatch");scope.finish();
            auto& state=e.sess[3];state.pos=e.pos;state.live.assign(call.image_ids.begin(),call.image_ids.begin()+(size_t)e.pos);state.imgs.clear();
            for(const auto& image:fresh.imgs) {
                require(!(image.cell<e.pos && e.pos<image.cell+image.n),"independent reference cuts an image");
                if(image.cell+image.n<=e.pos) state.imgs.push_back(image);
            }
            // Independently calculate retained offset from the LEGACY-built table,
            // not the candidate identity/delta helper or checkpoint CPU metadata.
            const auto at=((size_t)e.pos-1)*3;
            e.set_delta((int32_t)((int64_t)std::max({fresh.mrope[at],fresh.mrope[at+1],fresh.mrope[at+2]})+1-e.pos));
            prefix.push_back(call);
            for(auto& o:observations) if(!o.verified && o.pos==q36_schedule_size(prefix)) {
                require(schedule_compare(o.state,image_schedule_state(e,3,prefix),"image_restored"),"IMAGE_CHECKPOINT_FAILURE comparison=restored exact state required");
                o.verified=true;
            }
        }
        for(const auto& o:observations) require(o.verified,"restored boundary absent from independent image schedule");
        std::vector<float> reference;e.host_logits(reference);
        require(schedule_compare(actual,image_schedule_state(e,3,recipe),"image_matching"),"IMAGE_CHECKPOINT_FAILURE comparison=final_state");
        require(schedule_logits(logits,reference,"image_matching"),"IMAGE_CHECKPOINT_FAILURE comparison=logits full exact required");
        e.sess[3].pos=e.pos;e.kv_end(3);reset(3);pins.clear();e.select(slot);
        if(dump) {for(const auto& p:actual) dump->vector(p.bytes);dump->vector(logits);}
        const auto now=e.kv_bank->backing_info();const auto d=vmm_allocation_stats();
        require(now.bytes==bank.bytes && now.allocations==bank.allocations && d.allocations==driver.allocations && d.attempts==driver.attempts && d.releases==driver.releases,"physical allocation/release after readiness");
        require(e.d_img==staging && vision_gate::views_idle(e),"image staging grew or lease survived prefill");
        std::printf("IMAGE_CHECKPOINT_MEASURE case=%s turn=%zu phase=%s slot=%d reused=%llu replayed=%llu checkpoint=%llu prefill_ms=%.6f request_ms=%.6f copy_us=%llu copy_bytes=%llu metadata_us=%llu independent=1 exact=1\n",group.c_str(),++turns,fields[1].c_str(),slot,
            (unsigned long long)reused,(unsigned long long)replayed,(unsigned long long)cp,server.schedule_prefill_ms,elapsed,
            (unsigned long long)(after.checkpoint_copy_us-before.checkpoint_copy_us),(unsigned long long)(after.checkpoint_copy_bytes-before.checkpoint_copy_bytes),(unsigned long long)(e.image_metadata_us-metadata_before));std::fflush(stdout);
    }
    require(turns>=8 && zero && (!e.image_checkpoints || (positive && cross)),"missing steady rewrite reuse/cross-slot/zero coverage");
    require(e.restore_snapshot_deferred>=1,"no cross-slot complete-state donor observation was exercised");
    require(corrupt.empty() || injected,"corruption control never reached a checkpoint");
    require(control.size()==6,"missing native cancellation control");
    const auto control_ids=parse_ids(control[5].c_str());
    const auto command="BGENI 0 1 img_keys="+control[3]+" "+control[4]+" "+control[5];
    reset_all();server.vision_test_on_chunk=[&](int64_t done){if(done>=1024) server.stop.store(true);};
    server.request_body(command);server.vision_test_on_chunk={};
    require(e.sess[0].pos==0 && e.sess[0].live.empty() && e.sess[0].imgs.empty() && !e.sess[0].image_source && !e.sess[0].image_ready_end && vision_gate::views_idle(e),"cancel inside image retained usable state/views");
    for(const auto& p:e.sess[0].image_points) require(!p.ready,"cancel/reset retained image checkpoint identity");
    reset_all();server.vision_test_on_chunk=[&](int64_t done){if(done>=512) server.stop.store(true);};
    server.request_body(command);server.vision_test_on_chunk={};
    require(e.sess[0].pos==512 && e.sess[0].live.size()==512 && e.sess[0].imgs.empty() && e.sess[0].rdelta==0 && vision_gate::views_idle(e),"cancel before image lost valid prefix semantics");
    const auto cancelled_recipe=book.live[0];const auto cancelled=image_schedule_state(e,0,cancelled_recipe);
    {const KvIdlePin pin(e.kv_pool.get(),e.kv_ids[0]);
    reset(3);e.kv_begin(3,control_ids.size()+16,true);e.select(3);e.reset();e.clear_images();
    ImgReq fresh;require(build_images(control[4],control_ids,e.m->g.n_embd,{},fresh).empty(),"cancel reference image reconstruction failed");
    {Engine::VisionPrefillScope scope(e,true);e.set_images(fresh.rows,fresh.cell_row,fresh.mrope,(int64_t)control_ids.size(),fresh.delta);
     require(e.process(control_ids.data(),(int64_t)control_ids.size(),[](int64_t done){return done<512;})==512,"cancel reference did not stop at original callback");}
    e.sess[3].pos=e.pos;e.sess[3].live.assign(control_ids.begin(),control_ids.begin()+512);e.sess[3].imgs.clear();e.set_delta(0);
    require(schedule_compare(cancelled,image_schedule_state(e,3,cancelled_recipe),"image_cancel_before"),"IMAGE_CHECKPOINT_FAILURE comparison=cancelled_prefix");
    e.kv_end(3);reset(3);e.select(0);
    if(e.image_checkpoints) {
        require(e.checkpoint_available(0,0,&pin,nullptr,true) && !e.checkpoint_available(0,0,nullptr,nullptr,true),"image checkpoint donor-pin ownership mismatch");
        const auto old=e.sess[0].checkpoints.points[0];require(old.pos==512,"pre-image cancellation lost useful legal checkpoint");
    }
    } // release donor pin before reset/generation invalidation
    reset_all();require(!e.checkpoint_available(0,0,nullptr,nullptr,true),"reset left an image checkpoint available");
    require(!e.kv_pool->info().mapped_bytes,"image gate leaked KV mappings");
    const auto final_driver=vmm_allocation_stats();require(final_driver.allocations==driver.allocations && final_driver.attempts==driver.attempts && final_driver.releases==driver.releases,"cancellation allocated/released physical bank handles");
    std::printf("IMAGE_CHECKPOINT_CANCEL PASS inside_invalidated=1 before_independent_exact=1 pin_checked=%d reset_invalidated=1\n",e.image_checkpoints?1:0);
    std::printf("IMAGE_CHECKPOINT_OBSERVATIONS total=%zu deferred=%zu immediate=%zu refused=%zu balanced=%d\n",
        e.restore_snapshot_fires,e.restore_snapshot_deferred,e.restore_snapshot_immediate,e.restore_snapshot_refused,
        e.restore_snapshot_fires+e.restore_snapshot_refused==e.restore_snapshot_deferred+e.restore_snapshot_immediate?1:0);

    std::printf("IMAGE_CHECKPOINT_TEST PASS enabled=%d turns=%zu steady_after_reuse=%zu cross_slot=%zu zeros=%zu independent=1 exact=1\n",e.image_checkpoints?1:0,turns,positive,cross,zero);return 0;
}
