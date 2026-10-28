// Developer replay for diagnostics OFF/ON neutrality ONLY. Not an independent
// image oracle. Every arm reconstructs actual image rows/M-RoPE via GENI parsing.
int image_cache_regression(Engine& e,const std::string& fixture,const std::string& path) {
    auto require=[](bool ok,const char* why){if(!ok) throw std::runtime_error(why);};
    require(e.vision && e.kv_pool && e.kv_bank && e.kv_int8 && e.n_sess==4 && e.mtp_d==2 && e.checkpoint_count &&
            !e.m->hybrid && e.tmax==512 && e.vision_max_rows==1024,"image-cache gate needs deployed strict vision geometry");
    require(!path.empty(),"image-cache gate requires an exact state dump");
    std::ifstream file(fixture);require((bool)file,"cannot read image-cache fixture");GateDump dump(path);Serve sv(e);
    const auto bank=e.kv_bank->backing_info();const auto driver=strata::core::vmm_allocation_stats();const auto staging=e.d_img;
    std::string line,group="unnamed";size_t turns=0;int hits=0,misses=0;
    auto reset=[&](){
        require(!sv.n_active,"image-cache reset with active decode");
        for(int s=0;s<4;++s) {strata::core::kv_require(e.kv_pool->reset(e.kv_ids[(size_t)s],e.kv_ids[(size_t)s]));e.kv_required[(size_t)s]=0;}
    };
    reset();
    while(std::getline(file,line)) {
        if(line.empty()) continue;
        if(line.rfind("# case ",0)==0) {group=line.substr(7);reset();continue;}
        // MODE TAB EXPECT(hit/miss/observe) TAB KEY TAB SVE TAB CSV
        std::vector<std::string> fields;std::istringstream row(line);std::string field;
        while(std::getline(row,field,'\t')) fields.push_back(field);
        require(fields.size()==5,"image-cache fixture needs mode/expect/key/SVE/CSV");
        const auto& mode=fields[0];const auto& expect=fields[1];
        require(mode=="auto" || mode=="0" || mode=="1","invalid image-cache destination");
        require(expect=="hit" || expect=="miss" || expect=="observe","invalid cache expectation");
        const auto ids=parse_ids(fields[4].c_str());require(ids.size()>2 && ids.size()<=4096 && turns<12,"short image-cache fixture bound exceeded");
        std::vector<uint64_t> schedule;uint64_t read=0;
        sv.vision_test_on_chunk=[&](int64_t done) {
            read=(uint64_t)done;const auto start=(uint64_t)e.pos-read;
            schedule.insert(schedule.end(),{(uint64_t)e.cur,start,ids.size()-start,read});
        };
        const auto started=Clock::now();
        const auto command=(mode=="auto"?"GENI 1 ":"BGENI "+mode+" 1 ")+"img_keys="+fields[2]+" "+fields[3]+" "+fields[4];
        sv.request_body(command);sv.vision_test_on_chunk={};
        const int slot=e.cur;const auto reused=ids.size()-read;
        require(read && e.sess[(size_t)slot].live==ids && e.sess[(size_t)slot].pos==(int64_t)ids.size(),"incomplete image-cache request");
        require(expect!="hit" || reused>0,"native constructed-prefix control did not reuse");
        require(expect!="miss" || reused==0,"negative control unexpectedly reused");
        hits+=reused?1:0;misses+=reused?0:1;
        dump.vector(schedule);dump.vector(std::vector<uint64_t>{(uint64_t)slot,reused,read});
        for(int s=0;s<4;++s) {
            dump.vector(vision_gate::snapshot(e,s));
            const auto owner=e.kv_pool->session((uint32_t)s);
            dump.vector(std::vector<uint64_t>{owner.id.generation,owner.epoch,owner.tokens,owner.reserved_tokens,(uint64_t)owner.state});
            for(auto p:e.sess[(size_t)s].checkpoints.points) require(!p.pos,"image sequence captured a text checkpoint");
        }
        std::vector<float> logits;e.host_logits(logits);
        for(float v:logits) require(std::isfinite(v),"nonfinite image-cache logits");
        dump.vector(logits);dump.vector(std::vector<int>{e.greedy()});
        const auto now=e.kv_bank->backing_info();const auto d=strata::core::vmm_allocation_stats();
        require(vision_gate::views_idle(e) && e.d_img==staging,"vision staging lease/pointer changed");
        require(now.bytes==bank.bytes && now.allocations==bank.allocations && d.allocations==driver.allocations &&
                d.attempts==driver.attempts && d.releases==driver.releases,"post-ready physical allocation/release changed");
        std::printf("IMAGE_CACHE_TURN case=%s turn=%zu slot=%d reused=%llu read=%llu elapsed_ms=%.3f schedule_and_state_recorded=1\n",
            group.c_str(),++turns,slot,(unsigned long long)reused,(unsigned long long)read,ms_since(started));std::fflush(stdout);
    }
    reset();require(!e.kv_pool->info().mapped_bytes,"image-cache replay leaked mappings");
    require(turns>=8 && hits>=2 && misses>=2,"image-cache fixture missing required controls");
    std::printf("IMAGE_CACHE_DIAGNOSTIC_TEST PASS turns=%zu hits=%d misses=%d independent_image_reference=0\n",turns,hits,misses);
    return 0;
}
