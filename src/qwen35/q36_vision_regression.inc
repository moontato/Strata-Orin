// Developer-only model gate. No tolerance, image checkpoint or sampled-image MTP implementation.
namespace vision_gate {
bool views_idle(Engine& e) {
#ifdef STRATA_Q36_VISION_PREALLOCATION_IMPL
    return !e.vision_lease.busy() && !e.img_active && !e.map_on && !e.mtab_now;
#else
    (void)e;return true; // baseline has no banked vision; only the off arm can run
#endif
}
std::vector<uint8_t> snapshot(Engine& e,int slot) {
    e.sess[(size_t)e.cur].pos=e.pos;const auto& s=e.sess[(size_t)slot];const auto& g=e.m->g;
    std::vector<uint8_t> out;
    auto plane=[&](const void* p,size_t bytes,bool cpu=false) {
        if(!bytes) return;
        const size_t at=out.size();out.resize(at+bytes);
        if(cpu) std::memcpy(out.data()+at,p,bytes);
        else {CK(cudaMemcpyAsync(out.data()+at,p,bytes,cudaMemcpyDeviceToHost,e.st));CK(cudaStreamSynchronize(e.st));}
    };
    plane(&s.pos,sizeof s.pos,true);plane(&s.rdelta,sizeof s.rdelta,true);
    plane(s.live.data(),s.live.size()*4,true);
    if(s.live.empty() && !s.pos) return out; // empty owners may have no backed state VA
    plane(s.hlast,(size_t)g.n_embd*4);
    for(const auto& image:s.imgs) {plane(&image.cell,8,true);plane(&image.n,8,true);plane(&image.hash,8,true);}
    for(size_t l=0;l<s.conv.size();++l) {
        if(s.conv[l]) {plane(s.conv[l],(size_t)g.ssm_conv_channels*(g.ssm_d_conv-1)*4);plane(s.S[l],(size_t)g.ssm_v_heads*g.ssm_state_size*g.ssm_state_size*4);}
        for(int h=0;h<g.n_head_kv;++h) {
            const size_t at=(size_t)h*e.ctx*g.head_dim,cells=(size_t)s.pos*g.head_dim;
            if(s.kc[l] && cells) {plane(s.kc[l]+at,cells*4);plane(s.vc[l]+at,cells*4);}
            if(e.kv_int8 && s.kv8[l].k && cells) {
                plane(s.kv8[l].k+at,cells);plane(s.kv8[l].v+at,cells);
                plane(s.kv8[l].ks+at/64,cells/64*2);plane(s.kv8[l].vs+at/64,cells/64*2);
            }
        }
    }
    return out;
}
}
int vision_preallocation_regression(Engine& e,const std::string& fixture,const std::string& path) {
    auto require=[](bool ok,const char* why){if(!ok) throw std::runtime_error(why);};
    require(e.vision && e.kv_pool && e.kv_int8 && e.n_sess==4 && e.mtp_d==2 && e.checkpoint_count>0 && !e.m->hybrid && e.tmax==512,
            "vision gate needs vision, four slots, INT8 unified KV, full GPU MTP2, chunk512");
    std::ifstream file(fixture);std::string emb_a,a,emb_b,b,text;
    require((bool)std::getline(file,emb_a) && (bool)std::getline(file,a) && (bool)std::getline(file,emb_b) &&
            (bool)std::getline(file,b) && (bool)std::getline(file,text),"vision fixture needs A.sve/CSV/B.sve/CSV/text CSV");
    ImgReq ia,ib;const auto ids=parse_ids(a.c_str());
    require(build_images(emb_a,ids,e.m->g.n_embd,{},ia).empty() && build_images(emb_b,parse_ids(b.c_str()),e.m->g.n_embd,{},ib).empty(),"invalid actual encoder fixture");
    require(ia.delta!=ib.delta,"fixture must give distinct slot rotary offsets");
    bool inside=false;for(const auto& image:ia.imgs) inside |= image.cell<e.tmax && e.tmax<image.cell+image.n;
    require(inside && ids.size()>(size_t)e.tmax*3 && ia.imgs.back().cell+ia.imgs.back().n<e.tmax*3,
            "A must cross chunk1 inside image, end images before chunk3 and retain a longer text tail");
    Serve server(e);require(server.read_windows>0,"vision mixed gate needs interleaved read windows");
    std::unique_ptr<GateDump> dump;if(!path.empty()) dump=std::make_unique<GateDump>(path);
    const auto bank=e.kv_bank ? e.kv_bank->backing_info() : strata::core::KvBackingInfo{};
    const auto driver=strata::core::vmm_allocation_stats();const auto staging=e.d_img;
    if(e.kv_bank) require(e.logits_mr && e.pick_mr,"strict vision batch-MTP scratch was not reserved before readiness");
    auto reset=[&]() {
        require(server.n_active==0,"reset while a batch slot is active");
        for(int s=0;s<4;++s) {strata::core::kv_require(e.kv_pool->reset(e.kv_ids[(size_t)s],e.kv_ids[(size_t)s]));e.kv_required[(size_t)s]=0;}
        e.select(0);e.reset();e.clear_images();
    };
    auto check=[&]() {
        if(e.kv_bank) {
            require(vision_gate::views_idle(e),"image views outlived prefill");
            require(e.d_img==staging,"strict image staging pointer changed after readiness");
            const auto now=e.kv_bank->backing_info();const auto d=strata::core::vmm_allocation_stats();
            require(now.bytes==bank.bytes && now.allocations==bank.allocations && d.allocations==driver.allocations &&
                    d.attempts==driver.attempts && d.releases==driver.releases,"strict bank physical allocation/release changed");
        }
        for(int s=0;s<4;++s) if(!e.sess[(size_t)s].imgs.empty())
            for(auto p:e.sess[(size_t)s].checkpoints.points) require(!p.pos,"image request captured a text checkpoint");
    };
    auto command=[&](const std::string& line) {server.request_body(line);check();};
    auto record=[&](const char* name) {
        for(int s=0;s<4;++s) {
            const auto bytes=vision_gate::snapshot(e,s);if(dump) dump->vector(bytes);
        }
        std::vector<float> logits;e.host_logits(logits);for(float v:logits) require(std::isfinite(v),"nonfinite vision/text logits");
        if(dump) dump->vector(logits);
        std::printf("VISION_CASE name=%s slots=4 all_persistent_planes=1 finite_logits=1\n",name);
    };
    reset();
    command("BGENI 0 1 "+emb_a+" "+a);
    require(e.sess[0].rdelta==ia.delta && !e.sess[0].imgs.empty(),"image A positional state missing");
    const auto image0=vision_gate::snapshot(e,0);
    command("BGEN 1 1 "+text);
    bool text_checkpoint=false;for(auto p:e.sess[1].checkpoints.points) text_checkpoint |= p.pos>0;
    require(text_checkpoint,"text control did not retain any eligible checkpoint");
    require(e.sess[1].rdelta==0 && e.sess[1].imgs.empty() && image0==vision_gate::snapshot(e,0),"text request changed image slot");
    command("BGENI 2 1 "+emb_b+" "+b);
    require(e.sess[2].rdelta==ib.delta && image0==vision_gate::snapshot(e,0),"second image changed first slot/offset");
    const auto image2=vision_gate::snapshot(e,2);
    command("BGENI 3 1 "+emb_a+" "+a);command("BGEN 0 1 "+text);
    require(e.sess[0].rdelta==0 && e.sess[0].imgs.empty() && image2==vision_gate::snapshot(e,2),"image->text transition contaminated peers");record("image_text_four_slots");
    reset();
    command("BGEN 0 128 "+text);command("BGENI 1 64 "+emb_a+" "+a);
    command("BGENI 2 64 "+emb_b+" "+b);command("BGEN 3 64 "+text);
    require(server.n_active==4,"fixture ended too early to cover four active mixed slots");
    for(int i=0;i<3 && server.n_active;++i) server.window();
    for(auto& slot:server.slots) slot.stop=true;
    for(int i=0;i<16 && server.n_active;++i) server.window();
    require(!server.n_active,"mixed slots did not finish/reap leases");check();
    require(e.sess[0].rdelta==0 && e.sess[3].rdelta==0 && e.sess[1].rdelta==ia.delta && e.sess[2].rdelta==ib.delta,"mixed rows lost per-slot rotary state");record("mixed_interleaved_decode");
    reset();server.vision_test_on_chunk=[&](int64_t done){if(done>=e.tmax) server.stop.store(true);};
    command("BGENI 0 64 "+emb_a+" "+a);server.vision_test_on_chunk={};
    require(e.sess[0].live.empty() && e.sess[0].imgs.empty(),"cancelled partial image remained reusable");
    if(e.kv_bank) require(e.sess[0].pos==0 && e.sess[0].rdelta==0,"partial image cancellation did not invalidate owner");
    command("BGEN 0 1 "+text);record("cancel_inside_image_then_text");
    reset();server.vision_test_on_chunk=[&](int64_t done){if(done>=e.tmax*3) server.stop.store(true);};
    command("BGENI 0 64 "+emb_a+" "+a);server.vision_test_on_chunk={};
    require(e.sess[0].live.size()==(size_t)e.tmax*3 && !e.sess[0].imgs.empty(),"text-tail cancellation did not retain complete images");
    require(e.sess[0].rdelta==ia.delta,"retained image prefix has wrong decode offset");
    command("BGENI 0 1 "+emb_a+" "+a);record("cancel_text_tail_then_resume_image");
    reset();command("GENI 2 temperature=0.7 seed=17 sampled_mtp=1 "+emb_a+" "+a);record("sampled_image_non_speculative");
    // The external runner also checks GENI offered/accepted == 0, then text
    // offered > 0. Existing sampler math/tolerance are not changed by this gate.
    reset();command("GEN 12 temperature=0.7 seed=17 sampled_mtp=1 "+text);record("sampled_text_after_image");
    reset();check();require(!e.kv_pool->info().mapped_bytes,"vision gate leaked KV mappings");
    std::printf("VISION_PREALLOCATION_MODEL PASS bank=%d image_text=1 mixed_slots=4 cancellation=inside_and_after images_checkpointed=0 mapped_bytes=0\n",e.kv_bank?1:0);
    return 0;
}
