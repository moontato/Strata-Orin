#include "strata/core/q36_prefix_checkpoint.hpp"
#include <cassert>
#include <iostream>
#include <numeric>
using namespace strata::core;
using CP=Q36PrefixCheckpoints;

// Exact causal toy state, not a Qwen numerical or latency qualification.
struct State {
    uint64_t recurrent=0, conv=0, hidden=0;
    bool operator==(const State&) const = default;
};
struct Toy {
    State s;
    std::vector<uint64_t> kv, mtp;
    void step(int t) {
        s.recurrent=s.recurrent*33+(uint64_t)t+s.conv;
        s.conv=s.conv*17+(uint64_t)t;
        kv.push_back(s.recurrent^(uint64_t)t);
        mtp.push_back(s.hidden+kv.back());
        s.hidden=s.recurrent*19+mtp.back();
    }
};
using Tokens=std::vector<int>;
static size_t lcp(const Tokens& a,const Tokens& b) {
    size_t n=0;while(n<a.size() && n<b.size() && a[n]==b[n]) ++n;return n;
}
struct Slot {
    CP cp;
    std::array<State,4> payload{}; // NEVER a fifth checkpoint, even during replacement
    Toy toy;
    Tokens live;
    uint64_t generation=1, epoch=1;
    void reset() {++generation;++epoch;cp.clear();live.clear();toy={};}
};
struct Metrics { uint64_t replay=0, reused=0, captures=0, copies=0, rewrites=0; };
struct Runner {
    std::array<Slot,4> slots;
    Metrics m;
    explicit Runner(CP::Policy policy,size_t cap=4) {
        for(auto& s:slots) {s.cp.policy=policy;s.cp.points.resize(cap);}
    }
    // A model of engine request sequencing; actual pool ownership/failure gates
    // live in q36_prefix_checkpoint_test and kv_preallocation_test, not this toy.
    size_t request(const Tokens& in,int dst=0,size_t stop=SIZE_MAX) {
        CP::Learning learned;
        size_t closest=0,reuse=0;int source=-1,point=-1,full=-1;
        for(int i=0;i<4;++i) {
            auto& a=slots[i];const auto common=lcp(a.live,in);
            if(common && common==a.live.size() && common<in.size()) full=i;
            if(common>closest) {closest=common;learned=a.cp.learning;source=i;}
        }
        if(full>=0) {source=full;learned=slots[full].cp.learning;reuse=slots[full].live.size();}
        else {
            if(source>=0 && slots[dst].cp.policy==CP::Policy::adaptive &&
               learned.observe(closest,slots[source].live.size(),512)) ++m.rewrites;
            source=-1;
            for(int i=0;i<4;++i) {
                auto& a=slots[i];const int p=a.cp.best(lcp(a.live,in),in.size(),a.generation,a.epoch);
                if(p>=0 && a.cp.points[p].pos>reuse) {reuse=a.cp.points[p].pos;source=i;point=p;}
            }
        }
        auto& b=slots[dst];
        if(reuse) {
            if(source!=dst) {
                auto& a=slots[source];
                b.toy=a.toy;b.live.assign(a.live.begin(),a.live.begin()+reuse);
                if(full<0) {
                    b.cp.points=a.cp.points;b.payload=a.payload;
                    for(auto p:b.cp.points) if(p.pos && p.pos<=reuse) ++m.copies;
                } else b.cp.clear();
            }
            b.toy.kv.resize(reuse);b.toy.mtp.resize(reuse);b.live.resize(reuse);
            if(point>=0) {b.toy.s=b.payload[point];++m.copies;}
            ++b.epoch;b.cp.rewind(reuse,b.generation,b.epoch);
        } else b.reset();
        b.cp.learning=learned;b.cp.plan(reuse,in.size(),512);
        const auto count=std::min(in.size()-reuse,stop);
        // Cancellation is accepted at a completed chunk, as in the engine.
        assert(count==in.size()-reuse || count%512==0);
        for(size_t done=0;done<count;) {
            const auto n=std::min(size_t{512},count-done);
            for(size_t j=0;j<n;++j) b.toy.step(in[reuse+done+j]);
            done+=n;
            if(full<0 && in.size()-reuse>1) {
                const auto pos=reuse+done;const int i=b.cp.placement(pos);
                if(i>=0) {
                    b.cp.retire(i); // simulate completed device copy+sync
                    b.payload[i]=b.toy.s;b.cp.publish(i,{pos,b.generation,b.epoch});
                    ++m.captures;++m.copies;
                }
            }
        }
        b.live.assign(in.begin(),in.begin()+reuse+count);
        Toy fresh;for(int t:b.live) fresh.step(t);
        assert(b.toy.s==fresh.s && b.toy.kv==fresh.kv && b.toy.mtp==fresh.mtp);
        for(auto p:b.cp.points) assert(!p.pos || (p.pos<=b.live.size() && p.generation==b.generation && p.epoch==b.epoch));
        m.replay+=count;m.reused+=reuse;
        return reuse;
    }
};
static Tokens base() {Tokens a(20000);std::iota(a.begin(),a.end(),1);return a;}
static Tokens edit(Tokens a,size_t at) {a.at(at)+=50000;return a;}
static Tokens augment(Tokens a,size_t at,size_t count=2407) {
    a.insert(a.begin()+at,count,70001);return a;
}
static void compare(const char* name,const std::vector<Tokens>& sequence,bool improvement,bool cross=false) {
    Runner fixed(CP::Policy::fixed),adaptive(CP::Policy::adaptive);
    for(size_t i=0;i<sequence.size();++i) {
        fixed.request(sequence[i],cross ? i%2 : 0);adaptive.request(sequence[i],cross ? i%2 : 0);
    }
    if(improvement) assert(adaptive.m.reused>0 && adaptive.m.replay<fixed.m.replay);
    else assert(adaptive.m.replay==fixed.m.replay && adaptive.m.copies==fixed.m.copies);
    std::cout<<"TOY_SEQUENCE "<<name<<" turns="<<sequence.size()
             <<" fixed_replayed="<<fixed.m.replay<<" adaptive_replayed="<<adaptive.m.replay
             <<" fixed_captures="<<fixed.m.captures<<" adaptive_captures="<<adaptive.m.captures
             <<" fixed_payload_copies="<<fixed.m.copies<<" adaptive_payload_copies="<<adaptive.m.copies
             <<" rewrite_observations="<<adaptive.m.rewrites<<'\n';
}
// Frozen a9837161 placement oracle: successful fixed-policy capture positions
// must remain identical, including irregular natural chunk grids and tiny caps.
static int legacy(const CP& cp,uint64_t pos) {
    if(cp.points.empty() || pos<cp.interval) return -1;
    for(auto p:cp.points) if(p.pos==pos) return -1;
    if(!cp.points[0].pos) return 0;
    for(size_t i=1;i+1<cp.points.size();++i)
        if(!cp.points[i].pos && pos>=kv_mul(cp.interval,uint64_t{1}<<i)) return (int)i;
    uint64_t last=0;for(auto p:cp.points) last=std::max(last,p.pos);
    return cp.points.size()>1 && pos>=last && pos-last>=cp.interval ? (int)cp.points.size()-1 : -1;
}
int main() {
    for(size_t cap=0;cap<=4;++cap) for(uint64_t interval:{1u,127u,1024u}) for(uint64_t step:{1u,511u,512u,777u}) {
        CP fixed;fixed.points.resize(cap);fixed.interval=interval;
        for(uint64_t pos=step;pos<100000;pos+=step) {
            const int i=fixed.placement(pos);assert(i==legacy(fixed,pos));
            if(i>=0) fixed.publish(i,{pos,1,1});
        }
    }
    assert(CP::parse_policy("fixed")==CP::Policy::fixed);
    assert(CP::parse_policy("adaptive")==CP::Policy::adaptive);
    bool rejected=false;try {CP::parse_policy("guess");} catch(const std::invalid_argument&) {rejected=true;}
    assert(rejected);
    auto a=base();auto b=augment(a,14777);auto c=edit(a,6917);
    std::vector<Tokens> user{a,b,a,b,a,b,a};
    compare("user_augmentation_restoration",user,true);
    compare("system_edit",{a,c,a,c,a,c,a},true);
    std::vector<Tokens> combined{a,b,c,augment(c,14777),a,b,c,augment(c,14777),a,b,c};
    compare("combined_repeated",combined,true);
    compare("cross_slot",combined,true,true);
    std::vector<Tokens> append{a};
    for(int i=0;i<8;++i) {auto next=append.back();next.insert(next.end(),700,80000+i);append.push_back(next);}
    compare("append_only_tool_conversation",append,false);
    auto removed=a;removed.erase(removed.begin()+11800,removed.begin()+14000);
    auto summary=a;summary.erase(summary.begin()+9300,summary.begin()+15000);summary.insert(summary.begin()+9300,100,88);
    compare("tool_rewrite_remove_edit_summary",{a,edit(a,11777),removed,a,summary,a,summary,a,summary,a},true);
    auto template_edit=edit(a,2700);auto tools_edit=edit(a,333);
    compare("tools_template_move_earlier",{a,b,a,b,template_edit,a,template_edit,a,tools_edit,a,tools_edit,a},true);
    // No useful retained point: a first-turn miss cannot manufacture past state.
    Runner warm(CP::Policy::adaptive);
    warm.request(a);assert(warm.request(c)==4096);
    assert(warm.request(a)==6656); // learning-turn capture becomes useful next time
    assert(warm.request(edit(a,10))==0); // real mismatch before ANY anchor
    for(size_t cap=0;cap<=4;++cap) {
        Runner limited(CP::Policy::adaptive,cap);
        for(auto& in:combined) limited.request(in);
        for(auto& s:limited.slots) assert(s.cp.points.size()==cap);
        if(!cap) assert(limited.m.reused==0);
    }
    Runner cancel(CP::Policy::adaptive);
    cancel.request(a,0,8192); // cancel at safe natural boundary
    auto prefix=a;prefix.resize(8192);
    assert(cancel.slots[0].live==prefix);
    cancel.request(c);cancel.slots[0].reset();assert(cancel.request(c)==0);

    CP cp;cp.policy=CP::Policy::adaptive;cp.points.resize(4);
    cp.learning.observe(13917,20000,512);cp.plan(1536,20000,512);
    assert(cp.targets[0]==13824); // actual future end on suffix grid
    cp.plan(1234,20000,512);assert(cp.targets[0]==13522); // never silently splits chunks
    cp.points={{1024,1,1},{6656,1,1},{13824,1,1},{19456,1,1}};
    assert(cp.best(6917,20000,1,1)==1);
    assert(cp.best_available(15000,20000,1,1,[](int i){return i!=2;})==1);
    assert(cp.best_available(15000,20000,1,1,[](int){return false;})<0);
    assert(cp.best(6655,20000,1,1)==0); // NEVER reuse beyond changed tokens
    assert(cp.best(20000,20000,2,1)<0 && cp.best(20000,20000,1,2)<0);
    auto old=cp.retire(1);assert(old.pos==6656 && cp.best(6917,20000,1,1)==0);
    // Simulated copy failure: no publication; old identity MUST remain unavailable.
    assert(cp.points[1].pos==0);cp.publish(1,{6656,1,1});assert(cp.best(6917,20000,1,1)==1);
    cp.rewind(6656,2,2);assert(cp.best(20000,20000,1,1)<0);
    cp.clear();assert(cp.learning.rewrites==0 && cp.best(20000,20000,2,2)<0);

    CP::Learning learning;
    for(int i=0;i<10000;++i) assert(learning.observe(6000,20000,512));
    auto before=learning;
    for(int i=0;i<10000;++i) assert(!learning.observe(20000,20000,512));
    assert(learning.rewrites==before.rewrites); // append-only neither trains nor ages
    for(auto h:learning.hints) assert(h.hits<=8);
    for(uint64_t i=0;i<10000;++i) learning.observe(1000+i*512,10000000,512);
    assert(learning.hints.size()==8);
    bool stale=false;for(auto h:learning.hints) stale|=h.hits && h.common==6000;assert(!stale);
    cp.learning=learning;cp.plan(0,10000000,512);
    assert(cp.targets.size()==2 && sizeof(CP::Learning)<=256);
    // Learned point in rolling bank must be promoted before that bank rolls.
    cp.points={{1024,1,1},{2048,1,1},{4096,1,1},{6656,1,1}};
    cp.targets={6656,0};assert(cp.placement(6656)==1);
    std::cout<<"Q36_ADAPTIVE_CHECKPOINT_TEST PASS (CPU toy; no GPU timings or Qwen numerics)\n";
}
