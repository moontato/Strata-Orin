#include "strata/core/q36_checkpoint_schedule.hpp"
#include <cassert>
#include <iostream>
#include <functional>
using namespace strata::core;
static bool rejects(const std::function<void()>& f) {try {f();return false;} catch(const std::exception&) {return true;}}
// Schedule-sensitive CPU stand-in. Not GPU/model evidence. An ordinary singleton
// has no MTP side effect; declaring a longer call changes head/scratch semantics.
struct FakeReference {
    std::vector<int> state;std::vector<uint64_t> ends;bool cleared=false;
    void empty(){state.clear();ends.clear();cleared=true;}
    void call(const Q36ScheduleCall& c,size_t) {
        assert(cleared);
        for(auto e:c.ends) {state.push_back((int)(e-c.start));ends.push_back(e);}
        state.push_back(c.take==c.ids.size()?91:17); // head at end of DECLARED call
        state.push_back(c.ids.size()!=1 || c.checkpoint_suffix ? 37:0); // MTP
    }
};
int main() {
    Q36ScheduleBook b;std::vector<int32_t> ids(51);for(size_t i=0;i<ids.size();++i) ids[i]=(int32_t)i;
    b.begin(0,ids.data(),51,false);b.progress(0,16);b.progress(0,32);b.capture(0,0);
    b.progress(0,48);b.capture(0,1);b.progress(0,51);
    assert(b.saved[0][0][0].ids.size()==51 && b.saved[0][0][0].take==32);
    assert(b.live[0][0].take==51);
    b.restore(0,0,1);assert(q36_schedule_size(b.live[1])==32);
    b.begin(1,ids.data()+32,19,true);b.progress(1,16);b.progress(1,19);
    q36_schedule_check(b.live[1],ids,16);
    assert(b.saved[1][1].empty()); // later checkpoint cannot survive rewind
    FakeReference a,c;q36_schedule_replay(a,b.live[1]);q36_schedule_replay(c,b.live[1]);assert(a.state==c.state);
    auto wrong=b.live[1];wrong[0].ids.resize(32);q36_schedule_replay(c,wrong);
    assert(a.state!=c.state); // lost producer declared length/head detected
    wrong=b.live[1];wrong[0].ids[0]++;
    assert(rejects([&]{q36_schedule_check(wrong,ids,16);}));
    wrong=b.live[1];wrong[0].ends[0]++;
    assert(rejects([&]{q36_schedule_check(wrong,ids,16);}));
    wrong=b.live[1];wrong[1].start++;
    assert(rejects([&]{q36_schedule_check(wrong,ids,16);}));
    wrong=b.live[1];wrong[0].take--;
    assert(rejects([&]{q36_schedule_check(wrong,ids,16);}));
    b.restore(0,0,0);assert(q36_schedule_size(b.live[0])==32); // same-slot safe
    b.copy(1,2);b.reset(1);assert(q36_schedule_size(b.live[2])==51 && b.saved[2].empty());
    int32_t one=7;b.begin(1,&one,1,false);b.progress(1,1);assert(!q36_schedule_mtp(b.live[1])[0]);
    b.begin(1,&one,1,true);b.progress(1,1);assert(q36_schedule_mtp(b.live[1])[1]);
    assert(rejects([&]{b.progress(1,1);}));assert(rejects([&]{b.begin(1,&one,0,false);}));
    // The original 16384 / 29+22 design: NOT 512 individual process calls.
    Q36ScheduleBook direct;std::vector<int32_t> original(16435,198);uint64_t start=0;
    for(uint64_t n:{16384,29,22}) {
        direct.begin(0,original.data()+start,(int64_t)n,false);
        for(uint64_t d=512;d<n;d+=512) direct.progress(0,d);
        direct.progress(0,n);start+=n;
    }
    q36_schedule_check(direct.live[0],original,512);
    assert(direct.live[0].size()==3 && direct.live[0][0].ends.size()==32);
    std::cout<<"schedule metadata/independent adapter/negative controls PASS\n";
}
