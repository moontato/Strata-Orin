#include "strata/core/q36_cache_diagnostics.hpp"
#include <cassert>
#include <iostream>
#include <cstring>
using namespace strata::core;
struct Image {int64_t cell,n;uint64_t hash;bool operator==(const Image& o)const{return cell==o.cell && n==o.n && hash==o.hash;}};
int main() {
    std::vector<int32_t> old{3,9,9,9,9,4},ids=old;ids.push_back(7);ids.push_back(8);
    std::vector<Image> a{{1,4,17}},b=a;
    std::vector<Q36CacheImageDetail> x{{1,4,2,2,17,77,51,true}},y=x;
    auto check=[&](){return q36_cache_reason(old,ids,a,b,&x,&y);};
    const auto copy=old,copy_ids=ids;const auto before_a=a;
    auto d=check();assert(d.lcp==6 && d.proper && d.tokens && d.images && d.complete && d.details);
    assert(std::string(d.reason())=="exact_prefix");
    ids[2]++;d=check();assert(d.lcp==2 && !d.tokens && std::string(d.reason())=="token_divergence");ids=copy_ids;
    ids=old;d=check();assert(!d.proper && std::string(d.reason())=="not_proper_extension");ids=copy_ids;
    b[0].hash++;y[0].rows++;d=check();assert(d.identity && !d.images && d.rows && !d.grid && !d.key);
    b=a;y=x;y[0].nx=4;y[0].ny=1;b[0].hash++;d=check();assert(d.grid && !d.rows && !d.images);
    b=a;y=x;y[0].key++;b[0].hash++;d=check();assert(d.key && !d.rows && !d.grid);
    b=a;y=x;b[0].cell++;d=check();assert(d.placement && !d.images);
    b=a;y=x;old.resize(3);a.clear();x.clear();d=check();assert(!d.complete && !d.images && std::string(d.reason())=="incomplete_image_boundary");
    old.clear();d=check();assert(std::string(d.reason())=="empty");
    old=copy;ids=copy_ids;a=before_a;b=a;x={{1,4,2,2,17,77,51,true}};y=x;
    d=q36_cache_reason(old,ids,a,b);assert(!d.details);
    const auto line=q36_cache_line(1,0,d,6,"idle",2,3,6,false,true,true,true,-2,-2,true);
    assert(line.size()<1200 && line.back()=='\n');assert(line.find("lcp=6")!=std::string::npos);
    assert(line.find("actual_match=6")!=std::string::npos && line.find("idle_backed=1")!=std::string::npos);
    assert(old==copy && ids==copy_ids && a==before_a);
    std::cout<<"cache diagnostic classification/privacy bounds/read-only inputs PASS\n";
}
