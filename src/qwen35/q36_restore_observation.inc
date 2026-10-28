// Inside Engine. Developer-gate OBSERVATION BOUNDARY only. Nothing here runs in a
// normal engine invocation: with no hook installed no snapshot is requested, no
// counter moves and no throw is reachable. Production selection, caching, kernels
// and sampling are untouched.
//
// Why the boundary matters: the device rotary offset (d_delta) is ONE scalar bound
// to the CURRENT slot; each session's CPU copy (sess[s].rdelta) is mirrored into it
// only by select()/load_cur()/set_delta(). A complete-state donor copy fills another
// session's CPU fields without making that slot current, so observing state there
// mixes the destination's CPU fields with an unrelated current slot's device scalar,
// and a comparison against an independently replayed reference becomes meaningless.
//
// Contract for every restored/copied destination, with NO new input uploaded and no
// suffix processed in between:
//   copy state -> request_restore_snapshot(dst) -> select(dst) -> fire_restore_snapshot()
//   -> upload the incoming request's rows/table -> process its suffix.
// checkpoint_restore already selects dst before observing; copy_session defers.
int restore_snapshot_slot=-1;
size_t restore_snapshot_fires=0;    // observations actually taken
size_t restore_snapshot_deferred=0; // copy_session producers (consumed by fire)
size_t restore_snapshot_immediate=0;// checkpoint_restore producers (already current)
size_t restore_snapshot_refused=0;  // a producer whose boundary was violated (gate failure)

void discard_restore_snapshot() { restore_snapshot_slot=-1; }
void request_restore_snapshot(int dst) {
    if(!schedule_restored) return;
    restore_snapshot_slot=dst;++restore_snapshot_deferred;
}
/// The complete-state producer's consumer: only valid once dst is current, which
/// also means its CPU/device rotary state was loaded and no new input was uploaded.
bool fire_restore_snapshot() {
    const int dst=restore_snapshot_slot;
    restore_snapshot_slot=-1;
    if(dst<0 || !schedule_restored) return false;
    // A deferred producer MUST be consumed by the request that selected dst, while dst
    // is current. Refusing (instead of observing another slot's device state) is counted
    // against that producer, so fires+refused always equals deferred+immediate.
    if(cur!=dst) { ++restore_snapshot_refused;throw std::logic_error("restored-state snapshot for a non-current destination"); }
    ++restore_snapshot_fires;schedule_restored(dst);return true;
}
/// The checkpoint path's producer and consumer in one step: checkpoint_restore has
/// already selected dst and restored its CPU/device rotary state.
bool observe_restored_state(int dst) {
    if(!schedule_restored) return false;
    if(cur!=dst) throw std::logic_error("restored-state snapshot for a non-current destination");
    ++restore_snapshot_immediate;++restore_snapshot_fires;schedule_restored(dst);return true;
}
