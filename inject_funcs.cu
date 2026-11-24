// Device functions NVBit injects into the instrumented kernel (see common.h for the happens-before model):
//   syncsan_bar_pre / syncsan_bar_post   before / after every BAR.SYNC, BAR.RED (BAR.ARV: before only)
//   syncsan_mbar_arrive                  before every mbarrier arrive (SYNCS.ARRIVE*)
//   syncsan_ldgsts_dst                   before every cp.async (LDGSTS), for its shared-memory destination
//   syncsan_fence                        after every MEMBAR and CCTL.IVALL
//   syncsan_mem                          before every global (or generic) load, store, atomic and cp.async source
//   syncsan_bulk_copy                    before every UBLKCP (cp.async.bulk, non-tensor, either direction)
//   syncsan_tma                          before every UTMALDG / UTMASTG / UTMAREDG (cp.async.bulk.tensor, tiled)
//   syncsan_async_wait                   after every DEPBAR (cp.async.bulk.wait_group) and mbarrier try_wait
#include <stdint.h>
#include "utils/utils.h"
#include "nvbit_reg_rw.h"
#include "common.h"

// A launch's block, folded onto the S->nb blocks syncsan tracks (nb = min(blocks, MAX_CTAS)). Blocks b and b + n share a slot
// when a grid has more blocks than that: they look like one block, which only adds orderings (a missed race between the
// two, never a false one). Two blocks sharing a slot would have to be resident at once to mix their per-thread state, and
// n = 8192 is about twice the most blocks an H100 holds (132 SMs x 32).
__device__ __forceinline__ uint32_t linear_cta(const DevState* S) {
  const int4 c = get_ctaid(), n = get_nctaid();
  return (uint32_t)(c.x + n.x * (c.y + n.y * c.z)) % S->nb;
}
__device__ __forceinline__ uint32_t linear_tid() {
  uint32_t tx, ty, tz, nx, ny;
  asm("mov.u32 %0, %%tid.x;" : "=r"(tx)); asm("mov.u32 %0, %%tid.y;" : "=r"(ty)); asm("mov.u32 %0, %%tid.z;" : "=r"(tz));
  asm("mov.u32 %0, %%ntid.x;" : "=r"(nx)); asm("mov.u32 %0, %%ntid.y;" : "=r"(ny));
  return tx + nx * (ty + ny * tz);
}
__device__ __forceinline__ uint32_t my_warp(const DevState* S) { return linear_cta(S) * 32 + (linear_tid() >> 5); }   // launch-wide warp id
__device__ __forceinline__ uint32_t my_thread(const DevState* S) { return linear_cta(S) * 1024 + linear_tid(); }      // launch-wide thread slot

// An access, packed: valid:1 | write:1 | sync:1 | warp-in-block:5 | block:13 | barrier epoch:16 | block clock:27
constexpr unsigned long long VALID = 1ull << 63, F_WRITE = 1ull << 62, F_SYNC = 1ull << 61;
constexpr uint32_t CLOCK_MASK = (1u << 27) - 1, EPOCH_MASK = 0xFFFF;
__device__ __forceinline__ unsigned long long pack(bool write, bool sync, uint32_t wib, uint32_t block, uint32_t epoch, uint32_t clock) {
  return VALID | (write ? F_WRITE : 0) | (sync ? F_SYNC : 0) | ((unsigned long long)(wib & 31) << 56) |
         ((unsigned long long)(block & 0x1FFF) << 43) | ((unsigned long long)(epoch & EPOCH_MASK) << 27) | (clock & CLOCK_MASK);
}
__device__ __forceinline__ uint32_t p_wib(unsigned long long p) { return (uint32_t)(p >> 56) & 31; }
__device__ __forceinline__ uint32_t p_block(unsigned long long p) { return (uint32_t)(p >> 43) & 0x1FFF; }
__device__ __forceinline__ uint32_t p_epoch(unsigned long long p) { return (uint32_t)(p >> 27) & EPOCH_MASK; }
__device__ __forceinline__ uint32_t p_clock(unsigned long long p) { return (uint32_t)p & CLOCK_MASK; }
__device__ __forceinline__ uint32_t p_warp(unsigned long long p) { return p_block(p) * 32 + p_wib(p); }   // launch-wide, for reports

// Clocks are per block (v2): block b's row holds, for each block, the latest of its releases b has acquired. Warps of one
// block share the row, and are ordered among themselves by barrier epochs.
__device__ __forceinline__ uint32_t* clock_of(const DevState* S, uint32_t block) { return S->vc + (size_t)block * S->n; }

// Join src into dst entrywise, by the lanes in `lanes` working together (each takes every 32nd entry).
__device__ void join(const DevState* S, uint32_t* dst, const uint32_t* src, unsigned lanes) {
  const uint32_t lane = get_laneid(), rank = __popc(lanes & ((1u << lane) - 1)), width = __popc(lanes);
  for (uint32_t i = rank; i < S->n; i += width) { const uint32_t v = *(volatile const uint32_t*)(src + i); if (v) atomicMax(dst + i, v); }
}

__device__ __forceinline__ void prof(DevState* S, uint32_t slot, unsigned long long v = 1) { if (S->prof) atomicAdd(S->prof + slot, v); }

__device__ void report(DevState* S, uint64_t addr, uint32_t iid_prev, uint32_t iid_cur, unsigned long long prev,
                       unsigned long long cur, uint32_t kind) {
  const unsigned int i = atomicAdd(S->n_records, 1u);
  if (i >= MAX_RECORDS) return;
  RaceRecord& r = S->records[i];
  r.addr = addr; r.iid_prev = iid_prev; r.iid_cur = iid_cur; r.warp_prev = p_warp(prev); r.warp_cur = p_warp(cur); r.kind = kind;
  r.space = 0;
  __threadfence_system();
  r.ready = 1;
}

__device__ __forceinline__ uint32_t slot_hash(unsigned long long key, uint32_t bits) { return (uint32_t)((key * 0x9E3779B97F4A7C15ull) >> (64 - bits)); }

__device__ ShadowEntry* shadow_of(DevState* S, uint64_t word) {
  const unsigned long long key = word + 1; const uint32_t mask = (1u << S->shadow_bits) - 1;
  uint32_t idx = slot_hash(key, S->shadow_bits);
  for (uint32_t p = 0; p < PROBE; ++p, idx = (idx + 1) & mask) {
    ShadowEntry* e = S->shadow + idx;
    const unsigned long long k = *(volatile unsigned long long*)&e->key;
    if (k == key) return e;
    if (k == 0) { const unsigned long long old = atomicCAS(&e->key, 0ull, key); if (old == 0 || old == key) return e; }
  }
  atomicAdd(S->n_dropped, 1u);
  return nullptr;
}

// A release/acquire location's entry, created on first use with a zeroed clock (create = false: only look it up). When
// the clock pool is full the entry is marked NO_SLOT and the location isn't tracked; it never blocks a later lookup.
__device__ SyncEntry* sync_entry(DevState* S, uint64_t word, bool create) {
  const unsigned long long key = word + 1; const uint32_t mask = (1u << SYNC_BITS) - 1;
  uint32_t idx = slot_hash(key, SYNC_BITS);
  for (uint32_t p = 0; p < PROBE; ++p, idx = (idx + 1) & mask) {
    SyncEntry* e = S->sync + idx;
    unsigned long long k = *(volatile unsigned long long*)&e->key;
    if (k == 0 && create) {
      const unsigned long long old = atomicCAS(&e->key, 0ull, key);
      if (old == 0) {   // we own the entry: give it a zeroed clock, or mark it untracked
        const unsigned int s = atomicAdd(S->pool_used, 1u);
        if ((uint64_t)(s + 1) * S->n > S->pool_words) { atomicAdd(S->n_sync_dropped, 1u); e->slot = NO_SLOT; }
        else { uint32_t* c = S->pool + (size_t)s * S->n; for (uint32_t i = 0; i < S->n; ++i) c[i] = 0; e->slot = s; }
        __threadfence(); e->ready = 1;
        return e;
      }
      k = old;
    }
    if (k == key) { while (!e->ready) {} return e; }
    if (k == 0) return nullptr;
  }
  if (create) atomicAdd(S->n_sync_dropped, 1u);
  return nullptr;
}

__device__ __forceinline__ uint32_t* clock_at(const DevState* S, const SyncEntry* e) {
  return (!e || e->slot == NO_SLOT) ? nullptr : S->pool + (size_t)e->slot * S->n;
}

// Same block: another warp's access is ordered only by a barrier in between (a different epoch). Another block: by an
// edge that brought its clock into this block's row.
__device__ __forceinline__ bool ordered(const DevState* S, unsigned long long prev, uint32_t block, uint32_t wib, uint32_t epoch) {
  const uint32_t pb = p_block(prev);
  if (pb == block) return p_wib(prev) == wib || p_epoch(prev) != epoch;
  // A copy (block = an agent slot, wib/epoch = the issuing warp's) against an earlier access by its own block: ordered after
  // it as the issuing warp is, by program order or a barrier in between.
  if (block >= S->nb && pb == block - S->nb) return p_wib(prev) == wib || p_epoch(prev) != epoch;
  if (pb == S->nb + block && block < S->nb &&           // this block's copy agent: this warp may have waited on it itself
      *(volatile const uint32_t*)(S->warp_agent + block * 32 + wib) >= p_clock(prev)) return true;
  return (*(volatile const uint32_t*)(clock_of(S, block) + pb) & CLOCK_MASK) >= p_clock(prev);
}

__device__ void sm_bar_acquire(DevState* S, uint32_t id);

// After BAR.SYNC / BAR.RED id: the warp has passed one more barrier (global memory: epochs), and has seen what every
// participant released into the barrier's clock before it (shared memory).
extern "C" __device__ __noinline__ void syncsan_bar_post(int pred, uint32_t id, uint64_t pstate) {
  DevState* S = (DevState*)pstate;
  if (!pred) return;
  const unsigned lanes = __activemask();
  if (get_laneid() == (unsigned)(__ffs(lanes) - 1)) {
    prof(S, PROF_BAR); atomicAdd(S->epoch + my_warp(S), 1u);
    sm_bar_acquire(S, id);
  }
  __syncwarp(lanes);
}

extern "C" __device__ __noinline__ void syncsan_fence(int pred, uint32_t kind, uint64_t pstate) {
  DevState* S = (DevState*)pstate;
  if (!pred) return;
  const uint32_t t = my_thread(S);
  if (kind != FENCE_CCTL) {
    S->fenced[t] = 1;                                   // the thread's next strong store or atomic is a release
    // what this warp's waits saw of the block's copy agent goes into the block's clock, so a release carries it
    const uint32_t b = linear_cta(S), v = S->warp_agent[my_warp(S)];
    if (v && S->n > S->nb) atomicMax(clock_of(S, b) + S->nb + b, v);
  }
  if (kind == FENCE_SC) {                               // SC fences synchronize in their total order: join the SC clock, then publish
    const unsigned lanes = __activemask();
    if (get_laneid() == (unsigned)(__ffs(lanes) - 1)) { prof(S, PROF_SC); prof(S, PROF_JOIN_ENTRIES, 2ull * S->n); }
    uint32_t* mine = clock_of(S, linear_cta(S));
    join(S, mine, S->sc_clock, lanes);
    __syncwarp(lanes);
    join(S, S->sc_clock, mine, lanes);
    __threadfence();
    __syncwarp(lanes);
  }
  const uint32_t np = S->pending_n[t];
  if (np) {                                             // strong loads before this fence: each was an acquire
    S->pending_n[t] = 0;
    uint32_t* mine = clock_of(S, linear_cta(S));
    for (uint32_t k = 0; k < (np < PENDING ? np : PENDING); ++k) {
      const unsigned long long pw = S->pending[(size_t)t * PENDING + k];
      SyncEntry* e = sync_entry(S, pw - 1, false);
      const uint32_t* c = clock_at(S, e);
      if (!c) continue;
      const uint32_t ver = e->version;                // read before joining: a release during the join bumps it again
      if (S->joined_slot[t] == e->slot + 1 && S->joined_ver[t] == ver) { prof(S, PROF_ACQ_SKIP); continue; }   // spin-polling: nothing new to join
      prof(S, PROF_ACQ_JOIN); prof(S, PROF_JOIN_ENTRIES, S->n);
      for (uint32_t i = 0; i < S->n; ++i) { const uint32_t v = c[i]; if (v) atomicMax(mine + i, v); }
      S->joined_slot[t] = e->slot + 1; S->joined_ver[t] = ver;
    }
  }
}

__device__ void shadow_check(DevState* S, uint32_t iid, bool write, bool sync, uint32_t size, uint64_t addr, uint32_t block,
                             uint32_t wib, uint32_t epoch, uint32_t clock);
__device__ void sm_access(DevState* S, uint32_t iid, bool write, bool sync, uint32_t size, uint32_t off);

__device__ __forceinline__ void check_access(DevState* S, uint32_t iid, uint32_t flags, uint32_t size, uint64_t addr) {
  if (size < 4) return;               // sub-word accesses are not tracked
  if ((flags & ACC_SHARED) || ((flags & ACC_GENERIC) && __isShared((const void*)addr))) {
    // a generic address in the shared window, or (LDS/STS through NVBit) already an offset into the block's shared memory
    const uint32_t off = __isShared((const void*)addr) ? (uint32_t)__cvta_generic_to_shared((const void*)addr) : (uint32_t)addr & 0xFFFFFFu;
    if (S->sm_vc) sm_access(S, iid, flags & ACC_WRITE, flags & (ACC_STRONG | ACC_ATOMIC), size, off);
    return;
  }
  if ((flags & ACC_GENERIC) && !__isGlobal((const void*)addr)) return;
  const uint32_t block = linear_cta(S), wib = linear_tid() >> 5, t = my_thread(S);
  uint32_t* mine = clock_of(S, block);
  const bool write = flags & ACC_WRITE, sync = flags & (ACC_STRONG | ACC_ATOMIC);
  if (sync && write && S->fenced[t]) {                 // release: the location now carries this warp's clock
    S->fenced[t] = 0;
    prof(S, PROF_REL); prof(S, PROF_JOIN_ENTRIES, S->n);
    SyncEntry* e = sync_entry(S, addr >> 2, true);
    uint32_t* c = clock_at(S, e);
    if (c) {
      for (uint32_t i = 0; i < S->n; ++i) { const uint32_t v = mine[i]; if (flags & ACC_ATOMIC) { if (v) atomicMax(c + i, v); } else c[i] = v; }
      __threadfence();
      atomicAdd((unsigned int*)&e->version, 1u);
    }
    atomicAdd(mine + block, 1u);
  }
  if (sync && (!write || (flags & ACC_ATOMIC))) {       // an acquire if a fence follows (a newer location replaces the oldest)
    // A location already pending isn't added again: a spin loop that polls an error flag on every pass would otherwise
    // push out the loads it actually waits on (Magni's admitter polls err_code while it waits on free-ring cells).
    const unsigned long long w = (addr >> 2) + 1;
    unsigned long long* pend = S->pending + (size_t)t * PENDING;
    const uint32_t n = S->pending_n[t], live = n < PENDING ? n : PENDING;
    bool seen = false;
    for (uint32_t k = 0; k < live; ++k) seen |= pend[k] == w;
    if (!seen) { pend[n % PENDING] = w; S->pending_n[t] = n + 1; }
  }
  // Lanes of a warp touching the same address are ordered among themselves (one warp): one of them checks for all.
  const unsigned peers = __match_any_sync(__activemask(), addr);
  if (get_laneid() != (unsigned)(__ffs(peers) - 1)) return;
  const uint32_t epoch = *(volatile uint32_t*)(S->epoch + my_warp(S));
  shadow_check(S, iid, write, sync, size, addr, block, wib, epoch, *(volatile uint32_t*)(mine + block));
}

// Check one access (by a warp, or by a block's copy agent with the issuing warp's wib and epoch) against the shadow of each word it covers,
// and leave it there.
__device__ __forceinline__ uint32_t wpage(uint64_t a) { return (uint32_t)((a >> 12) * 0x9E3779B1u) >> (32 - WPAGE_BITS); }
__device__ __forceinline__ void mark_written(DevState* S, uint64_t a) {
  const uint32_t p = wpage(a), bit = 1u << (p & 31);
  if (!(*(volatile uint32_t*)(S->wpages + (p >> 5)) & bit)) atomicOr(S->wpages + (p >> 5), bit);
}
__device__ __forceinline__ bool maybe_written(const DevState* S, uint64_t a) {
  const uint32_t p = wpage(a);
  return *(volatile const uint32_t*)(S->wpages + (p >> 5)) & (1u << (p & 31));
}

__device__ void shadow_check(DevState* S, uint32_t iid, bool write, bool sync, uint32_t size, uint64_t addr, uint32_t block,
                             uint32_t wib, uint32_t epoch, uint32_t clock) {
  prof(S, PROF_MEM); prof(S, PROF_MEM_WORDS, size / 4);
  if (write) for (uint64_t pg = addr >> 12; pg <= (addr + size - 1) >> 12; ++pg) mark_written(S, pg << 12);
  const unsigned long long cur = pack(write, sync, wib, block, epoch, clock);
  for (uint32_t w = 0; w < size / 4; ++w) {
    const uint64_t a = addr + 4ull * w;
    ShadowEntry* e = shadow_of(S, a >> 2);
    if (!e) continue;
    // Publish this access first, fence, then read the other kind: of a write and a read racing on one word, at least
    // one sees the other (a check-then-update order lets both miss when two blocks hit the word at the same moment).
    // When the entry already holds this very access (same warp, kind, barrier epoch and clock, as in a spin loop or a
    // warp re-reading a word), it was published and fenced before: only the check is left, and it needs no fence.
    if (write) {
      if (S->adversarial) e->old = *(volatile const uint32_t*)a;   // checked before the write lands: the value it replaces
      const unsigned long long held = *(volatile unsigned long long*)&e->writer;
      if (held != cur) {
        const uint32_t prev_iid = e->writer_iid;
        const unsigned long long pw = atomicExch(&e->writer, cur); e->writer_iid = iid;
        if ((pw & VALID) && !((pw & F_SYNC) && sync) && !ordered(S, pw, block, wib, epoch)) report(S, a, prev_iid, iid, pw, cur, 1u);
        __threadfence();
      } else { prof(S, PROF_SAME); }
      const unsigned long long pr = *(volatile unsigned long long*)&e->reader;
      if ((pr & VALID) && !((pr & F_SYNC) && sync) && !ordered(S, pr, block, wib, epoch)) report(S, a, e->reader_iid, iid, pr, cur, 2u);
    } else {
      const unsigned long long held = *(volatile unsigned long long*)&e->reader;
      if (held != cur) { atomicExch(&e->reader, cur); e->reader_iid = iid; __threadfence(); } else { prof(S, PROF_SAME); }
      const unsigned long long pw = *(volatile unsigned long long*)&e->writer;
      if ((pw & VALID) && !((pw & F_SYNC) && sync) && !ordered(S, pw, block, wib, epoch)) {
        report(S, a, e->writer_iid, iid, pw, cur, 0u);
        if (S->adversarial && block < S->nb && w < 4) {   // a warp's own load: remember what it may legally read instead
          const uint32_t t = my_thread(S);
          S->adv_mask[t] |= 1u << w; S->adv_old[t * 4 + w] = e->old;
        }
      }
    }
  }
}

extern "C" __device__ __noinline__ void syncsan_mem(int pred, uint32_t iid, uint32_t flags, uint32_t size, uint64_t addr,
                                                    uint64_t pstate) {
  if (pred) check_access((DevState*)pstate, iid, flags, size, addr);
}

// The same check, inserted before a run of loads that precedes the instruction (see syncsan.cu: NVBit's patch right after
// an in-flight load can corrupt that load's destination on sm_90). The address is rebuilt from the instruction's own
// operands, read where the call sits; the loads in between write none of them. pred is the instruction's predicate
// register (1 when it has none), negated when flags has ACC_PRED_NEG.
extern "C" __device__ __noinline__ void syncsan_mem_hoisted(int pred, uint32_t iid, uint32_t flags, uint32_t size, uint32_t ra_lo,
                                                            uint32_t ra_hi, uint32_t ur_lo, uint32_t ur_hi, uint32_t imm,
                                                            uint64_t pstate) {
  if ((flags & ACC_PRED_NEG) ? pred : !pred) return;
  const uint64_t addr = (((uint64_t)ra_hi << 32) | ra_lo) + (((uint64_t)ur_hi << 32) | ur_lo) + (uint64_t)(int64_t)(int32_t)imm;
  check_access((DevState*)pstate, iid, flags & ~ACC_PRED_NEG, size, addr);
}

// Takes syncsan_mem's arguments and does nothing (SYNCSAN_NOP=1): tells a fault NVBit's patching causes from one syncsan's
// checks cause.
extern "C" __device__ __noinline__ void syncsan_nop(int pred, uint32_t iid, uint32_t flags, uint32_t size, uint64_t addr, uint64_t pstate) {}

extern "C" __device__ __noinline__ void syncsan_bare() {}

// ---- async copies: the block's copy agent (see common.h) ----
__device__ __forceinline__ bool has_agents(const DevState* S) { return S->n > S->nb; }
// A shared-memory address as an offset into this block's shared memory. Generic addresses are converted; 32-bit
// shared::cluster addresses (what TMA and mbarrier operands carry in cluster launches: the CTA's rank in the upper bits,
// UPRMT URx, URrank, 0x654, URoff) are reduced to their offset, so a TMA's mbarrier and the try_wait on it match.
__device__ __forceinline__ uint32_t smem_off(uint64_t a) {
  return __isShared((const void*)a) ? (uint32_t)__cvta_generic_to_shared((const void*)a) : (uint32_t)a & 0xFFFFFFu;
}

// Issue: the agent starts from what the issuing block has seen, and ticks. Returns the copy's clock.
__device__ uint32_t agent_issue(DevState* S, uint32_t block) {
  const uint32_t a = S->nb + block;
  uint32_t* ar = clock_of(S, a); const uint32_t* br = clock_of(S, block);
  for (uint32_t i = 0; i < S->n; ++i) { const uint32_t v = *(volatile const uint32_t*)(br + i); if (v) atomicMax(ar + i, v); }
  return atomicAdd(ar + a, 1u) + 1;
}

// A wait a warp saw complete: everything the block's agent has done so far is ordered before what this warp does next,
// and before what the rest of the block does after its next barrier (bar_post) or this warp's next release (fence).
// (The agent's row holds nothing else the block hasn't seen: it only ever joins the block's own row.)
__device__ void agent_complete(DevState* S, uint32_t block) {
  const uint32_t a = S->nb + block, v = *(volatile const uint32_t*)(clock_of(S, a) + a);
  atomicMax(S->warp_agent + my_warp(S), v);
}

__device__ __forceinline__ bool leader() { return get_laneid() == (unsigned)(__ffs(__activemask()) - 1); }

// units: the size operand, in 16-byte units (cp.async.bulk of 256 bytes is UMOV UR5, 0x10; UBLKCP.G.S [UR6], [UR4], UR5).
// write: shared -> global (a write of global memory), else global -> shared (a read).
__device__ void sm_copy(DevState* S, uint32_t iid, bool to_shared, uint32_t smem, uint32_t mbar, uint32_t bytes);

extern "C" __device__ __noinline__ void syncsan_bulk_copy(int pred, uint32_t iid, uint32_t write, uint32_t lo, uint32_t hi, uint32_t units,
                                                         uint32_t smem, uint32_t mbar, uint64_t pstate) {
  DevState* S = (DevState*)pstate;
  if (!pred || !leader()) return;   // uniform operands: one copy per warp
  if (S->sm_vc) sm_copy(S, iid, write == 0, smem, mbar, units * 16);
  if (!has_agents(S)) return;
  const uint32_t block = linear_cta(S), c = agent_issue(S, block);
  shadow_check(S, iid, write != 0, false, units * 16, ((uint64_t)hi << 32) | lo, S->nb + block, linear_tid() >> 5,
               *(volatile uint32_t*)(S->epoch + my_warp(S)), c);
}

// DEPBAR (cp.async.bulk.wait_group, also cp.async's waits) or an mbarrier try_wait that returned true (ok != 0).
__device__ void sm_wait(DevState* S, bool mbar, uint32_t addr);

// DEPBAR (cp.async.bulk.wait_group, cp.async.wait_*: mbar = 0) or an mbarrier try_wait that returned true (ok != 0,
// mbar = 1, at shared address base + imm).
extern "C" __device__ __noinline__ void syncsan_async_wait(int pred, int ok, uint32_t mbar, uint64_t base, uint32_t imm, uint64_t pstate) {
  DevState* S = (DevState*)pstate;
  if (!pred || !ok || !leader()) return;
  if (has_agents(S)) agent_complete(S, linear_cta(S));
  if (S->sm_vc) sm_wait(S, mbar != 0, ((uint32_t)base + (uint32_t)(base >> 32) + imm) & 0xFFFFFFu);   // register + uniform + imm
}

__device__ const TmaDesc* tma_lookup(const DevState* S, uint64_t map) {
  unsigned long long h = 0xcbf29ce484222325ull;
  for (int i = 0; i < 16; ++i) { h ^= ((const unsigned long long*)map)[i]; h *= 0x100000001b3ull; }
  if (!h) h = 1;
  const uint32_t mask = (1u << TMA_BITS) - 1;
  for (uint32_t p = 0, i = (uint32_t)(h >> (64 - TMA_BITS)); p < 64; ++p, i = (i + 1) & mask) {
    const TmaDesc* d = S->tma + i;
    const unsigned long long k = *(volatile const unsigned long long*)&d->hash;
    if (k == h) return d;
    if (!k) return nullptr;
  }
  return nullptr;
}

// cp.async.bulk.tensor, tiled: kind 0 load (global -> shared), 1 store, 2 reduce (store-side atomic). The box at coordinates
// c[0..rank) is walked row by row (dimension 0 is contiguous), clipped to the tensor as the hardware clips it.
extern "C" __device__ __noinline__ void syncsan_tma(int pred, uint32_t iid, uint32_t kind, uint32_t map_lo, uint32_t map_hi, int32_t c0,
                                                   int32_t c1, int32_t c2, int32_t c3, int32_t c4, uint32_t smem, uint32_t mbar,
                                                   uint64_t pstate) {
  DevState* S = (DevState*)pstate;
  if (!pred || !leader()) return;
  const TmaDesc* d = tma_lookup(S, ((uint64_t)map_hi << 32) | map_lo);
  if (!d) { atomicAdd(S->n_dropped, 1u); return; }
  if (S->sm_vc) {   // the box lands in (or leaves from) shared memory densely: box elements x element size
    uint32_t box = (uint32_t)d->stride[0];
    for (uint32_t j = 0; j < d->rank; ++j) box *= d->box[j];
    sm_copy(S, iid, kind == 0, smem, mbar, box);
  }
  if (!has_agents(S)) return;
  const int32_t c[5] = {c0, c1, c2, c3, c4};
  const uint32_t block = linear_cta(S), clock = agent_issue(S, block), a = S->nb + block, wib = linear_tid() >> 5;
  const uint32_t epoch = *(volatile uint32_t*)(S->epoch + my_warp(S));
  const unsigned long long e = d->stride[0];
  const long long lo = c[0] < 0 ? 0 : c[0], hi = (long long)c[0] + d->box[0] > (long long)d->dim[0] ? (long long)d->dim[0] : (long long)c[0] + d->box[0];
  if (hi <= lo) return;
  uint32_t rows = 1;
  for (uint32_t j = 1; j < d->rank; ++j) rows *= d->box[j];
  for (uint32_t r = 0; r < rows; ++r) {
    unsigned long long off = d->base + lo * e;
    bool in = true;
    for (uint32_t j = 1, q = r; j < d->rank; ++j) {
      const long long k = (long long)c[j] + q % d->box[j]; q /= d->box[j];
      if (k < 0 || k >= (long long)d->dim[j]) { in = false; break; }
      off += k * d->stride[j];
    }
    // A TMA load is checked only where this launch has written (weights it streams in are never written, and checking
    // every byte of them is most of a run's time). It can miss a later write racing with the load on a fresh page.
    const uint32_t bytes = (uint32_t)((hi - lo) * e);
    bool written = kind != 0;
    for (uint64_t pg = off >> 12; in && !written && pg <= (off + bytes - 1) >> 12; ++pg) written = maybe_written(S, pg << 12);
    if (in && written)
      shadow_check(S, iid, kind != 0, kind == 2, bytes, off, a, wib, epoch, clock);
  }
}


// ======================================================================================================================
// Shared memory (v4): per-warp vector clocks inside each block, with the block's copy engine as two more slots.
// ======================================================================================================================
__device__ __forceinline__ uint32_t* sm_row(const DevState* S, uint32_t b, uint32_t slot) { return S->sm_vc + (size_t)b * SM_BLOCK + slot * SM_ROW; }
__device__ __forceinline__ void sm_join(uint32_t* dst, const uint32_t* src) {
  for (uint32_t i = 0; i < SM_SLOTS; ++i) { const uint32_t v = *(volatile const uint32_t*)(src + i); if (v) atomicMax(dst + i, v); }
}
// release: everything the slot has seen, and its own accesses so far, go into x; then the slot's own clock ticks
__device__ __forceinline__ void sm_release(const DevState* S, uint32_t b, uint32_t slot, uint32_t* x) {
  uint32_t* r = sm_row(S, b, slot); sm_join(x, r); atomicAdd(r + slot, 1u);
}
__device__ __forceinline__ void sm_acquire(const DevState* S, uint32_t b, uint32_t slot, const uint32_t* x) { sm_join(sm_row(S, b, slot), x); }

constexpr unsigned long long SM_VALID = 1ull << 63, SM_WRITE = 1ull << 62, SM_SYNC = 1ull << 61;
__device__ __forceinline__ uint32_t sm_slot_of(unsigned long long p) { return (uint32_t)(p >> 32) & 63; }

__device__ SmShadow* sm_shadow_of(DevState* S, unsigned long long key) {
  const uint32_t mask = (1u << S->sm_shadow_bits) - 1;
  uint32_t idx = slot_hash(key, S->sm_shadow_bits);
  for (uint32_t p = 0; p < PROBE; ++p, idx = (idx + 1) & mask) {
    SmShadow* e = S->sm_shadow + idx;
    const unsigned long long k = *(volatile unsigned long long*)&e->key;
    if (k == key) return e;
    if (k == 0) { const unsigned long long old = atomicCAS(&e->key, 0ull, key); if (old == 0 || old == key) return e; }
  }
  atomicAdd(S->n_dropped, 1u);
  return nullptr;
}


__device__ void sm_report(DevState* S, uint32_t off, uint32_t iid_prev, uint32_t iid_cur, uint32_t b, unsigned long long prev,
                          uint32_t slot, uint32_t kind) {
  const unsigned int i = atomicAdd(S->n_records, 1u);
  if (i >= MAX_RECORDS) return;
  RaceRecord& r = S->records[i];
  r.addr = off; r.iid_prev = iid_prev; r.iid_cur = iid_cur; r.warp_prev = b * 64 + sm_slot_of(prev); r.warp_cur = b * 64 + slot;
  r.kind = kind; r.space = 1;
  __threadfence_system();
  r.ready = 1;
}

// One access by `slot` of block b at its clock `clock`, to [off, off + size) of the block's shared memory.
__device__ void sm_check(DevState* S, uint32_t iid, bool write, bool sync, uint32_t size, uint32_t off, uint32_t b, uint32_t slot,
                         uint32_t clock) {
  const uint32_t* me = sm_row(S, b, slot);
  const unsigned long long cur = SM_VALID | (write ? SM_WRITE : 0) | (sync ? SM_SYNC : 0) | ((unsigned long long)slot << 32) | clock;
  auto ordered = [&](unsigned long long p) {
    const uint32_t ps = sm_slot_of(p);
    return ps == slot || *(volatile const uint32_t*)(me + ps) >= (uint32_t)p;
  };
  for (uint32_t w = 0; w < size / 4; ++w) {
    const uint32_t o = off + 4 * w;
    SmShadow* e = sm_shadow_of(S, (((unsigned long long)b << 32) | (o >> 2)) + 1);
    if (!e) continue;
    if (write) {
      const unsigned long long held = *(volatile unsigned long long*)&e->writer;
      if (held != cur) {
        const uint32_t prev_iid = e->writer_iid;
        const unsigned long long pw = atomicExch(&e->writer, cur); e->writer_iid = iid;
        if ((pw & SM_VALID) && !((pw & SM_SYNC) && sync) && !ordered(pw)) sm_report(S, o, prev_iid, iid, b, pw, slot, 1u);
        __threadfence_block();
      }
      const unsigned long long pr = *(volatile unsigned long long*)&e->reader;
      if ((pr & SM_VALID) && !((pr & SM_SYNC) && sync) && !ordered(pr)) sm_report(S, o, e->reader_iid, iid, b, pr, slot, 2u);
    } else {
      const unsigned long long held = *(volatile unsigned long long*)&e->reader;
      if (held != cur) { atomicExch(&e->reader, cur); e->reader_iid = iid; __threadfence_block(); }
      const unsigned long long pw = *(volatile unsigned long long*)&e->writer;
      if ((pw & SM_VALID) && !((pw & SM_SYNC) && sync) && !ordered(pw)) sm_report(S, o, e->writer_iid, iid, b, pw, slot, 0u);
    }
  }
}

// A warp's own shared access (LDS / STS / ATOMS / generic to shared). Lanes of a warp on one word: one checks for all.
__device__ void sm_access(DevState* S, uint32_t iid, bool write, bool sync, uint32_t size, uint32_t off) {
  const unsigned peers = __match_any_sync(__activemask(), off);
  if (get_laneid() != (unsigned)(__ffs(peers) - 1)) return;
  const uint32_t b = linear_cta(S), w = linear_tid() >> 5;
  sm_check(S, iid, write, sync, size, off, b, w, *(volatile uint32_t*)(sm_row(S, b, w) + w));
}

// An async copy's shared-memory side, as the copy engine's slot: it starts from what the issuing warp has seen.
__device__ uint32_t sm_agent_issue(const DevState* S, uint32_t b, uint32_t w, uint32_t agent) {
  uint32_t* a = sm_row(S, b, agent);
  sm_join(a, sm_row(S, b, w));
  return atomicAdd(a + agent, 1u) + 1;
}

// The mbarrier's clock (created on first use in this block; untracked, returns null, when the block's table is full).
__device__ uint32_t* sm_mbar(const DevState* S, uint32_t b, uint32_t addr) {
  const unsigned long long key = (unsigned long long)addr + 1;
  unsigned long long* keys = S->sm_mbar_key + (size_t)b * SM_MBARS;
  for (uint32_t p = 0, i = (addr >> 3) % SM_MBARS; p < SM_MBARS; ++p, i = (i + 1) % SM_MBARS) {
    unsigned long long k = *(volatile unsigned long long*)(keys + i);
    if (k == 0) { k = atomicCAS(keys + i, 0ull, key); if (k == 0) k = key; }
    if (k == key) return S->sm_mbar + ((size_t)b * SM_MBARS + i) * SM_SLOTS;
  }
  atomicAdd(S->n_sync_dropped, 1u);
  return nullptr;
}


// Before BAR.SYNC / BAR.RED / BAR.ARV id: the warp releases into the barrier's clock.
extern "C" __device__ __noinline__ void syncsan_bar_pre(int pred, uint32_t neg, uint32_t id, uint64_t pstate) {
  DevState* S = (DevState*)pstate;
  if ((neg ? pred : !pred) || !leader()) return;
  const uint32_t b = linear_cta(S);
  // global: copies this warp waited for are, after the barrier, ordered before every warp of the block
  if (const uint32_t v = S->warp_agent[my_warp(S)]) if (S->n > S->nb) atomicMax(clock_of(S, b) + S->nb + b, v);
  if (!S->sm_vc) return;
  const uint32_t w = linear_tid() >> 5;
  if (id == 0xFFFFFFFFu) {   // the id register couldn't be read where the call sits: release into every barrier
    uint32_t* r = sm_row(S, b, w);
    for (uint32_t k = 0; k < SM_BARS; ++k) sm_join(S->sm_bar + ((size_t)b * SM_BARS + k) * SM_SLOTS, r);
    atomicAdd(r + w, 1u);
    return;
  }
  sm_release(S, b, w, S->sm_bar + ((size_t)b * SM_BARS + (id % SM_BARS)) * SM_SLOTS);
}

// mbarrier arrive (SYNCS.ARRIVE.*): a release into the mbarrier's clock.
extern "C" __device__ __noinline__ void syncsan_mbar_arrive(int pred, uint32_t neg, uint64_t base, uint32_t imm, uint64_t pstate) {
  DevState* S = (DevState*)pstate;
  if ((neg ? pred : !pred) || !S->sm_vc || !leader()) return;
  const uint32_t b = linear_cta(S);
  if (uint32_t* m = sm_mbar(S, b, ((uint32_t)base + (uint32_t)(base >> 32) + imm) & 0xFFFFFFu)) sm_release(S, b, linear_tid() >> 5, m);
}

// cp.async (LDGSTS) destination: the copy engine writes it, done at the issuing thread's DEPBAR.
extern "C" __device__ __noinline__ void syncsan_ldgsts_dst(int pred, uint32_t iid, uint32_t size, uint64_t addr, uint64_t pstate) {
  DevState* S = (DevState*)pstate;
  const unsigned act = __activemask();
  if (!S->sm_vc) return;
  const uint32_t b = linear_cta(S), w = linear_tid() >> 5;
  const int lead = __ffs(act) - 1;
  uint32_t c = 0;
  if ((int)get_laneid() == lead) c = sm_agent_issue(S, b, w, SM_AGENT_WAIT);
  c = __shfl_sync(act, c, lead);
  if (!pred) return;
  const uint32_t off = smem_off(addr);
  const unsigned peers = __match_any_sync(__activemask(), off);
  if (get_laneid() != (unsigned)(__ffs(peers) - 1)) return;
  sm_check(S, iid, true, false, size, off, b, SM_AGENT_WAIT, c);
}

__device__ void sm_bar_acquire(DevState* S, uint32_t id) {
  if (!S->sm_vc) return;
  const uint32_t b = linear_cta(S);
  sm_acquire(S, b, linear_tid() >> 5, S->sm_bar + ((size_t)b * SM_BARS + (id % SM_BARS)) * SM_SLOTS);
}

__device__ void sm_wait(DevState* S, bool mbar, uint32_t addr) {
  const uint32_t b = linear_cta(S), w = linear_tid() >> 5;
  if (!mbar) { sm_acquire(S, b, w, sm_row(S, b, SM_AGENT_WAIT)); return; }   // DEPBAR: the warp's cp.async / bulk-store reads are done
  if (const uint32_t* m = sm_mbar(S, b, addr)) sm_acquire(S, b, w, m);
}

// The shared side of a bulk or TMA copy issued by this warp. to_shared: a load, written by SM_AGENT_MBAR and completed
// through the mbarrier at `mbar` (its clock gets the agent's now, which waiters pick up once the phase completes);
// otherwise a store, read by SM_AGENT_WAIT and completed at the issuer's DEPBAR.
__device__ void sm_copy(DevState* S, uint32_t iid, bool to_shared, uint32_t smem, uint32_t mbar, uint32_t bytes) {
  const uint32_t b = linear_cta(S), w = linear_tid() >> 5, agent = to_shared ? SM_AGENT_MBAR : SM_AGENT_WAIT;
  const uint32_t c = sm_agent_issue(S, b, w, agent);
  const uint32_t off = smem_off(smem);
  // One thread checks a copy's whole box; a megakernel streams its weights through shared memory this way, so checking
  // every word makes it crawl. The copy is checked at one word per SM_COPY_STRIDE bytes: the warps that read or write the
  // box touch those words too, so a race with the copy is still seen there.
  for (uint32_t o = 0; o < bytes; o += SM_COPY_STRIDE) sm_check(S, iid, to_shared, false, 4, off + o, b, agent, c);
  if (to_shared) if (uint32_t* m = sm_mbar(S, b, smem_off(mbar))) sm_join(m, sm_row(S, b, agent));
}

// SYNCSAN_DELAY=offset:ns[,offset:ns...]: hold each warp this long right before the instruction at that offset, to turn a
// reported race into an observed one (delay the earlier access of a pair and see whether the later one reads stale data).
extern "C" __device__ __noinline__ void syncsan_delay(int pred, uint32_t ns) {
  if (!pred) return;
  const long long end = clock64() + (long long)ns * 2;   // ~2 cycles per ns at the clocks these GPUs run
  while (clock64() < end) __nanosleep(1000);
}

// SYNCSAN_ADVERSARIAL: a plain load not ordered after the last write of a word may, by the PTX memory model, still return
// the value that write replaced. This makes it do so: the load's address register is pointed at a per-thread copy holding
// the previous value of each racing word (the current value of the others). Only loads whose destination overwrites their
// own 64-bit address register are eligible (LDG.E.128 R8, [R8.64]): changing the register then leaves nothing behind.
// ra: the address register; the effective address is ra + (uniform offset + immediate), so the same delta is kept.
__device__ void adv_substitute(DevState* S, uint32_t t, uint32_t size, uint64_t addr, uint32_t ra) {
  const uint32_t m = S->adv_mask[t];
  if (!m || size > 16) return;
  uint32_t* sc = S->adv_scratch + (size_t)t * 4;
  // SYNCSAN_ADVERSARIAL=control: the same redirection with every word's current value (checks the mechanism alone)
  for (uint32_t w = 0; w < size / 4; ++w)
    sc[w] = ((m >> w) & 1) && S->adversarial == 1 ? S->adv_old[t * 4 + w] : *(volatile const uint32_t*)(addr + 4ull * w);
  __threadfence();
  const uint64_t base = ((uint64_t)(uint32_t)nvbit_read_reg(ra + 1) << 32) | (uint32_t)nvbit_read_reg(ra);
  const uint64_t moved = (uint64_t)sc - (addr - base);
  nvbit_write_reg(ra, (int32_t)(uint32_t)moved);
  nvbit_write_reg(ra + 1, (int32_t)(uint32_t)(moved >> 32));
  atomicAdd(S->adv_count, 1u);
}

extern "C" __device__ __noinline__ void syncsan_mem_adv(int pred, uint32_t iid, uint32_t flags, uint32_t size, uint64_t addr,
                                                        uint32_t ra, uint64_t pstate) {
  if (!pred) return;
  DevState* S = (DevState*)pstate;
  const uint32_t t = my_thread(S);
  S->adv_mask[t] = 0;
  check_access(S, iid, flags, size, addr);
  adv_substitute(S, t, size, addr, ra);
}

// The same, placed before a run of loads (the sm_90 hazard; none of those loads reads or writes ra).
extern "C" __device__ __noinline__ void syncsan_mem_hoisted_adv(int pred, uint32_t iid, uint32_t flags, uint32_t size, uint32_t ra_lo,
                                                                uint32_t ra_hi, uint32_t ur_lo, uint32_t ur_hi, uint32_t imm,
                                                                uint32_t ra, uint64_t pstate) {
  if ((flags & ACC_PRED_NEG) ? pred : !pred) return;
  DevState* S = (DevState*)pstate;
  const uint64_t addr = (((uint64_t)ra_hi << 32) | ra_lo) + (((uint64_t)ur_hi << 32) | ur_lo) + (uint64_t)(int64_t)(int32_t)imm;
  const uint32_t t = my_thread(S);
  S->adv_mask[t] = 0;
  check_access(S, iid, flags & ~ACC_PRED_NEG, size, addr);
  adv_substitute(S, t, size, addr, ra);
}

// SYNCSAN_ADVERSARIAL, general form: the original load is removed (nvbit_remove_orig) and done here instead, so its
// destination registers can be written directly. A word whose load is not ordered after the word's last write gets the
// value that write replaced (adversarial = 1); every other word, and every word under the control (adversarial = 2),
// gets what memory holds now. The load is done as a volatile (strong, relaxed) load, which returns a value the
// original load could also have returned. dst: the first destination register; size: 4, 8 or 16 bytes.
extern "C" __device__ __noinline__ void syncsan_load_emul(int pred, uint32_t iid, uint32_t flags, uint32_t size, uint64_t addr,
                                                          uint32_t dst, uint64_t pstate) {
  if (!pred) return;
  DevState* S = (DevState*)pstate;
  const uint32_t t = my_thread(S);
  S->adv_mask[t] = 0;
  check_access(S, iid, flags, size, addr);
  const uint32_t m = S->adversarial == 1 ? S->adv_mask[t] : 0;
  for (uint32_t w = 0; w < size / 4; ++w) {
    const uint32_t v = (m >> w) & 1 ? S->adv_old[t * 4 + w] : *(volatile const uint32_t*)(addr + 4ull * w);
    nvbit_write_reg(dst + w, (int32_t)v);
  }
  if (m) atomicAdd(S->adv_count, 1u);
  else if (S->adversarial == 2 && S->adv_mask[t]) atomicAdd(S->adv_count, 1u);   // control: loads that would have been changed
}
