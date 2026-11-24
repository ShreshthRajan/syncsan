// Shared between the host tool (syncsan.cu) and the injected device functions (inject_funcs.cu).
#pragma once
#include <stdint.h>

// Happens-before (v2). Each block b keeps a vector clock VC_b over the launch's N blocks, shared by its warps; within a
// block, warps are ordered by barrier epochs (each warp counts the full-block barriers it has passed). Edges:
//   barrier  BAR.SYNC / BAR.RED: the warp's epoch ticks. Two warps of one block are ordered iff a barrier separates
//            their accesses (different epochs). Synchronization between warps of one block without a barrier (an intra-
//            block flag) isn't modeled, and an acquire by one warp counts for its whole block (both trade precision for
//            joins N entries long instead of 32 N).
//   release  a strong store or atomic issued after a MEMBAR by the same thread: the location's clock becomes VC_g, then g ticks.
//   acquire  a strong load or atomic followed (same thread) by CCTL.IVALL or MEMBAR, which is how ld.acquire and
//            "volatile load; __threadfence()" compile: VC_g joins the location's clock.
//   fence.sc MEMBAR.SC (__threadfence): SC fences are totally ordered and each synchronizes with the later ones (PTX
//            memory model), so VC_g joins a launch-wide SC clock, then publishes its own clock into it.
// Async copies (v3). cp.async.bulk / TMA copies are done by the copy engine, modeled as one agent per block with a clock
// slot of its own (block slots 0..nb-1, agent slots nb..2nb-1). Issuing a copy joins the issuing block's clock into the
// agent's and ticks the agent; the copy's global bytes are checked as the agent's access. A warp that waits
// (cp.async.bulk.wait_group = DEPBAR for stores, a successful mbarrier try_wait for loads) is ordered after everything the
// agent did so far; the rest of its block is after its next barrier, and other blocks after its next release. A wait counts every copy the block issued as done (wait_group N > 0, several mbarriers): that can hide a race,
// never invent one.
// Every data access is checked against the last write and last read of its 4-byte word: an earlier access by warp p at
// clock t is ordered before the current one by warp g iff p == g or VC_g[p] >= t. Two unordered accesses, one a write,
// not both synchronizing (strong or atomic), are a race.

struct ShadowEntry {
  unsigned long long key;     // (word address >> 2) + 1, 0 = empty
  unsigned long long writer;  // packed access: valid:1 | write:1 | sync:1 | pad:1 | warp:28 | clock:32
  unsigned long long reader;
  uint32_t writer_iid, reader_iid;
  uint32_t old;               // SYNCSAN_ADVERSARIAL: the word's value before its last write
  uint32_t writer_s, reader_s;   // the accessing warp's own shared-clock entry at the access (warp-level ordering)
  uint32_t flags;             // OLD_BAD: a write the shadow did not track (sub-word) came after `old`; bits 8..15: last write's size
};
constexpr uint32_t OLD_BAD = 1;

struct SyncEntry {            // a location used for release/acquire, and the vector clock its last release left there
  unsigned long long key;
  uint32_t slot;              // index into the clock pool once ready != 0; NO_SLOT when the pool was full (not tracked)
  volatile uint32_t ready;
  volatile uint32_t version;  // bumped by every release, so an acquire that already joined this version can skip the join
  uint32_t pad;
};
constexpr uint32_t NO_SLOT = 0xFFFFFFFFu;

struct RaceRecord {
  uint64_t addr;
  uint32_t iid_prev, iid_cur;  // instruction ids of the earlier access (from the shadow) and the one that found the race
  uint32_t warp_prev, warp_cur;  // launch-wide warp ids (block = id / 32)
  uint32_t kind;               // 0 write->read, 1 write->write, 2 read->write
  uint32_t space;              // 0 global (warp ids as above), 1 shared (warp ids = block * 64 + slot, see SM_SLOTS)
  volatile uint32_t ready;     // set last: the host reads a record only once it is complete
};

constexpr uint32_t SHADOW_BITS_MIN = 23, SHADOW_BITS_MAX = 27;   // 8M-128M words (256 MB-4 GB), sized at start from free memory
constexpr uint32_t SYNC_BITS = 18;                   // 256K release/acquire locations
constexpr uint32_t PROBE = 32;
constexpr uint32_t MAX_CTAS = 8192;                  // larger grids are folded onto this many block slots (13-bit block field)
constexpr uint32_t MAX_WARPS = MAX_CTAS * 32;
constexpr uint64_t POOL_WORDS_MAX = 1ull << 31;      // clocks for release/acquire locations: up to 8 GB, sized at start from
                                                     // free device memory (DevState::pool_words), POOL_WORDS / n of them
constexpr uint32_t MAX_RECORDS = 1u << 14;
constexpr uint32_t PAIR_BITS = 16;
constexpr uint32_t PENDING = 8;
constexpr uint32_t WPAGE_BITS = 27;                  // hashed 4 KB pages written in this launch (16 MB bitmap)
constexpr uint32_t TMA_BITS = 12;                    // tensor maps syncsan can decode (a hash table in mapped host memory)
constexpr uint32_t AGENT_MAX_BLOCKS = MAX_CTAS / 2;
// Warp-level ordering between blocks (v0.9): launches of up to WO_MAX_BLOCKS blocks keep, for each warp, what it has
// acquired, and for each block release, which warps' accesses it publishes. Larger grids fall back to block-level clocks.
constexpr uint32_t WO_MAX_BLOCKS = 320, WO_N = 2 * WO_MAX_BLOCKS, WO_RELH = 64, WO_COV = 33;  // async-copy agents need a clock slot per block: grids up to this size

// ---- shared memory (v4) ----
// Each block keeps a vector clock per warp over SM_SLOTS slots: warps 0..31, then two async agents of the block's copy
// engine: SM_AGENT_WAIT (cp.async / LDGSTS writes and TMA / bulk store reads of shared memory, done at the issuing
// thread's DEPBAR) and SM_AGENT_MBAR (TMA / bulk loads into shared memory, done when their mbarrier phase completes).
// Edges: a full or named barrier (each participant releases into the barrier's clock before it and acquires after),
// mbarrier arrive (release into the mbarrier's clock) and a successful try_wait (acquire). Shared-memory accesses are
// checked against the last write and last read of each word of the block's shared memory.
constexpr uint32_t SM_SLOTS = 34, SM_AGENT_WAIT = 32, SM_AGENT_MBAR = 33;
constexpr uint32_t SM_ROW = SM_SLOTS, SM_BLOCK = (SM_SLOTS + 1) * SM_SLOTS;   // a padding row: diagonals at a fixed stride
constexpr uint32_t SM_BARS = 16, SM_MBARS = 64;      // named barriers; mbarriers tracked per block
constexpr uint32_t SM_SHADOW_BITS_MAX = 26;          // shared-memory shadow: sized at the first launch for its blocks x 228 KB, up to 64M words
constexpr uint32_t SM_COPY_STRIDE = 128;             // a copy's shared-memory side is checked at one word per 128 bytes

struct SmShadow {
  unsigned long long key;      // (block << 32 | word offset) + 1
  unsigned long long writer;   // valid:1 | write:1 | sync:1 | slot:6 | clock:32
  unsigned long long reader;
  uint32_t writer_iid, reader_iid;
};

// A tensor map, recorded by the host when cuTensorMapEncodeTiled returns: enough to turn a TMA instruction's coordinates
// into the global bytes it reads or writes.
struct TmaDesc {
  unsigned long long hash;   // FNV-1a of the map's 16 words, 0 = empty slot
  unsigned long long base;
  unsigned long long dim[5];
  unsigned long long stride[5];   // bytes; stride[0] is the element size
  uint32_t box[5];
  uint32_t rank;
};

struct DevState {
  ShadowEntry* shadow;
  SyncEntry* sync;
  uint32_t* vc;              // [MAX_CTAS][n]: each block's clock over the n blocks
  uint32_t* epoch;           // [MAX_WARPS]: full-block barriers each warp has passed
  uint32_t* sc_clock;        // [n]: the join of every clock at its fence.sc so far
  uint32_t* pool;            // [pool_words / n][n]
  unsigned long long pool_words;
  unsigned long long* pending;   // per thread, PENDING slots: 1 + word of each strong load since its last fence (a fence
                                 // after several, as in an unrolled poll of four ring cells, makes every one an acquire)
  uint32_t* pending_n;           // per thread: strong loads since its last fence
  uint32_t* fenced;          // per thread: a MEMBAR since its last release
  uint32_t* joined_slot;     // per thread: the location and version its last acquire joined
  uint32_t* joined_ver;
  uint32_t* wpages;          // [1 << WPAGE_BITS bits]: pages some write touched this launch (TMA loads skip the others)
  uint32_t wo;               // this launch uses warp-level ordering (nb <= WO_MAX_BLOCKS and shared clocks on)
  uint32_t* wrow;            // [WO_MAX_BLOCKS][32][n]: what each warp has acquired of each block's releases
  uint32_t* wbar;            // [WO_MAX_BLOCKS][SM_BARS][n]: carried across each named barrier
  uint32_t* wmbar;           // [WO_MAX_BLOCKS][SM_MBARS][n]: carried through each mbarrier
  uint32_t* relcov;          // [WO_MAX_BLOCKS][WO_RELH][WO_COV]: after the release that published block clock k, the
                             // latest own-clock entry of each warp that releases up to k cover; [32] = k + 1 (the tag)
  unsigned int* adv_mixed;   // injections whose load size differs from the last write's size (mapped)
  unsigned int* adv_torn;    // injections skipped because the word's last write changed while it was read (mapped)
  uint32_t* sm_vc;           // [nb][SM_BLOCK]: per block, SM_SLOTS rows of SM_ROW (and one padding row)
  uint32_t* sm_bar;          // [nb][SM_BARS][SM_SLOTS]
  unsigned long long* sm_mbar_key;   // [nb][SM_MBARS]: shared address + 1 of each tracked mbarrier
  uint32_t* sm_mbar;         // [nb][SM_MBARS][SM_SLOTS]
  SmShadow* sm_shadow;       // [1 << sm_shadow_bits]
  uint32_t sm_shadow_bits;
  uint32_t adversarial;      // SYNCSAN_ADVERSARIAL: racy plain loads return the value before the last write (see syncsan_mem_adv)
  uint32_t* adv_mask;        // [threads]: words of the current load that raced with their last write
  uint32_t* adv_old;         // [threads][4]: those words' previous values
  uint32_t* adv_scratch;     // [threads][4]: where a substituted load reads from
  unsigned int* adv_count;
  unsigned long long* pairs;  // [1 << PAIR_BITS]: instruction pairs already recorded (so a flood of one pair can't crowd out the rest)
  uint32_t watch_iid;        // SYNCSAN_WATCH=<offset>: log every read whose last writer is this instruction (watch_log)
  uint32_t* watch_log;       // mapped: [0] count, then [count][6] reader iid, writer block, writer clock, reader block, what the reader's clock holds of the writer, writes seen
   // mapped host memory: loads that returned a previous value (readable after the context is gone)
  uint32_t* warp_agent;      // [MAX_WARPS]: the latest of its block's agent clock each warp has waited for
  struct TmaDesc* tma;       // [1 << TMA_BITS], keyed by the hash of the 128-byte CUtensorMap
  uint32_t n;                // the length of every clock: nb block slots, then (when nb <= AGENT_MAX_BLOCKS) nb agent slots
  uint32_t nb;               // block slots in this launch: min(blocks, MAX_CTAS); larger grids fold onto them
  uint32_t shadow_bits;      // the shadow table holds 1 << shadow_bits words
  uint32_t threads_per_cta;
  unsigned int* pool_used;
  RaceRecord* records;
  unsigned int* n_records;   // may exceed MAX_RECORDS: the excess is counted, not stored
  unsigned int* n_dropped;   // accesses or locations not tracked because a table was full
  unsigned long long* prof;  // [PROF_N], null unless SYNCSAN_PROFILE
  unsigned int* n_sync_dropped;  // releases lost to a full clock pool or sync table: reports after them may be false
};

enum AccessFlags : uint32_t { ACC_WRITE = 1, ACC_STRONG = 2, ACC_ATOMIC = 4, ACC_GENERIC = 8, ACC_PRED_NEG = 16, ACC_SHARED = 32 };
// SYNCSAN_PROFILE=1: per-kind work counters, printed at exit
enum ProfSlot : uint32_t { PROF_MEM, PROF_MEM_WORDS, PROF_ACQ_JOIN, PROF_ACQ_SKIP, PROF_REL, PROF_SC, PROF_BAR, PROF_JOIN_ENTRIES, PROF_SAME, PROF_N };
enum FenceKind : uint32_t { FENCE_MEMBAR = 0, FENCE_CCTL = 1, FENCE_SC = 2 };
