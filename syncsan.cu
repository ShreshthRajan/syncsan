// syncsan: an NVBit tool that reports global-memory accesses no happens-before edge orders, within a block and across
// blocks (v1; the model is in common.h).
//   LD_PRELOAD=syncsan.so ./app
// Environment: SYNCSAN_KERNEL=<substring> instruments only matching kernels (default: all); SYNCSAN_VERBOSE=1 lists
// what gets instrumented. Reports go to stderr, one line per distinct pair of instructions, with source lines when the
// app was built with -lineinfo, and a summary at exit.
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <map>
#include <mutex>
#include <string>
#include <tuple>
#include <unordered_set>
#include <vector>
#include <unistd.h>
#include <stddef.h>
#include <algorithm>

#include "nvbit_tool.h"
#include "nvbit.h"
#include "common.h"

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "syncsan: %s: %s\n", #x, cudaGetErrorString(e_)); exit(1); } } while (0)

struct InstrInfo { std::string func, sass, where; uint32_t offset; };

static std::mutex g_mu;
static bool g_skip = false;
static int g_verbose = 0;
static std::string g_filter;
static bool g_do_mem = true, g_do_bar = true, g_do_fence = true;   // SYNCSAN_INSTRUMENT=mem,bar,fence (default all)
static std::vector<std::string> g_skip_ops;                         // SYNCSAN_SKIP_OPS=PREFIX,... or =OPCODE: memory opcodes left alone
static std::map<std::string, int> g_op_counts;
static bool g_shared = true;
static bool g_adv = false, g_adv_control = false;   // SYNCSAN_ADVERSARIAL=1 | control
static uint64_t g_adv_sites = 0;
static unsigned int* g_adv_count_host = nullptr;
static uint32_t g_first_nb = 1;   // blocks in the first instrumented launch (sizes the shared shadow)
static bool g_check_nc = false, g_async_tma = true, g_async_bulk = true, g_async_wait = true;
static uint32_t g_sample = 1, g_sample_seed = 0;
static uint64_t g_hoisted = 0, g_hazard_skipped = 0, g_folded = 0;
static std::string g_only_op;                    // SYNCSAN_ONLY_OP=OPCODE with SYNCSAN_NTH=lo:hi instruments only the lo-th..hi-1-th
static long g_nth_lo = 0, g_nth_hi = -1, g_nth = 0;
static bool g_after = false, g_nop = false;     // SYNCSAN_NOP=1: memory instructions call an empty function (experiment)
static bool g_bare = false;                      // SYNCSAN_BARE=1: the empty function takes no arguments at all                     // SYNCSAN_IPOINT=after: insert the memory check after the instruction (experiment)   // instance of that opcode (in instrumentation order), to find a bad one                      // SYNCSAN_VERBOSE=1 prints these at exit
static std::vector<InstrInfo> g_instrs;                 // index = instruction id passed to the device
static std::unordered_set<CUfunction> g_done;
static DevState g_host;                                 // device pointers (records and counters are mapped host memory)
static DevState* g_dev = nullptr;                       // the same struct, in device memory, passed to every injected call
static unsigned int g_seen = 0;                         // records already printed
static std::map<std::tuple<uint32_t, uint32_t, uint32_t>, uint64_t> g_pairs;   // (prev, cur, kind) -> count
static uint64_t g_launches = 0;
static int g_exitcode = 0;                              // SYNCSAN_EXITCODE=N: exit with N when a race was reported (CI, cuHadron)

void nvbit_at_init() {
  GET_VAR_INT(g_verbose, "SYNCSAN_VERBOSE", 0, "list instrumented functions and instructions");
  if (const char* f = getenv("SYNCSAN_KERNEL")) g_filter = f;
  if (const char* x = getenv("SYNCSAN_EXITCODE")) g_exitcode = atoi(x);
  if (const char* w = getenv("SYNCSAN_INSTRUMENT")) {
    const std::string v = w;
    g_do_mem = v.find("mem") != std::string::npos; g_do_bar = v.find("bar") != std::string::npos; g_do_fence = v.find("fence") != std::string::npos;
  }
  if (const char* o = getenv("SYNCSAN_ONLY_OP")) g_only_op = o;
  if (const char* r = getenv("SYNCSAN_NTH")) sscanf(r, "%ld:%ld", &g_nth_lo, &g_nth_hi);
  if (const char* ip = getenv("SYNCSAN_IPOINT")) g_after = !strcmp(ip, "after");
  g_check_nc = getenv("SYNCSAN_CHECK_NC") != nullptr;
  if (const char* sh = getenv("SYNCSAN_SHARED")) g_shared = strcmp(sh, "0") != 0;
  g_adv = getenv("SYNCSAN_ADVERSARIAL") && strcmp(getenv("SYNCSAN_ADVERSARIAL"), "0");
  g_adv_control = g_adv && !strcmp(getenv("SYNCSAN_ADVERSARIAL"), "control");
  if (const char* a = getenv("SYNCSAN_ASYNC")) { g_async_tma = strstr(a, "tma"); g_async_bulk = strstr(a, "bulk"); g_async_wait = strstr(a, "wait"); }
  if (const char* k = getenv("SYNCSAN_SAMPLE")) { g_sample = std::max(1, atoi(k)); if (const char* c = strchr(k, ':')) g_sample_seed = (uint32_t)atoi(c + 1); }
  g_nop = getenv("SYNCSAN_NOP") != nullptr; g_bare = getenv("SYNCSAN_BARE") != nullptr;
  if (const char* k = getenv("SYNCSAN_SKIP_OPS")) {
    std::string v = k; size_t a = 0;
    while (a <= v.size()) { size_t b = v.find(',', a); if (b == std::string::npos) b = v.size(); if (b > a) g_skip_ops.push_back(v.substr(a, b - a)); a = b + 1; }
  }
}

static void alloc_state() {
  if (g_dev) return;
  size_t free_b = 0, total_b = 0;
  CK(cudaMemGetInfo(&free_b, &total_b));
  uint32_t bits = SHADOW_BITS_MIN;   // the largest table within 15% of free memory
  while (bits < SHADOW_BITS_MAX - 1 && (sizeof(ShadowEntry) << (bits + 1)) <= free_b * 0.15) ++bits;   // 2 GB at most by default: it is cleared every launch
  if (const char* b = getenv("SYNCSAN_SHADOW_BITS")) bits = std::max(16, std::min((int)SHADOW_BITS_MAX, atoi(b)));
  g_host.shadow_bits = bits;
  CK(cudaMalloc(&g_host.shadow, sizeof(ShadowEntry) << bits));
  CK(cudaMalloc(&g_host.sync, sizeof(SyncEntry) << SYNC_BITS));
  CK(cudaMalloc(&g_host.vc, sizeof(uint32_t) * MAX_CTAS * MAX_CTAS));
  CK(cudaMalloc(&g_host.epoch, sizeof(uint32_t) * MAX_WARPS));
  CK(cudaMalloc(&g_host.sc_clock, sizeof(uint32_t) * MAX_CTAS));
  CK(cudaMemGetInfo(&free_b, &total_b));
  uint64_t words = std::min<uint64_t>(POOL_WORDS_MAX, (uint64_t)(free_b * 0.35) / sizeof(uint32_t));
  words = std::max<uint64_t>(words, 1ull << 24);
  CK(cudaMalloc(&g_host.pool, sizeof(uint32_t) * words));
  g_host.pool_words = words;
  if (g_verbose) fprintf(stderr, "syncsan: shadow %.1f GB (%u words), clock pool %.1f GB of %.1f GB free\n", (sizeof(ShadowEntry) << bits) / 1e9, 1u << bits, words * 4.0 / 1e9, free_b / 1e9);
  CK(cudaMalloc(&g_host.pending, sizeof(unsigned long long) * MAX_CTAS * 1024 * PENDING));
  CK(cudaMalloc(&g_host.pending_n, sizeof(uint32_t) * MAX_CTAS * 1024));
  CK(cudaMalloc(&g_host.fenced, sizeof(uint32_t) * MAX_CTAS * 1024));
  CK(cudaMalloc(&g_host.joined_slot, sizeof(uint32_t) * MAX_CTAS * 1024));
  CK(cudaMalloc(&g_host.joined_ver, sizeof(uint32_t) * MAX_CTAS * 1024));
  CK(cudaMalloc(&g_host.pool_used, sizeof(unsigned int)));
  CK(cudaMalloc(&g_host.warp_agent, sizeof(uint32_t) * MAX_WARPS));
  if (g_adv) {
    g_host.adversarial = g_adv_control ? 2 : 1;
    CK(cudaHostAlloc(&g_adv_count_host, sizeof(unsigned int), cudaHostAllocMapped));
    *g_adv_count_host = 0;
    CK(cudaMalloc(&g_host.adv_mask, sizeof(uint32_t) * (MAX_CTAS * 1024 + 1)));
    CK(cudaMemset(g_host.adv_mask, 0, sizeof(uint32_t) * (MAX_CTAS * 1024 + 1)));
    CK(cudaMalloc(&g_host.adv_old, sizeof(uint32_t) * MAX_CTAS * 1024 * 4));
    CK(cudaMalloc(&g_host.adv_scratch, sizeof(uint32_t) * MAX_CTAS * 1024 * 4));
  }
  if (g_shared) {   // shared-memory checking (v4): per-warp clocks, barrier and mbarrier clocks, a shadow of shared words
    CK(cudaMalloc(&g_host.sm_vc, sizeof(uint32_t) * MAX_CTAS * SM_BLOCK));
    CK(cudaMalloc(&g_host.sm_bar, sizeof(uint32_t) * MAX_CTAS * SM_BARS * SM_SLOTS));
    CK(cudaMalloc(&g_host.sm_mbar_key, sizeof(unsigned long long) * MAX_CTAS * SM_MBARS));
    CK(cudaMalloc(&g_host.sm_mbar, sizeof(uint32_t) * MAX_CTAS * SM_MBARS * SM_SLOTS));
    // twice the shared words the first launch's blocks can hold (228 KB each), so probes stay short: a full table makes
    // every shared access walk its probe sequence (a megakernel crawled with 4M entries for 132 blocks x 57K words)
    uint32_t bits = 22;
    while (bits < SM_SHADOW_BITS_MAX && (1ull << bits) < 2ull * g_first_nb * (228 * 1024 / 4)) ++bits;
    g_host.sm_shadow_bits = bits;
    CK(cudaMalloc(&g_host.sm_shadow, sizeof(SmShadow) << bits));
  }
  CK(cudaMalloc(&g_host.wpages, (1ull << WPAGE_BITS) / 8));
  CK(cudaHostAlloc(&g_host.tma, sizeof(TmaDesc) << TMA_BITS, cudaHostAllocMapped));   // the host adds maps as they're made
  memset(g_host.tma, 0, sizeof(TmaDesc) << TMA_BITS);
  CK(cudaHostAlloc(&g_host.records, sizeof(RaceRecord) * MAX_RECORDS, cudaHostAllocMapped));
  CK(cudaHostAlloc(&g_host.n_records, sizeof(unsigned int), cudaHostAllocMapped));
  CK(cudaHostAlloc(&g_host.n_dropped, sizeof(unsigned int), cudaHostAllocMapped));
  CK(cudaHostAlloc(&g_host.n_sync_dropped, sizeof(unsigned int), cudaHostAllocMapped));
  memset(g_host.records, 0, sizeof(RaceRecord) * MAX_RECORDS);
  *g_host.n_records = 0; *g_host.n_dropped = 0; *g_host.n_sync_dropped = 0;
  DevState d = g_host;   // the device sees mapped host memory through its device pointer
  CK(cudaHostGetDevicePointer((void**)&d.records, g_host.records, 0));
  CK(cudaHostGetDevicePointer((void**)&d.n_records, g_host.n_records, 0));
  CK(cudaHostGetDevicePointer((void**)&d.n_dropped, g_host.n_dropped, 0));
  CK(cudaHostGetDevicePointer((void**)&d.n_sync_dropped, g_host.n_sync_dropped, 0));
  CK(cudaHostGetDevicePointer((void**)&d.tma, g_host.tma, 0));
  if (g_adv_count_host) CK(cudaHostGetDevicePointer((void**)&d.adv_count, g_adv_count_host, 0));
  if (getenv("SYNCSAN_PROFILE")) {   // mapped host memory, like the records: readable at exit after the context is gone
    CK(cudaHostAlloc(&g_host.prof, sizeof(unsigned long long) * PROF_N, cudaHostAllocMapped));
    memset(g_host.prof, 0, sizeof(unsigned long long) * PROF_N);
    CK(cudaHostGetDevicePointer((void**)&d.prof, g_host.prof, 0));
  }
  CK(cudaMalloc(&g_dev, sizeof(DevState)));
  CK(cudaMemcpy(g_dev, &d, sizeof(DevState), cudaMemcpyHostToDevice));
}

// Ordered on the launch's stream after whatever ran there before: empty tables, zero clocks except each warp's own entry,
// which starts at 1 (so nothing another warp does counts as seen before an edge says so), and this launch's clock length.
static uint32_t g_nb = 0; static bool g_agents = false;   // the latest launch's, for naming warps in reports

static void reset_for_launch(cudaStream_t st, uint32_t grid) {
  const uint32_t nb = grid < MAX_CTAS ? grid : MAX_CTAS, ctas = nb;
  const uint32_t n = nb <= AGENT_MAX_BLOCKS ? 2 * nb : nb;   // block slots, then each block's copy agent
  g_nb = nb; g_agents = n > nb;
  CK(cudaMemsetAsync(g_host.shadow, 0, sizeof(ShadowEntry) << g_host.shadow_bits, st));
  CK(cudaMemsetAsync(g_host.sync, 0, sizeof(SyncEntry) << SYNC_BITS, st));
  CK(cudaMemsetAsync(g_host.vc, 0, sizeof(uint32_t) * (size_t)n * n, st));
  CK(cudaMemsetAsync(g_host.epoch, 0, sizeof(uint32_t) * (size_t)ctas * 32, st));
  CK(cudaMemsetAsync(g_host.sc_clock, 0, sizeof(uint32_t) * n, st));
  CK(cudaMemsetAsync(g_host.pending_n, 0, sizeof(uint32_t) * (size_t)ctas * 1024, st));
  CK(cudaMemsetAsync(g_host.fenced, 0, sizeof(uint32_t) * (size_t)ctas * 1024, st));
  CK(cudaMemsetAsync(g_host.joined_slot, 0, sizeof(uint32_t) * (size_t)ctas * 1024, st));
  CK(cudaMemsetAsync(g_host.pool_used, 0, sizeof(unsigned int), st));
  CK(cudaMemsetAsync(g_host.wpages, 0, (1ull << WPAGE_BITS) / 8, st));
  if (g_host.sm_vc) {
    CK(cudaMemsetAsync(g_host.sm_vc, 0, sizeof(uint32_t) * (size_t)nb * SM_BLOCK, st));
    CK(cudaMemsetAsync(g_host.sm_bar, 0, sizeof(uint32_t) * (size_t)nb * SM_BARS * SM_SLOTS, st));
    CK(cudaMemsetAsync(g_host.sm_mbar_key, 0, sizeof(unsigned long long) * (size_t)nb * SM_MBARS, st));
    CK(cudaMemsetAsync(g_host.sm_mbar, 0, sizeof(uint32_t) * (size_t)nb * SM_MBARS * SM_SLOTS, st));
    CK(cudaMemsetAsync(g_host.sm_shadow, 0, sizeof(SmShadow) << g_host.sm_shadow_bits, st));
    // each slot's own entry starts at 1: rows are SM_ROW long with a padding row per block, so the diagonals of all
    // blocks are every (SM_ROW + 1)-th word
    if (cuMemsetD2D32Async((CUdeviceptr)g_host.sm_vc, (size_t)(SM_ROW + 1) * 4, 1u, 1, (size_t)nb * SM_SLOTS, (CUstream)st) != CUDA_SUCCESS) {
      fprintf(stderr, "syncsan: cuMemsetD2D32Async (shared clocks) failed\n"); exit(1);
    }
  }
  CK(cudaMemsetAsync(g_host.warp_agent, 0, sizeof(uint32_t) * (size_t)ctas * 32, st));
  if (cuMemsetD2D32Async((CUdeviceptr)g_host.vc, (size_t)(n + 1) * 4, 1u, 1, n, (CUstream)st) != CUDA_SUCCESS) {
    fprintf(stderr, "syncsan: cuMemsetD2D32Async failed\n"); exit(1);
  }
  if (cuMemsetD32Async((CUdeviceptr)((char*)g_dev + offsetof(DevState, n)), n, 1, (CUstream)st) != CUDA_SUCCESS ||
      cuMemsetD32Async((CUdeviceptr)((char*)g_dev + offsetof(DevState, nb)), nb, 1, (CUstream)st) != CUDA_SUCCESS) {
    fprintf(stderr, "syncsan: cuMemsetD32Async failed\n"); exit(1);
  }
}

// Source location with its inlining chain for every offset of f, innermost first ("tma.cuh:74 < attention_partial.cu:577
// < attention_partial.cu:612 < storer.cuh:6"): which op of a megakernel an access belongs to, where NVBit's own line
// info gives only the innermost frame (often a library header). From nvdisasm -gi on the function's cubin; empty when
// nvdisasm isn't available or SYNCSAN_SYMBOLIZE=0.
static std::map<uint32_t, std::string> inline_chains(CUcontext ctx, CUfunction f) {
  std::map<uint32_t, std::string> out;
  const char* sym = getenv("SYNCSAN_SYMBOLIZE");
  if (sym && !strcmp(sym, "0")) return out;
  char cubin[64];
  snprintf(cubin, sizeof cubin, "/tmp/syncsan_%d_%p.cubin", (int)getpid(), (void*)f);
  if (!nvbit_dump_cubin(ctx, f, cubin)) return out;
  const char* nvd = getenv("NVDISASM") ? getenv("NVDISASM") : "nvdisasm";
  std::string cmd = std::string(nvd) + " -gi -c " + cubin + " 2>/dev/null";
  FILE* p = popen(cmd.c_str(), "r");
  if (!p) { unlink(cubin); return out; }
  std::vector<std::string> frames, pending;
  char buf[4096];
  const std::string section = std::string(".text.") + nvbit_get_func_name(ctx, f, true);   // offsets restart per function
  bool inside = false;
  auto base = [](const std::string& path) { const size_t k = path.rfind('/'); return k == std::string::npos ? path : path.substr(k + 1); };
  while (fgets(buf, sizeof buf, p)) {
    if (const char* sec = strstr(buf, ".section")) {
      const char* name = strstr(sec, ".text.");
      if (name) {
        std::string n(name); n = n.substr(0, n.find_first_of(",\n\t "));
        inside = n == section;
        frames.clear(); pending.clear();
      }
      continue;
    }
    if (!inside) continue;
    const char* hdr = strstr(buf, "//## File \"");
    if (hdr) {   // one hop: File "a", line n [inlined at "b", line m]
      char file[1024]; unsigned line = 0;
      if (sscanf(hdr, "//## File \"%1023[^\"]\", line %u", file, &line) == 2) pending.push_back(base(file) + ":" + std::to_string(line));
      continue;
    }
    const char* o = strstr(buf, "/*");
    unsigned off = 0;
    if (!o || sscanf(o, "/*%x*/", &off) != 1 || !strchr(buf, ';')) continue;
    if (!pending.empty()) { frames = pending; pending.clear(); }   // nvdisasm prints a chain only when it changes
    std::string chain;
    for (size_t k = 0; k < frames.size() && k < 6; ++k) {
      if (k && frames[k] == frames[k - 1]) continue;
      chain += (chain.empty() ? "" : " < ") + frames[k];
    }
    if (!chain.empty()) out[off] = chain;
  }
  pclose(p);
  unlink(cubin);
  return out;
}

static bool is_full_barrier(const std::string& op) {   // a CTA barrier a warp waits at: BAR.SYNC / BAR.RED (any id, any count)
  return op.rfind("BAR.", 0) == 0 && (op.find(".SYNC") != std::string::npos || op.find(".RED") != std::string::npos);
}

static void instrument(CUcontext ctx, CUfunction kernel) {
  std::vector<CUfunction> fs = nvbit_get_related_functions(ctx, kernel);
  fs.push_back(kernel);
  for (CUfunction f : fs) {
    if (!g_done.insert(f).second) continue;
    const char* fname = nvbit_get_func_name(ctx, f);
    if (g_verbose) fprintf(stderr, "syncsan: instrumenting %s\n", fname);
    std::map<uint32_t, std::string> chains = inline_chains(ctx, f);
    if (const char* dl = getenv("SYNCSAN_DELAY")) {   // offset:ns,...: delays before chosen instructions (see syncsan_delay)
      std::string v = dl; size_t a = 0;
      while (a < v.size()) {
        size_t b = v.find(',', a); if (b == std::string::npos) b = v.size();
        unsigned off = 0, ns = 0;
        if (sscanf(v.substr(a, b - a).c_str(), "%x:%u", &off, &ns) == 2)
          for (Instr* x : nvbit_get_instrs(ctx, f)) if (x->getOffset() == off) {
            nvbit_insert_call(x, "syncsan_delay", IPOINT_BEFORE);
            nvbit_add_call_arg_guard_pred_val(x);
            nvbit_add_call_arg_const_val32(x, ns);
            fprintf(stderr, "syncsan: delaying %s+0x%x (%s) by %u ns\n", fname, off, x->getSass(), ns);
          }
        a = b + 1;
      }
    }
    const std::vector<Instr*>& instrs = nvbit_get_instrs(ctx, f);
    // Only memory loads that complete out of order count (LDG, LD, LDL, LDS); constant loads (LDC, ULDC) never showed the
    // sm_90 hazard, and the compiler puts one right before most accesses to fetch the pointer that access then uses.
    auto in_flight = [](Instr* x) {
      if (!strncmp(x->getSass(), "@!PT ", 5)) return false;   // never issues (the compiler's padding before LDGSTS)
      const std::string o = x->getOpcode();
      return o.rfind("LDG", 0) == 0 || o.rfind("LD.", 0) == 0 || o == "LD" || o.rfind("LDL", 0) == 0 || o.rfind("LDS", 0) == 0;
    };
    for (size_t pos = 0; pos < instrs.size(); ++pos) {
      Instr* in = instrs[pos];
      const std::string op = in->getOpcode();
      const bool bar_arrive = op.rfind("BAR.ARV", 0) == 0;
      if (is_full_barrier(op) || bar_arrive) {
        if (!g_do_bar) continue;
        // barrier id: operand 0, an immediate or a register (BAR.SYNC.DEFER_BLOCKING 0x0, BAR.SYNC R2, R3, BAR.ARV 0x1, 0x80)
        const InstrType::operand_t* o = in->getNumOperands() ? in->getOperand(0) : nullptr;
        const bool reg_id = o && o->type == InstrType::OperandType::REG;
        const uint32_t imm_id = o && o->type == InstrType::OperandType::IMM_UINT64 ? (uint32_t)o->u.imm_uint64.value : 0;
        auto add_id = [&](const Instr* at) {
          if (reg_id) nvbit_add_call_arg_reg_val(at, o->u.reg.num); else nvbit_add_call_arg_const_val32(at, imm_id);
        };
        // before: release into the barrier (placed before a preceding run of in-flight loads: the sm_90 hazard)
        size_t j = pos;
        while (j > 0 && in_flight(instrs[j - 1])) --j;
        const Instr* at = instrs[j];
        bool ok = true;
        for (size_t k = j; reg_id && k < pos; ++k) {
          const InstrType::operand_t* d = instrs[k]->getNumOperands() ? instrs[k]->getOperand(0) : nullptr;
          if (d && d->type == InstrType::OperandType::REG && d->u.reg.num == o->u.reg.num) ok = false;
        }
        {   // when a load in the run overwrites the id register, the release goes to every barrier (id unknown)
          nvbit_insert_call(at, "syncsan_bar_pre", IPOINT_BEFORE);
          // the barrier's own guard, read where the call sits (@!P0 BAR.SYNC: runs when P0 is false)
          if (in->hasPred()) { if (in->isPredUniform()) nvbit_add_call_arg_upred_val_at(at, in->getPredNum()); else nvbit_add_call_arg_pred_val_at(at, in->getPredNum()); }
          else nvbit_add_call_arg_const_val32(at, 1);
          nvbit_add_call_arg_const_val32(at, in->hasPred() && in->isPredNeg() ? 1u : 0u);
          if (ok) add_id(at); else nvbit_add_call_arg_const_val32(at, 0xFFFFFFFFu);
          nvbit_add_call_arg_const_val64(at, (uint64_t)g_dev);
        }
        if (!bar_arrive) {
          nvbit_insert_call(in, "syncsan_bar_post", IPOINT_AFTER);
          nvbit_add_call_arg_guard_pred_val(in);
          add_id(in);
          nvbit_add_call_arg_const_val64(in, (uint64_t)g_dev);
        }
        continue;
      }
      const bool membar = op.rfind("MEMBAR", 0) == 0, cctl = op.rfind("CCTL", 0) == 0 && op.find("IVALL") != std::string::npos;
      if (membar || cctl) {
        if (!g_do_fence) continue;
        if (g_verbose > 1) fprintf(stderr, "syncsan:   fence %s\n", in->getSass());
        nvbit_insert_call(in, "syncsan_fence", IPOINT_AFTER);
        nvbit_add_call_arg_guard_pred_val(in);
        nvbit_add_call_arg_const_val32(in, !membar ? FENCE_CCTL : op.find(".SC") != std::string::npos ? FENCE_SC : FENCE_MEMBAR);
        nvbit_add_call_arg_const_val64(in, (uint64_t)g_dev);
        continue;
      }
      // "[URn]", "[Rn]", "[URn+0x10]": the shared address of an mbarrier operand, as (uniform?, register, immediate)
      // An mbarrier operand, "[R4+URZ+0x8]", "[UR5]", "[R2+UR6]": the shared address is the sum of its register, uniform
      // register and immediate terms (RZ / URZ are zero). (uniform is kept for the hazard check: the R register.)
      auto mbar_operand = [](const char* sass, bool& uniform, int& reg, int& ureg, uint32_t& imm) {
        const char* b = strchr(sass, '['); const char* e = b ? strchr(b, ']') : nullptr;
        if (!b || !e) return false;
        reg = -1; ureg = -1; imm = 0; uniform = false;
        std::string t(b + 1, e);
        size_t a = 0;
        while (a <= t.size()) {
          size_t k = t.find('+', a); if (k == std::string::npos) k = t.size();
          std::string term = t.substr(a, k - a);
          while (!term.empty() && term[0] == ' ') term.erase(0, 1);
          if (term.rfind("URZ", 0) == 0 || term.rfind("RZ", 0) == 0) {}
          else if (term.rfind("UR", 0) == 0) ureg = atoi(term.c_str() + 2);
          else if (term.rfind("R", 0) == 0) reg = atoi(term.c_str() + 1);
          else if (term.rfind("0x", 0) == 0 || term.rfind("-0x", 0) == 0) imm += (uint32_t)strtol(term.c_str(), nullptr, 0);
          else return false;
          a = k + 1;
        }
        uniform = reg < 0;
        return reg >= 0 || ureg >= 0;
      };
      auto add_mbar_args = [&](const Instr* at, bool, int reg, int ureg, uint32_t imm) {   // device side: base = reg | ureg << 32
        if (reg >= 0 && reg < 255) nvbit_add_call_arg_reg_val(at, reg); else nvbit_add_call_arg_const_val32(at, 0);
        if (ureg >= 0 && ureg < 63) nvbit_add_call_arg_ureg_val(at, ureg); else nvbit_add_call_arg_const_val32(at, 0);
        nvbit_add_call_arg_const_val32(at, imm);
      };
      if (op.rfind("SYNCS.ARRIVE", 0) == 0) {   // mbarrier arrive: a release into the mbarrier
        if (!g_do_bar || !g_shared) continue;
        bool u = false; int r = -1, ur = -1; uint32_t imm = 0;
        if (!mbar_operand(in->getSass(), u, r, ur, imm)) { if (g_verbose) fprintf(stderr, "syncsan: not modeled: %s\n", in->getSass()); continue; }
        size_t j = pos;   // before a preceding run of in-flight loads (the sm_90 hazard), if none of them writes the address register
        while (j > 0 && in_flight(instrs[j - 1])) --j;
        bool clobbered = false;
        for (size_t k = j; !u && k < pos; ++k) {
          const InstrType::operand_t* d = instrs[k]->getNumOperands() ? instrs[k]->getOperand(0) : nullptr;
          if (d && d->type == InstrType::OperandType::REG && d->u.reg.num <= r && r < d->u.reg.num + std::max(1, instrs[k]->getSize() / 4)) clobbered = true;
        }
        if (clobbered) { ++g_hazard_skipped; continue; }
        const Instr* at = instrs[j];
        nvbit_insert_call(at, "syncsan_mbar_arrive", IPOINT_BEFORE);
        if (in->hasPred()) { if (in->isPredUniform()) nvbit_add_call_arg_upred_val_at(at, in->getPredNum()); else nvbit_add_call_arg_pred_val_at(at, in->getPredNum()); }
        else nvbit_add_call_arg_const_val32(at, 1);
        nvbit_add_call_arg_const_val32(at, in->hasPred() && in->isPredNeg() ? 1u : 0u);
        add_mbar_args(at, u, r, ur, imm);
        nvbit_add_call_arg_const_val64(at, (uint64_t)g_dev);
        ++g_op_counts[op + " arrive"];
        continue;
      }
      const bool trywait = op.rfind("SYNCS.PHASECHK", 0) == 0;
      if (op.rfind("UBLKCP", 0) == 0 || op.rfind("DEPBAR", 0) == 0 || op.rfind("UTMA", 0) == 0 || trywait) {
        if (!g_do_mem || op == "UTMACMDFLUSH") continue;
        // SYNCSAN_ASYNC=tma,bulk,wait (default all): which async-copy instructions to model
        if (op.rfind("UTMA", 0) == 0 ? !g_async_tma : op.rfind("UBLKCP", 0) == 0 ? !g_async_bulk : !g_async_wait) continue;
        const char* sass = in->getSass();
        if (op.rfind("DEPBAR", 0) == 0 || trywait) {   // a wait: the block's copies are done (try_wait: when it returned true)
          int p = -1;
          bool u = false; int r = -1, ur = -1; uint32_t imm = 0;
          const char* sp = strchr(sass, ' ');   // SYNCS.PHASECHK.TRANS64[.TRYWAIT] Pn, [addr], ...
          if (trywait && (!sp || sscanf(sp, " P%d", &p) != 1 || !mbar_operand(sass, u, r, ur, imm))) {
            if (g_verbose) fprintf(stderr, "syncsan: not modeled: %s\n", sass);
            continue;
          }
          nvbit_insert_call(in, "syncsan_async_wait", IPOINT_AFTER);
          nvbit_add_call_arg_guard_pred_val(in);
          if (trywait) nvbit_add_call_arg_pred_val_at(in, p); else nvbit_add_call_arg_const_val32(in, 1);
          nvbit_add_call_arg_const_val32(in, trywait ? 1u : 0u);
          if (trywait) add_mbar_args(in, u, r, ur, imm);
          else { nvbit_add_call_arg_const_val32(in, 0); nvbit_add_call_arg_const_val32(in, 0); nvbit_add_call_arg_const_val32(in, 0); }
          nvbit_add_call_arg_const_val64(in, (uint64_t)g_dev);
          ++g_op_counts[op + " wait"];
          continue;
        }
        auto new_iid = [&]() {
          const uint32_t iid = (uint32_t)g_instrs.size();
          InstrInfo info{fname, sass, "", in->getOffset()};
          char *file = nullptr, *dir = nullptr; uint32_t line = 0;
          if (nvbit_get_line_info(ctx, f, in->getOffset(), &file, &dir, &line)) info.where = std::string(file) + ":" + std::to_string(line);
          if (chains.count(in->getOffset())) info.where = chains[in->getOffset()];
          g_instrs.push_back(info);
          return iid;
        };
        if (op.rfind("UTMA", 0) == 0) {
          // UTMALDG.kD [URs], [URmap] (URs+1 = mbarrier, coordinates from URs+2); UTMASTG.kD / UTMAREDG.kD [URs], [URmap]
          // (coordinates from URs+1). Other forms (im2col, prefetch, multicast operands) are not modeled.
          const int kind = op.rfind("UTMALDG", 0) == 0 ? 0 : op.rfind("UTMASTG", 0) == 0 ? 1 : op.rfind("UTMAREDG", 0) == 0 ? 2 : -1;
          int rank = 0, us = -1, um = -1;
          for (int r = 1; r <= 5; ++r) if (op.find("." + std::to_string(r) + "D") != std::string::npos) rank = r;
          const char* at = strstr(sass, "UTMA");
          if (kind < 0 || !rank || op.find("IM2COL") != std::string::npos || !at ||
              sscanf(strchr(at, ' '), " [UR%d], [UR%d]", &us, &um) != 2) {
            if (g_verbose) fprintf(stderr, "syncsan: not modeled: %s\n", sass);
            ++g_op_counts[op + " NOT MODELED"];
            continue;
          }
          const uint32_t iid = new_iid();
          ++g_op_counts[op + " tma"];
          if (g_verbose) fprintf(stderr, "syncsan: tma %s (smem UR%d, map UR%d, rank %d) +0x%x\n", sass, us, um, rank, in->getOffset());
          nvbit_insert_call(in, "syncsan_tma", IPOINT_BEFORE);
          nvbit_add_call_arg_guard_pred_val(in);
          nvbit_add_call_arg_const_val32(in, iid);
          nvbit_add_call_arg_const_val32(in, (uint32_t)kind);
          nvbit_add_call_arg_ureg_val(in, um);
          nvbit_add_call_arg_ureg_val(in, um + 1);
          const int c0 = us + (kind == 0 ? 2 : 1);
          for (int j = 0; j < 5; ++j) {
            if (j < rank && c0 + j < 63) nvbit_add_call_arg_ureg_val(in, c0 + j); else nvbit_add_call_arg_const_val32(in, 0);
          }
          nvbit_add_call_arg_ureg_val(in, us);                                   // shared side
          if (kind == 0 && us + 1 < 63) nvbit_add_call_arg_ureg_val(in, us + 1); else nvbit_add_call_arg_const_val32(in, 0);   // mbarrier
          nvbit_add_call_arg_const_val64(in, (uint64_t)g_dev);
          continue;
        }
        // UBLKCP.G.S [URglobal], [URshared], URunits: shared -> global (a write). UBLKCP.S.G [URshared], [URglobal], URunits:
        // global -> shared (a read, completed through the mbarrier at URshared + 1).
        int dst = -1, src = -1, bytes = -1;
        const bool to_global = op.find(".G.S") != std::string::npos, to_shared = op.find(".S.G") != std::string::npos;
        if ((!to_global && !to_shared) || sscanf(strstr(sass, "UBLKCP"), "%*s [UR%d], [UR%d], UR%d", &dst, &src, &bytes) != 3) {
          if (g_verbose) fprintf(stderr, "syncsan: not modeled: %s\n", sass);
          continue;
        }
        const int g = to_global ? dst : src;
        const uint32_t iid = new_iid();
        ++g_op_counts[op + " bulk copy"];
        nvbit_insert_call(in, "syncsan_bulk_copy", IPOINT_BEFORE);
        nvbit_add_call_arg_guard_pred_val(in);
        nvbit_add_call_arg_const_val32(in, iid);
        nvbit_add_call_arg_const_val32(in, to_global ? 1u : 0u);
        nvbit_add_call_arg_ureg_val(in, g);
        nvbit_add_call_arg_ureg_val(in, g + 1);
        nvbit_add_call_arg_ureg_val(in, bytes);
        nvbit_add_call_arg_ureg_val(in, to_global ? src : dst);                 // shared side
        if (!to_global && dst + 1 < 63) nvbit_add_call_arg_ureg_val(in, dst + 1); else nvbit_add_call_arg_const_val32(in, 0);   // mbarrier
        nvbit_add_call_arg_const_val64(in, (uint64_t)g_dev);
        continue;
      }
      const InstrType::MemorySpace ms = in->getMemorySpace();
      const bool global = ms == InstrType::MemorySpace::GLOBAL || ms == InstrType::MemorySpace::GLOBAL_TO_SHARED;
      const bool shared = ms == InstrType::MemorySpace::SHARED;
      if (!global && !shared && ms != InstrType::MemorySpace::GENERIC) continue;
      if (shared && !g_shared) continue;
      if (!g_do_mem) continue;
      bool skip = false;
      for (const std::string& pre : g_skip_ops)   // "=OP" skips exactly OP, anything else is a prefix
        if (pre[0] == '=' ? op == pre.substr(1) : op.rfind(pre, 0) == 0) skip = true;
      // ld.global.nc (LDG.E.CONSTANT): the program promises the data is read-only for the whole launch, so it can't race
      // unless that promise is broken. Not checked unless SYNCSAN_CHECK_NC=1, which saves a call on most weight loads.
      if (op.find(".CONSTANT") != std::string::npos && !g_check_nc) skip = true;
      // SYNCSAN_SAMPLE=k[:seed]: check one plain (weak) load or store site in k, chosen by its offset; strong accesses,
      // atomics, barriers and fences are always instrumented, so a skipped site only hides its own races, never adds one.
      // NVBit's call costs most of the time (an empty call per access is most of the slowdown), so this is the lever.
      const bool plain = op.find(".STRONG") == std::string::npos && op.find(".VOLATILE") == std::string::npos &&
                         op.rfind("ATOM", 0) != 0 && op.rfind("RED", 0) != 0;
      if (plain && g_sample > 1 && ((in->getOffset() / 16) * 2654435761u + g_sample_seed) % g_sample != 0) skip = true;
      if (skip) continue;
      if (!g_only_op.empty()) {
        if (op != g_only_op) continue;
        const long k = g_nth++;
        if (k < g_nth_lo || (g_nth_hi >= 0 && k >= g_nth_hi)) continue;
        if (g_verbose) fprintf(stderr, "syncsan: nth %ld: %s +0x%x %s\n", k, in->getSass(), in->getOffset(), fname);
      }
      ++g_op_counts[op + " " + InstrType::MemorySpaceStr[(int)ms] + " mrefs=" + std::to_string([&] { int n = 0; for (int i = 0; i < in->getNumOperands(); ++i) n += in->getOperand(i)->type == InstrType::OperandType::MREF; return n; }())];
      const bool atomic = op.rfind("ATOM", 0) == 0 || op.rfind("RED", 0) == 0;
      uint32_t flags = 0;
      if (in->isStore() || atomic) flags |= ACC_WRITE;
      if (atomic) flags |= ACC_ATOMIC;
      if (op.find(".STRONG") != std::string::npos || op.find(".VOLATILE") != std::string::npos) flags |= ACC_STRONG;
      if (ms == InstrType::MemorySpace::GENERIC) flags |= ACC_GENERIC;
      if (shared) flags |= ACC_SHARED;
      const uint32_t iid = (uint32_t)g_instrs.size();
      InstrInfo info{fname, in->getSass(), "", in->getOffset()};
      char *file = nullptr, *dir = nullptr; uint32_t line = 0;
      if (nvbit_get_line_info(ctx, f, in->getOffset(), &file, &dir, &line)) info.where = std::string(file) + ":" + std::to_string(line);
          if (chains.count(in->getOffset())) info.where = chains[in->getOffset()];
      g_instrs.push_back(info);
      if (g_verbose > 1) fprintf(stderr, "syncsan:   [%u] %s %s\n", iid, info.sass.c_str(), info.where.c_str());
      // On sm_90, NVBit's call patched in directly after an in-flight load can clobber that load's destination register
      // (an empty, argument-free call at mvm_exec_kernel+0x38620, after LDG.E R39, makes Magni VM fault). So the check
      // of an instruction that follows a run of loads is inserted before that run, rebuilding its address from operands
      // the loads don't write; where they do, the instruction is skipped and counted.
      // Loads never branch, so a check placed before the run always runs before the instruction (control entering the
      // run from elsewhere skips the check: a missed access, never a wrong one).
      size_t j = pos;
      // Only memory loads that complete out of order count (LDG, LD, LDL, LDS); constant loads (LDC, ULDC) never showed the
      // hazard, and the compiler puts one right before most accesses to fetch the pointer that access then uses.
      while (j > 0 && in_flight(instrs[j - 1])) --j;
      if (j < pos) {   // on every architecture: the same placement is correct, and A100 builds may hit the hazard too
        const InstrType::operand_t* m = nullptr;
        const bool copy = ms == InstrType::MemorySpace::GLOBAL_TO_SHARED;   // LDGSTS: the second reference is the global source
        for (int i = 0, seen = 0; i < in->getNumOperands(); ++i)
          if (in->getOperand(i)->type == InstrType::OperandType::MREF && seen++ == (copy ? 1 : 0)) { m = in->getOperand(i); break; }
        if (copy) flags &= ~ACC_WRITE;
        bool ok = m && m->u.mref.has_ra;
        const int ra = ok ? m->u.mref.ra_num : 0, ra_w = (ok && m->u.mref.ra_mod == InstrType::RegModifierType::U64) ? 2 : 1;
        for (size_t k = j; ok && k < pos; ++k) {   // a load in the run writing an address register makes the rebuild wrong
          const InstrType::operand_t* d = instrs[k]->getNumOperands() ? instrs[k]->getOperand(0) : nullptr;
          if (!d || d->type != InstrType::OperandType::REG) continue;
          const int lo = d->u.reg.num, w = std::max(1, instrs[k]->getSize() / 4);
          if (ra < lo + w && lo < ra + ra_w) ok = false;
        }
        if (!ok) { ++g_hazard_skipped; continue; }
        const Instr* at = instrs[j];
        // adversarial (see syncsan_mem_adv): also needs no load in the run to use ra as its address
        const InstrType::operand_t* d0 = in->getOperand(0);
        const int dst_lo = d0->type == InstrType::OperandType::REG ? d0->u.reg.num : -1, dst_n = std::max(1, in->getSize() / 4);
        bool adv_ok = g_adv && !copy && !(flags & (ACC_WRITE | ACC_STRONG | ACC_ATOMIC | ACC_GENERIC | ACC_SHARED)) &&
                      op.rfind("LDG", 0) == 0 && ra_w == 2 && dst_lo >= 0 && dst_lo <= ra && ra + 1 < dst_lo + dst_n && in->getSize() <= 16;
        for (size_t k = j; adv_ok && k < pos; ++k)
          for (int q = 0; q < instrs[k]->getNumOperands(); ++q) {
            const InstrType::operand_t* x = instrs[k]->getOperand(q);
            if (x->type == InstrType::OperandType::MREF && x->u.mref.has_ra && (x->u.mref.ra_num == ra || x->u.mref.ra_num == ra + 1)) adv_ok = false;
          }
        nvbit_insert_call(at, adv_ok ? "syncsan_mem_hoisted_adv" : "syncsan_mem_hoisted", IPOINT_BEFORE);
        uint32_t f = flags;
        if (in->hasPred()) {
          if (in->isPredUniform()) nvbit_add_call_arg_upred_val_at(at, in->getPredNum()); else nvbit_add_call_arg_pred_val_at(at, in->getPredNum());
          if (in->isPredNeg()) f |= ACC_PRED_NEG;
        } else {
          nvbit_add_call_arg_const_val32(at, 1);
        }
        nvbit_add_call_arg_const_val32(at, iid);
        nvbit_add_call_arg_const_val32(at, f);
        nvbit_add_call_arg_const_val32(at, (uint32_t)in->getSize());
        nvbit_add_call_arg_reg_val(at, ra);
        if (ra_w == 2) nvbit_add_call_arg_reg_val(at, ra + 1); else nvbit_add_call_arg_const_val32(at, 0);
        if (m->u.mref.has_ur) {
          nvbit_add_call_arg_ureg_val(at, m->u.mref.ur_num);
          if (m->u.mref.ur_mod == InstrType::RegModifierType::U64) nvbit_add_call_arg_ureg_val(at, m->u.mref.ur_num + 1); else nvbit_add_call_arg_const_val32(at, 0);
        } else {
          nvbit_add_call_arg_const_val32(at, 0); nvbit_add_call_arg_const_val32(at, 0);
        }
        nvbit_add_call_arg_const_val32(at, m->u.mref.has_imm ? (uint32_t)m->u.mref.imm : 0u);
        if (adv_ok) { nvbit_add_call_arg_const_val32(at, (uint32_t)ra); ++g_adv_sites; }
        nvbit_add_call_arg_const_val64(at, (uint64_t)g_dev);
        ++g_hoisted;
        continue;
      }
      // LDGSTS (cp.async) has a shared and a global reference; its global one is the source, a read.
      int mref = 0;
      for (int i = 0; i < in->getNumOperands(); ++i) {
        if (in->getOperand(i)->type != InstrType::OperandType::MREF) continue;
        const bool src_of_copy = ms == InstrType::MemorySpace::GLOBAL_TO_SHARED;
        if (src_of_copy && mref == 0 && g_shared && !g_bare && !g_nop) {   // its shared destination, written by the copy engine
          nvbit_insert_call(in, "syncsan_ldgsts_dst", IPOINT_BEFORE);
          nvbit_add_call_arg_guard_pred_val(in);
          nvbit_add_call_arg_const_val32(in, iid);
          nvbit_add_call_arg_const_val32(in, (uint32_t)in->getSize());
          nvbit_add_call_arg_mref_addr64(in, 0);
          nvbit_add_call_arg_const_val64(in, (uint64_t)g_dev);
        }
        if (!src_of_copy || mref == 1) {
          if (g_bare) { nvbit_insert_call(in, "syncsan_bare", g_after ? IPOINT_AFTER : IPOINT_BEFORE); ++mref; continue; }
          // adversarial: a weak global load whose destination overwrites its whole 64-bit address register
          const InstrType::operand_t* mo = in->getOperand(i);
          const InstrType::operand_t* d0 = in->getOperand(0);
          const int dst_lo = d0->type == InstrType::OperandType::REG ? d0->u.reg.num : -1, dst_n = std::max(1, in->getSize() / 4);
          const bool adv_ok = g_adv && !src_of_copy && !(flags & (ACC_WRITE | ACC_STRONG | ACC_ATOMIC | ACC_GENERIC | ACC_SHARED)) &&
                              op.rfind("LDG", 0) == 0 && mo->u.mref.has_ra && mo->u.mref.ra_mod == InstrType::RegModifierType::U64 &&
                              dst_lo >= 0 && dst_lo <= mo->u.mref.ra_num && mo->u.mref.ra_num + 1 < dst_lo + dst_n && in->getSize() <= 16;
          if (adv_ok) {
            nvbit_insert_call(in, "syncsan_mem_adv", IPOINT_BEFORE);
            nvbit_add_call_arg_guard_pred_val(in);
            nvbit_add_call_arg_const_val32(in, iid);
            nvbit_add_call_arg_const_val32(in, flags);
            nvbit_add_call_arg_const_val32(in, (uint32_t)in->getSize());
            nvbit_add_call_arg_mref_addr64(in, mref);
            nvbit_add_call_arg_const_val32(in, (uint32_t)mo->u.mref.ra_num);
            nvbit_add_call_arg_const_val64(in, (uint64_t)g_dev);
            ++g_adv_sites; ++mref;
            continue;
          }
          nvbit_insert_call(in, g_nop ? "syncsan_nop" : "syncsan_mem", g_after ? IPOINT_AFTER : IPOINT_BEFORE);
          nvbit_add_call_arg_guard_pred_val(in);
          nvbit_add_call_arg_const_val32(in, iid);
          nvbit_add_call_arg_const_val32(in, src_of_copy ? (flags & ~ACC_WRITE) : flags);
          nvbit_add_call_arg_const_val32(in, (uint32_t)in->getSize());
          nvbit_add_call_arg_mref_addr64(in, mref);
          nvbit_add_call_arg_const_val64(in, (uint64_t)g_dev);
        }
        ++mref;
      }
    }
  }
}

static const char* KIND[] = {"write then read", "write then write", "read then write"};

static void drain() {   // print records the device has written since the last call
  const unsigned int n = *(volatile unsigned int*)g_host.n_records;
  const unsigned int upto = n < MAX_RECORDS ? n : MAX_RECORDS;
  unsigned int i = g_seen;
  for (; i < upto; ++i) {
    if (!g_host.records[i].ready) break;   // still being written: pick it up on a later call
    const RaceRecord r = g_host.records[i];
    auto key = std::make_tuple(r.iid_prev, r.iid_cur, r.kind);
    if (g_pairs[key]++) continue;
    const InstrInfo& a = g_instrs[r.iid_prev]; const InstrInfo& b = g_instrs[r.iid_cur];
    const bool sh = r.space == 1;
    auto who = [sh](uint32_t w) {   // block slots past nb are copy agents (see common.h)
      char buf[96];
      if (sh) {
        const uint32_t b = w / 64, sl = w % 64;
        if (sl == SM_AGENT_WAIT) snprintf(buf, sizeof buf, "block %u copy engine (cp.async write / TMA-bulk store read)", b);
        else if (sl == SM_AGENT_MBAR) snprintf(buf, sizeof buf, "block %u copy engine (TMA / bulk load write)", b);
        else snprintf(buf, sizeof buf, "block %u warp %u", b, sl);
      } else if (g_agents && w / 32 >= g_nb) snprintf(buf, sizeof buf, "block %u copy engine (issued by warp %u)", w / 32 - g_nb, w % 32);
      else snprintf(buf, sizeof buf, "block %u warp %u", w / 32, w % 32);
      return std::string(buf);
    };
    fprintf(stderr,
            "syncsan: RACE (%s) at %s0x%llx\n"
            "  %s: %s  [%s+0x%x %s]\n"
            "  %s: %s  [%s+0x%x %s]\n"
            "  no barrier, release/acquire pair, async-copy wait or atomic orders these two accesses\n",
            KIND[r.kind], sh ? "shared " : "", (unsigned long long)r.addr, who(r.warp_prev).c_str(), a.sass.c_str(), a.func.c_str(), a.offset,
            a.where.c_str(), who(r.warp_cur).c_str(), b.sass.c_str(), b.func.c_str(), b.offset, b.where.c_str());
  }
  g_seen = i;
}

static bool wanted(CUcontext ctx, CUfunction f) {
  return g_filter.empty() || strstr(nvbit_get_func_name(ctx, f), g_filter.c_str()) != nullptr;
}

// Tensor maps the program encoded, waiting to go into the device-visible table (which exists once a launch is instrumented).
static std::vector<TmaDesc> g_tma_new;
static uint64_t g_tma_maps = 0, g_tma_skipped = 0;

static void record_tensor_map(const cuTensorMapEncodeTiled_params* p) {
  static const int bytes[] = {1, 2, 4, 4, 8, 8, 2, 4, 8, 2, 4, 4, 4};   // CU_TENSOR_MAP_DATA_TYPE_UINT8 .. TFLOAT32_FTZ
  const int dt = (int)p->tensorDataType;
  bool ok = dt >= 0 && dt < 13 && p->tensorRank >= 1 && p->tensorRank <= 5 && p->interleave == CU_TENSOR_MAP_INTERLEAVE_NONE;
  for (cuuint32_t j = 0; ok && j < p->tensorRank; ++j) ok = !p->elementStrides || p->elementStrides[j] == 1;
  if (!ok) { ++g_tma_skipped; return; }   // packed sub-byte types, interleaved layouts, strided boxes: not decoded
  TmaDesc d{};
  unsigned long long h = 0xcbf29ce484222325ull;
  for (int i = 0; i < 16; ++i) { h ^= ((const unsigned long long*)p->tensorMap)[i]; h *= 0x100000001b3ull; }
  d.hash = h ? h : 1;
  d.base = (unsigned long long)p->globalAddress;
  d.rank = p->tensorRank;
  d.stride[0] = (unsigned long long)bytes[dt];
  for (cuuint32_t j = 0; j < p->tensorRank; ++j) { d.dim[j] = p->globalDim[j]; d.box[j] = p->boxDim[j]; }
  for (cuuint32_t j = 1; j < p->tensorRank; ++j) d.stride[j] = p->globalStrides[j - 1];
  g_tma_new.push_back(d);
  ++g_tma_maps;
}

static void flush_tensor_maps() {   // mapped host memory: the device sees these in launches that start after this
  if (!g_host.tma) return;
  const uint32_t mask = (1u << TMA_BITS) - 1;
  for (const TmaDesc& d : g_tma_new) {
    for (uint32_t p = 0, i = (uint32_t)(d.hash >> (64 - TMA_BITS)); p < 64; ++p, i = (i + 1) & mask) {
      TmaDesc& e = g_host.tma[i];
      if (e.hash == d.hash) break;
      if (!e.hash) { TmaDesc t = d; t.hash = 0; e = t; __sync_synchronize(); *(volatile unsigned long long*)&e.hash = d.hash; break; }
    }
  }
  g_tma_new.clear();
}

void nvbit_at_cuda_event(CUcontext ctx, int is_exit, nvbit_api_cuda_t cbid, const char* name, void* params, CUresult* status) {
  std::lock_guard<std::mutex> lock(g_mu);
  if (g_skip) return;
  g_skip = true;
  CUfunction func = nullptr; CUstream stream = 0; uint64_t ctas = 0;
  switch (cbid) {
    case API_CUDA_cuLaunchKernel: case API_CUDA_cuLaunchKernel_ptsz:
    case API_CUDA_cuLaunchCooperativeKernel: case API_CUDA_cuLaunchCooperativeKernel_ptsz: {
      auto* p = (cuLaunchKernel_params*)params; func = p->f; stream = p->hStream;
      ctas = (uint64_t)p->gridDimX * p->gridDimY * p->gridDimZ; } break;
    case API_CUDA_cuLaunchKernelEx: case API_CUDA_cuLaunchKernelEx_ptsz: {
      auto* p = (cuLaunchKernelEx_params*)params; func = p->f; stream = p->config->hStream;
      ctas = (uint64_t)p->config->gridDimX * p->config->gridDimY * p->config->gridDimZ; } break;
    case API_CUDA_cuTensorMapEncodeTiled:
      if (is_exit && status && *status == CUDA_SUCCESS) record_tensor_map((const cuTensorMapEncodeTiled_params*)params);
      break;
    default: break;
  }
  if (func && !is_exit) {
    bool on = wanted(ctx, func);
    if (on && ctas > MAX_CTAS) {
      static std::unordered_set<CUfunction> warned;
      if (warned.insert(func).second && g_verbose)
        fprintf(stderr, "syncsan: %s: %llu blocks, folded onto %u (blocks b and b+%u look like one block to syncsan)\n",
                nvbit_get_func_name(ctx, func), (unsigned long long)ctas, MAX_CTAS, MAX_CTAS);
      ++g_folded;
    }
    if (on) {
      g_first_nb = (uint32_t)(ctas < MAX_CTAS ? ctas : MAX_CTAS);
      alloc_state();
      flush_tensor_maps();
      instrument(ctx, func);
      reset_for_launch((cudaStream_t)stream, (uint32_t)(ctas < MAX_CTAS ? ctas : MAX_CTAS));
      ++g_launches;
    }
    nvbit_enable_instrumented(ctx, func, on);
  }
  if (g_dev) drain();
  g_skip = false;
}

void nvbit_at_term() {
  if (g_verbose) for (auto& kv : g_op_counts) fprintf(stderr, "syncsan: instrumented %5d x %s\n", kv.second, kv.first.c_str());
  if (!g_dev) { fprintf(stderr, "syncsan: no instrumented launches\n"); return; }
  cudaDeviceSynchronize();
  drain();
  if (g_host.prof) {
    const unsigned long long* p = g_host.prof;
    fprintf(stderr, "syncsan: PROFILE memory checks %llu (words %llu), acquire joins %llu (skipped %llu), releases %llu, SC fences %llu, barriers %llu, clock entries joined %llu, words already holding the access %llu\n",
            p[PROF_MEM], p[PROF_MEM_WORDS], p[PROF_ACQ_JOIN], p[PROF_ACQ_SKIP], p[PROF_REL], p[PROF_SC], p[PROF_BAR], p[PROF_JOIN_ENTRIES], p[PROF_SAME]);
  }
  const unsigned int n = *g_host.n_records;
  if (g_hoisted || g_hazard_skipped)
    fprintf(stderr, "syncsan: %llu memory checks placed before a preceding run of loads (sm_90 NVBit hazard), %llu instructions left unchecked for it\n",
            (unsigned long long)g_hoisted, (unsigned long long)g_hazard_skipped);

  if (g_adv) {
    fprintf(stderr, "syncsan: ADVERSARIAL%s %llu load sites eligible, %u loads %s\n", g_adv_control ? " (control)" : "",
            (unsigned long long)g_adv_sites, g_adv_count_host ? *g_adv_count_host : 0u,
            g_adv_control ? "redirected with their current values" : "returned the value before a racing write");
  }
  if (g_tma_maps || g_tma_skipped)
    fprintf(stderr, "syncsan: %llu tensor maps decoded for TMA checks, %llu not (unsupported layout)\n",
            (unsigned long long)g_tma_maps, (unsigned long long)g_tma_skipped);
  if (g_folded)
    fprintf(stderr, "syncsan: %llu launches had more than %u blocks; blocks b and b+%u shared a slot (races between those two are not reported)\n",
            (unsigned long long)g_folded, MAX_CTAS, MAX_CTAS);
  if (*g_host.n_sync_dropped)
    fprintf(stderr, "syncsan: WARNING %u releases were not tracked (clock pool or sync table full): reports may include false races\n", *g_host.n_sync_dropped);
  fprintf(stderr, "syncsan: %llu instrumented launches, %zu distinct racing instruction pairs (%u race records%s), %u accesses untracked (shadow full)\n",
          (unsigned long long)g_launches, g_pairs.size(), n, n > MAX_RECORDS ? ", some not stored" : "", *g_host.n_dropped);
  if (g_exitcode && n) { fflush(stderr); _exit(g_exitcode); }
}
