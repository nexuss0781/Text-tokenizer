#pragma once
#include <cstdint>
#include <cstddef>
#include <cstring>
#include <vector>
#include <string>
#include <string_view>
#include <array>
#include <algorithm>
#include <numeric>
#include <bit>
#include <cassert>
#include <queue>
#include <stack>
#include <optional>
#include <memory>
#include <functional>
#include <cmath>
#include <random>
#include <map>
#include <span>
#include <fstream>
#include <sstream>
#include <iostream>
#include <thread>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <chrono>
#include <span>

namespace tokenizer {

using u8  = uint8_t;
using u16 = uint16_t;
using u32 = uint32_t;
using u64 = uint64_t;
using TokenID = u32;

constexpr u32 kMaxTokenLength = 256;
constexpr u32 kCacheLineSize  = 64;
constexpr u32 kPageSize       = 4096;
constexpr u32 kArenaSlabSize  = 4 * 1024 * 1024;
constexpr u32 kMaxVocabSize   = 128 * 1024;
constexpr u32 kBloomSizeBits  = 1 << 20;
constexpr u32 kNumHashFuncs   = 3;
constexpr u32 kSpecialTokenCount = 4;

// ============================================================
// BitVector — rank/select support
// ============================================================
class BitVector {
    std::vector<u64> blocks_;
    u64 size_ = 0;
    std::vector<u32> rank_lut_;
public:
    BitVector() = default;
    void resize(u64 bits);
    void set(u64 pos);
    bool test(u64 pos) const;
    u64 size() const { return size_; }
    u64 rank1(u64 pos) const;
    u64 select1(u64 k) const;
    u64 rank0(u64 pos) const;
    u64 select0(u64 k) const;
};

// ============================================================
// LOUDS Trie (§1.3) — token string → TokenID
// ============================================================
class LoudsTrie {
    BitVector L_;
    std::vector<u8> labels_;
    std::vector<TokenID> node_id_;
    std::vector<bool> is_leaf_;
    std::vector<u32> node_first_child_;
    std::vector<u32> node_num_children_;
    std::vector<u32> edge_to_node_;
    u32 num_nodes_ = 0;
public:
    void build(const std::vector<std::pair<std::string, TokenID>>& vocab);
    std::optional<TokenID> lookup(std::string_view s) const;

    // Longest prefix match for greedy BPE encoding.
    // Returns (token_id, match_length_bytes) for the longest vocabulary token
    // starting at bytes[0..len). Guaranteed to match at least 1 byte
    // (single bytes 0x00-0xFF are always in the vocabulary).
    struct LongestMatch { TokenID id; u32 length; };
    LongestMatch longest_prefix(const u8* bytes, u32 len) const;

    u32 num_nodes() const { return num_nodes_; }
    std::vector<std::pair<std::string, TokenID>> enumerate() const;
    u64 byte_size() const;
};

// ============================================================
// CHD Minimal Perfect Hash + Decoder (§1.4) — ID → string
// ============================================================
class CHDDecoder {
    std::vector<u8> concat_buf_;
    std::vector<u32> offsets_;
    u32 size_ = 0;
public:
    void build(const std::vector<std::pair<std::string, TokenID>>& vocab);
    std::string_view decode(TokenID id) const;
    u32 size() const { return size_; }
    u64 byte_size() const;
};

// ============================================================
// Arena Allocator (§2.3)
// ============================================================
class Arena {
    struct Slab { std::unique_ptr<u8[]> data; u32 capacity; u32 used; Slab* next; };
    Slab* head_ = nullptr;
    Slab* tail_ = nullptr;
    u32 slab_size_ = kArenaSlabSize;
    std::mutex mtx_;
public:
    Arena(u32 slab_size = kArenaSlabSize);
    ~Arena();
    u8* alloc(u32 size);
    void reset();
    u32 total_capacity() const;
    u32 total_used() const;
};

// ============================================================
// FST Engine (§1.2/3.2) — deterministic finite-state transducer
// ============================================================
struct FSTTransition {
    u16 next_state;
    u16 flags;  // bit 0: has_output, bits 1-15: output_token_id
};

class FST {
    std::vector<std::array<FSTTransition, 256>> transitions_;
    u32 num_states_ = 0;
public:
    u32 add_state();
    void set_transition(u32 state, u8 input, u16 next_state, bool has_output = false, TokenID output_id = 0);
    FSTTransition get_transition(u32 state, u8 input) const;
    u32 num_states() const { return num_states_; }
    bool is_deterministic() const;
    void minimize();
    u64 byte_size() const;
};

// ============================================================
// Merge Rule — used in training and compilation
// ============================================================
struct MergeRule {
    std::string left;
    std::string right;
    std::string merged;
    double score;
};

// ============================================================
// FST Compiler (§3.2) — merge table → FST
// ============================================================
class FSTCompiler {
public:
    FST compile(const std::vector<MergeRule>& rules, const std::vector<std::string>& vocab) const;
};

// ============================================================
// Suffix Array + LCP (for training)
// ============================================================
class SuffixArray {
    std::vector<u32> sa_;
    std::vector<u32> lcp_;
    std::string_view data_;
public:
    void build(std::string_view s);
    const std::vector<u32>& sa() const { return sa_; }
    const std::vector<u32>& lcp() const { return lcp_; }
    u32 size() const { return (u32)sa_.size(); }
};

// ============================================================
// Fenwick Tree (Binary Indexed Tree)
// ============================================================
class Fenwick {
    std::vector<u64> bit_;
public:
    Fenwick(u32 n) : bit_(n + 1, 0) {}
    void add(u32 idx, u64 delta);
    u64 sum(u32 idx) const;
    u64 range_sum(u32 l, u32 r) const;
};

// ============================================================
// Entropy-Weighted BPE Training (§1.1/3.1)
// ============================================================
class EWBPETrainer {
public:
    struct Result {
        std::vector<MergeRule> rules;
        std::vector<std::string> vocab;
    };
    Result train(std::string_view corpus, u32 target_vocab_size) const;
};

// ============================================================
// Bloom Filter (§3.4) — for parallel boundary detection
// ============================================================
class BloomFilter {
    std::vector<u64> bits_;
    u32 num_hashes_;
public:
    BloomFilter(u32 num_bits = kBloomSizeBits, u32 num_hashes = kNumHashFuncs);
    void add(std::string_view s);
    bool contains(std::string_view s) const;
    void clear();
    // Build from vocabulary for boundary detection
    void build_from_vocab(const std::vector<std::string>& vocab);
};

// ============================================================
// UTF-8 Validation (§2.1 Layer 0)
// ============================================================
class UTF8Validator {
public:
    static bool validate(std::string_view s);
    static std::vector<u8> encode_to_bytes(std::string_view s);
    static std::string decode_from_bytes(const std::vector<u8>& bytes);
    static bool is_valid_continuation(u8 c) { return (c & 0xC0) == 0x80; }
};

// ============================================================
// Special Token Injector (§2.1 Layer 3)
// ============================================================
struct SpecialTokenConfig {
    TokenID bos_id = 0;
    TokenID eos_id = 0;
    TokenID pad_id = 0;
    TokenID unk_id = 0;
    bool add_bos = false;
    bool add_eos = false;
};

class SpecialTokenInjector {
    SpecialTokenConfig config_;
    std::vector<std::string> special_strings_;
public:
    SpecialTokenInjector() = default;
    void configure(const SpecialTokenConfig& cfg, const std::vector<std::string>& strings);
    std::vector<TokenID> inject(const std::vector<TokenID>& ids) const;
    SpecialTokenConfig config() const { return config_; }
};

// ============================================================
// Parallel Boundary Detector (§1.5/3.4)
// ============================================================
class BoundaryDetector {
    BloomFilter bloom_;
    u32 max_token_len_ = kMaxTokenLength;
    std::vector<u64> vocab_hashes_;
    u64 vocab_hash_mask_;
public:
    BoundaryDetector(u32 max_token_len = kMaxTokenLength);
    void build(const std::vector<std::string>& vocab);
    std::vector<u32> find_boundaries(std::string_view input) const;
};

// ============================================================
// Tokenizer — Top-Level API
// ============================================================
class Tokenizer {
    std::unique_ptr<LoudsTrie> trie_;
    std::unique_ptr<CHDDecoder> decoder_;
    std::unique_ptr<FST> fst_;
    std::unique_ptr<SpecialTokenInjector> injector_;
    std::unique_ptr<BoundaryDetector> boundary_detector_;
    std::vector<std::string> vocab_;
    std::vector<std::string> special_tokens_;
    SpecialTokenConfig special_config_;
    bool parallel_enabled_ = false;
    u32 num_threads_ = 1;

public:
    Tokenizer() = default;

    // Direct vocab load (fast, no training)
    void set_vocab(const std::vector<std::string>& vocab);

    // Training
    void train(std::string_view corpus, u32 target_vocab_size);

    // FST compilation
    void compile_fst();

    // Load/save
    bool load(const std::string& path);
    bool save(const std::string& path) const;

    // Encode (single-thread, streaming)
    std::vector<TokenID> encode(std::string_view s) const;
    // Streaming encode — processes a chunk, returns whether more input is expected
    struct StreamState {
        std::vector<u32> state_stack;
        std::vector<u8> partial;
        bool has_partial = false;
    };
    std::vector<TokenID> encode_streaming(std::string_view s, StreamState& state) const;

    // Encode (parallel)
    std::vector<TokenID> encode_parallel(std::string_view s, u32 num_threads = 0) const;

    // Decode
    std::string decode(const std::vector<TokenID>& ids) const;

    // Configuration
    void set_special_tokens(const SpecialTokenConfig& cfg, const std::vector<std::string>& special_strings);
    void enable_parallel(u32 num_threads);
    void disable_parallel();

    // Accessors
    const LoudsTrie& trie() const { return *trie_; }
    const CHDDecoder& decoder() const { return *decoder_; }
    const FST& fst() const { return *fst_; }
    const std::vector<std::string>& vocab() const { return vocab_; }
    u32 vocab_size() const { return (u32)vocab_.size(); }
    u32 num_threads() const { return num_threads_; }

    // RAM usage
    u64 total_ram_bytes() const;
};

// ============================================================
// Confidence-Annotated Token (§6.2)
// ============================================================
struct AnnotatedToken {
    TokenID id;
    float confidence;
};

// ============================================================
// Differential Re-tokenization (§6.3)
// ============================================================
struct DiffEdit {
    u32 offset;
    u32 delete_len;
    std::string insert;
};

std::vector<TokenID> differential_retokenize(
    const Tokenizer& tok,
    std::string_view original,
    std::string_view modified,
    const DiffEdit& edit,
    u32 context_bytes = 64);

// ============================================================
// Domain-Adaptive Vocabulary Slicing (§6.4)
// ============================================================
enum class DomainType { General, Code, Multilingual };

LoudsTrie build_domain_trie(const Tokenizer& tok, DomainType domain);

} // namespace tokenizer
