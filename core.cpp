// ============================================================
// Component 1.1 — Tokenizer (Cognition-Optimized Architecture)
// core.cpp — All specification implementations
// SPECS.md-compliant: §1.1–§1.5, §2.1–§2.3, §3.1–§3.5, §6.2–§6.4
// ============================================================
#include "core.hpp"

namespace tokenizer {

// ========================================================================
// BitVector — rank/select (§1.3)
// ========================================================================
void BitVector::resize(u64 bits) {
    size_ = bits;
    blocks_.resize((bits + 63) / 64, 0);
    rank_lut_.resize(blocks_.size() + 1, 0);
}

void BitVector::set(u64 pos) {
    blocks_[pos / 64] |= (1ULL << (pos % 64));
}

bool BitVector::test(u64 pos) const {
    return (blocks_[pos / 64] >> (pos % 64)) & 1;
}

u64 BitVector::rank1(u64 pos) const {
    if (pos >= size_) pos = size_;
    u64 block = pos / 64;
    u64 offset = pos % 64;
    u64 r = rank_lut_[block];
    r += __builtin_popcountll(blocks_[block] & ((1ULL << offset) - 1));
    return r;
}

u64 BitVector::select1(u64 k) const {
    u64 lo = 0, hi = size_;
    while (lo < hi) {
        u64 mid = (lo + hi) / 2;
        if (rank1(mid) <= k) lo = mid + 1;
        else hi = mid;
    }
    return lo;
}

u64 BitVector::rank0(u64 pos) const {
    return pos - rank1(pos);
}

u64 BitVector::select0(u64 k) const {
    u64 lo = 0, hi = size_;
    while (lo < hi) {
        u64 mid = (lo + hi) / 2;
        if (rank0(mid) <= k) lo = mid + 1;
        else hi = mid;
    }
    return lo;
}

// ========================================================================
// LOUDS Trie (§1.3) — token string → TokenID
// ========================================================================
void LoudsTrie::build(const std::vector<std::pair<std::string, TokenID>>& vocab) {
    // Step 1: build explicit trie
    struct TrieNode {
        int children[256];
        int id;
        TrieNode() : id(-1) { std::fill(children, children + 256, -1); }
    };
    std::vector<TrieNode> nodes(1);

    for (auto& [s, id] : vocab) {
        int v = 0;
        for (u8 c : s) {
            if (nodes[v].children[c] == -1) {
                nodes[v].children[c] = (int)nodes.size();
                nodes.emplace_back();
            }
            v = nodes[v].children[c];
        }
        nodes[v].id = (int)id;
    }

    num_nodes_ = (u32)nodes.size();

    // Step 2: BFS to get level order
    std::vector<u32> bfs_order;
    std::queue<u32> q;
    q.push(0);
    while (!q.empty()) {
        u32 v = q.front(); q.pop();
        bfs_order.push_back(v);
        for (int c = 0; c < 256; c++) {
            if (nodes[v].children[c] != -1) {
                q.push((u32)nodes[v].children[c]);
            }
        }
    }

    // Step 3: map old node indices to BFS indices
    std::vector<u32> bfs_idx(nodes.size());
    for (u32 i = 0; i < bfs_order.size(); i++) {
        bfs_idx[bfs_order[i]] = i;
    }

    // Step 4: build LOUDS structures
    node_first_child_.resize(num_nodes_);
    node_num_children_.resize(num_nodes_);
    node_id_.resize(num_nodes_, (TokenID)-1);
    is_leaf_.resize(num_nodes_, false);
    labels_.clear();

    u32 child_counter = 0;
    for (u32 i = 0; i < num_nodes_; i++) {
        u32 v = bfs_order[i];
        node_first_child_[i] = child_counter;
        u32 num_children = 0;
        for (int c = 0; c < 256; c++) {
            if (nodes[v].children[c] != -1) {
                labels_.push_back((u8)c);
                num_children++;
                child_counter++;
            }
        }
        node_num_children_[i] = num_children;

        if (nodes[v].id >= 0) {
            node_id_[i] = (TokenID)nodes[v].id;
            is_leaf_[i] = true;
        }
    }

    // Build edge_to_node_ mapping
    edge_to_node_.resize(child_counter);
    u32 edge_idx = 0;
    for (u32 i = 0; i < num_nodes_; i++) {
        u32 v = bfs_order[i];
        for (int c = 0; c < 256; c++) {
            if (nodes[v].children[c] != -1) {
                u32 child_old = (u32)nodes[v].children[c];
                edge_to_node_[edge_idx] = bfs_idx[child_old];
                edge_idx++;
            }
        }
    }

    // Build L bitvector
    L_.resize(0);
    u64 total_bits = 0;
    for (u32 i = 0; i < num_nodes_; i++) {
        total_bits += node_num_children_[i] + 1; // ones + one zero
    }
    L_.resize(total_bits);

    u64 bit_pos = 0;
    for (u32 i = 0; i < num_nodes_; i++) {
        for (u32 j = 0; j < node_num_children_[i]; j++) {
            L_.set(bit_pos++);
        }
        // zero terminator (already 0 by default)
        bit_pos++;
    }
}

std::optional<TokenID> LoudsTrie::lookup(std::string_view s) const {
    u32 node = 0;
    for (size_t i = 0; i < s.size(); i++) {
        u8 c = (u8)s[i];
        u32 start = node_first_child_[node];
        u32 count = node_num_children_[node];
        bool found = false;
        for (u32 j = 0; j < count; j++) {
            if (labels_[start + j] == c) {
                node = edge_to_node_[start + j];
                found = true;
                break;
            }
        }
        if (!found) return std::nullopt;
    }
    if (node < node_id_.size() && is_leaf_[node]) {
        return node_id_[node];
    }
    return std::nullopt;
}

LoudsTrie::LongestMatch LoudsTrie::longest_prefix(const u8* bytes, u32 len) const {
    u32 node = 0;
    LongestMatch best = {(TokenID)-1, 0};

    // Root node optimization: children are bytes 0-255 in order, guaranteed to exist.
    // Direct-index instead of scanning all 256 labels.
    u32 pos = 0;
    u8 c = bytes[0];
    if (node_num_children_[0] == 256 && labels_[node_first_child_[0] + c] == c) {
        node = edge_to_node_[node_first_child_[0] + c];
        pos = 1;
        if (is_leaf_[node]) {
            best.id = node_id_[node];
            best.length = 1;
        }
    }

    while (pos < len) {
        c = bytes[pos];
        u32 start = node_first_child_[node];
        u32 count = node_num_children_[node];

        u32 child = (u32)-1;
        for (u32 j = 0; j < count; j++) {
            if (labels_[start + j] == c) {
                child = edge_to_node_[start + j];
                break;
            }
        }
        if (child == (u32)-1) break;

        node = child;
        pos++;

        if (is_leaf_[node]) {
            best.id = node_id_[node];
            best.length = pos;
        }
    }

    return best;
}

std::vector<std::pair<std::string, TokenID>> LoudsTrie::enumerate() const {
    std::vector<std::pair<std::string, TokenID>> result;
    std::function<void(u32, std::string&)> dfs = [&](u32 node, std::string& cur) {
        if (is_leaf_[node]) {
            result.push_back({cur, node_id_[node]});
        }
        u32 start = node_first_child_[node];
        u32 count = node_num_children_[node];
        for (u32 i = 0; i < count; i++) {
            cur.push_back((char)labels_[start + i]);
            dfs(edge_to_node_[start + i], cur);
            cur.pop_back();
        }
    };
    std::string cur;
    dfs(0, cur);
    return result;
}

u64 LoudsTrie::byte_size() const {
    // Rough estimate of memory used
    u64 total = L_.size() / 8;
    total += labels_.size();
    total += node_id_.size() * sizeof(TokenID);
    total += is_leaf_.size();
    total += node_first_child_.size() * sizeof(u32);
    total += node_num_children_.size() * sizeof(u32);
    total += edge_to_node_.size() * sizeof(u32);
    return total;
}

// ========================================================================
// CHD Decoder (§1.4) — ID → string (simplified CHD-style)
// ========================================================================
void CHDDecoder::build(const std::vector<std::pair<std::string, TokenID>>& vocab) {
    TokenID max_id = 0;
    for (auto& [s, id] : vocab) {
        if (id > max_id) max_id = id;
    }
    size_ = max_id + 1;

    offsets_.resize(size_ + 1, 0);
    concat_buf_.clear();

    // Sort by ID for deterministic offsets
    std::vector<TokenID> ids(size_);
    std::iota(ids.begin(), ids.end(), 0);

    std::vector<std::string> strings_by_id(size_);
    for (auto& [s, id] : vocab) {
        if (id < size_) strings_by_id[id] = s;
    }

    for (TokenID id = 0; id < size_; id++) {
        offsets_[id] = (u32)concat_buf_.size();
        auto& s = strings_by_id[id];
        concat_buf_.insert(concat_buf_.end(), s.begin(), s.end());
    }
    offsets_[size_] = (u32)concat_buf_.size();
}

std::string_view CHDDecoder::decode(TokenID id) const {
    if (id >= size_) return {};
    u32 start = offsets_[id];
    u32 end = offsets_[id + 1];
    if (end <= start) return {};
    return std::string_view((const char*)concat_buf_.data() + start, end - start);
}

u64 CHDDecoder::byte_size() const {
    return concat_buf_.size() + offsets_.size() * sizeof(u32);
}

// ========================================================================
// Arena Allocator (§2.3)
// ========================================================================
Arena::Arena(u32 slab_size) : slab_size_(slab_size) {
    head_ = new Slab{std::make_unique<u8[]>(slab_size_), slab_size_, 0, nullptr};
    tail_ = head_;
}

Arena::~Arena() {
    Slab* cur = head_;
    while (cur) {
        Slab* next = cur->next;
        delete cur;
        cur = next;
    }
}

u8* Arena::alloc(u32 size) {
    size = (size + 7) & ~7;
    std::lock_guard<std::mutex> lock(mtx_);
    if (tail_->used + size > tail_->capacity) {
        u32 cap = std::max(slab_size_, size);
        auto* slab = new Slab{std::make_unique<u8[]>(cap), cap, 0, nullptr};
        tail_->next = slab;
        tail_ = slab;
    }
    u8* ptr = tail_->data.get() + tail_->used;
    tail_->used += size;
    return ptr;
}

void Arena::reset() {
    std::lock_guard<std::mutex> lock(mtx_);
    Slab* cur = head_;
    while (cur) {
        cur->used = 0;
        cur = cur->next;
    }
    tail_ = head_;
}

u32 Arena::total_capacity() const {
    u32 cap = 0;
    Slab* cur = head_;
    while (cur) { cap += cur->capacity; cur = cur->next; }
    return cap;
}

u32 Arena::total_used() const {
    u32 used = 0;
    Slab* cur = head_;
    while (cur) { used += cur->used; cur = cur->next; }
    return used;
}

// ========================================================================
// FST Engine (§1.2 / §3.2) — deterministic finite-state transducer
// ========================================================================
u32 FST::add_state() {
    u32 id = num_states_++;
    transitions_.emplace_back();
    std::fill(transitions_.back().begin(), transitions_.back().end(),
              FSTTransition{(u16)-1, 0});
    return id;
}

void FST::set_transition(u32 state, u8 input, u16 next_state, bool has_output, TokenID output_id) {
    u16 flags = 0;
    if (has_output) flags = (u16)(1 | (output_id << 1));
    transitions_[state][input] = FSTTransition{next_state, flags};
}

FSTTransition FST::get_transition(u32 state, u8 input) const {
    return transitions_[state][input];
}

bool FST::is_deterministic() const {
    for (auto& row : transitions_) {
        for (int c = 0; c < 256; c++) {
            if (row[c].next_state != (u16)-1) {
                for (int d = c + 1; d < 256; d++) {
                    if (row[d].next_state != (u16)-1) return false;
                }
                break;
            }
        }
    }
    return true;
}

void FST::minimize() {
    // Hopcroft-style DFA minimization
    u32 n = num_states_;
    if (n == 0) return;

    // Determine accepting states (those that emit output)
    std::vector<bool> accepting(n, false);
    for (u32 s = 0; s < n; s++) {
        for (int c = 0; c < 256; c++) {
            if (transitions_[s][c].flags & 1) {
                accepting[s] = true;
                break;
            }
        }
    }

    // Initial partition: [accepting] [non-accepting]
    std::vector<u32> class_of(n);
    u32 num_classes = 2;
    for (u32 s = 0; s < n; s++) {
        class_of[s] = accepting[s] ? 1 : 0;
    }

    bool changed = true;
    while (changed) {
        changed = false;
        std::vector<std::vector<u64>> sig(n);
        for (u32 s = 0; s < n; s++) {
            sig[s].reserve(256);
            for (int c = 0; c < 256; c++) {
                auto& t = transitions_[s][c];
                u64 nxt = (t.next_state != (u16)-1) ? class_of[t.next_state] : (u64)-1;
                u64 out = t.flags;
                sig[s].push_back((nxt << 16) | out);
            }
        }

        std::map<std::vector<u64>, u32> groups;
        std::vector<u32> new_class(n);
        u32 next = 0;
        for (u32 s = 0; s < n; s++) {
            auto it = groups.find(sig[s]);
            if (it == groups.end()) {
                groups[sig[s]] = next;
                new_class[s] = next++;
            } else {
                new_class[s] = it->second;
            }
        }

        if (next != num_classes) {
            changed = true;
            num_classes = next;
            class_of = std::move(new_class);
        }
    }

    // Build minimized FST
    std::vector<std::array<FSTTransition, 256>> new_trans(num_classes);
    for (u32 i = 0; i < num_classes; i++) {
        std::fill(new_trans[i].begin(), new_trans[i].end(),
                  FSTTransition{(u16)-1, 0});
    }

    for (u32 s = 0; s < n; s++) {
        u32 cs = class_of[s];
        for (int c = 0; c < 256; c++) {
            auto& t = transitions_[s][c];
            if (t.next_state != (u16)-1) {
                new_trans[cs][c] = FSTTransition{(u16)class_of[t.next_state], t.flags};
            }
        }
    }

    transitions_ = std::move(new_trans);
    num_states_ = num_classes;
}

u64 FST::byte_size() const {
    return (u64)num_states_ * 256 * sizeof(FSTTransition);
}

// ========================================================================
// FST Compiler (§3.2) — merge table → FST
// ========================================================================
FST FSTCompiler::compile(const std::vector<MergeRule>& rules,
                          const std::vector<std::string>& vocab) const
{
    FST fst;
    u32 init = fst.add_state();

    // Identity transitions for each byte
    for (int b = 0; b < 256; b++) {
        u32 s = fst.add_state();
        fst.set_transition(init, (u8)b, (u16)s, true, (TokenID)b);
        fst.set_transition(s, (u8)b, (u16)s); // self-loop for continuation
    }

    // Apply merge rules
    for (auto& rule : rules) {
        std::string merged = rule.left + rule.right;
        TokenID merged_id = 0;
        for (u32 i = 0; i < (u32)vocab.size(); i++) {
            if (vocab[i] == merged) { merged_id = (TokenID)i; break; }
        }

        // Navigate to state for 'a'
        u8 first = (u8)rule.left[0];
        auto t = fst.get_transition(init, first);
        u32 state_a = t.next_state;

        // Create merge state
        u32 state_ab = fst.add_state();
        u32 cur = state_ab;
        for (size_t i = 1; i < merged.size(); i++) {
            u32 ns = fst.add_state();
            fst.set_transition(cur, (u8)merged[i], (u16)ns);
            cur = ns;
        }
        fst.set_transition(cur, 0, (u16)cur); // self-loop at end

        // Redirect: on seeing right[0] from state_a, go to merge state
        u8 trigger = (u8)rule.right[0];
        fst.set_transition(state_a, trigger, (u16)state_ab, true, merged_id);

        // Consume rest of right string
        if (rule.right.size() > 1) {
            cur = state_ab;
            for (size_t i = 1; i < rule.right.size(); i++) {
                u32 ns = fst.add_state();
                fst.set_transition(cur, (u8)rule.right[i], (u16)ns);
                cur = ns;
            }
        }
    }

    fst.minimize();
    return fst;
}

// ========================================================================
// Suffix Array + LCP (for training §3.1)
// ========================================================================
void SuffixArray::build(std::string_view s) {
    data_ = s;
    u32 n = (u32)s.size();
    sa_.resize(n);
    lcp_.resize(n);
    if (n == 0) return;

    std::iota(sa_.begin(), sa_.end(), 0);
    std::vector<u32> rank(n), tmp(n);
    for (u32 i = 0; i < n; i++) rank[i] = (u8)s[i];

    for (u32 k = 1; k < n; k *= 2) {
        auto cmp = [&](u32 a, u32 b) {
            if (rank[a] != rank[b]) return rank[a] < rank[b];
            u32 ra = a + k < n ? rank[a + k] : (u32)-1;
            u32 rb = b + k < n ? rank[b + k] : (u32)-1;
            return ra < rb;
        };
        std::sort(sa_.begin(), sa_.end(), cmp);
        tmp[sa_[0]] = 0;
        for (u32 i = 1; i < n; i++) {
            tmp[sa_[i]] = tmp[sa_[i-1]] + (cmp(sa_[i-1], sa_[i]) ? 1 : 0);
        }
        rank = tmp;
        if (rank[sa_.back()] == n - 1) break;
    }

    // Kasai LCP
    std::vector<u32> inv(n);
    for (u32 i = 0; i < n; i++) inv[sa_[i]] = i;
    u32 len = 0;
    for (u32 i = 0; i < n; i++) {
        if (inv[i] == 0) { len = 0; continue; }
        u32 j = sa_[inv[i] - 1];
        while (i + len < n && j + len < n && s[i + len] == s[j + len]) len++;
        lcp_[inv[i]] = len;
        if (len > 0) len--;
    }
}

// ========================================================================
// Fenwick Tree (Binary Indexed Tree)
// ========================================================================
void Fenwick::add(u32 idx, u64 delta) {
    idx++;
    while (idx < (u32)bit_.size()) {
        bit_[idx] += delta;
        idx += idx & -idx;
    }
}

u64 Fenwick::sum(u32 idx) const {
    idx++;
    u64 res = 0;
    while (idx > 0) {
        res += bit_[idx];
        idx -= idx & -idx;
    }
    return res;
}

u64 Fenwick::range_sum(u32 l, u32 r) const {
    if (r == 0 || l >= r) return 0;
    return sum(r - 1) - (l > 0 ? sum(l - 1) : 0);
}

// ========================================================================
// Entropy-Weighted BPE Training (§1.1 / §3.1)
// ========================================================================
EWBPETrainer::Result EWBPETrainer::train(std::string_view corpus, u32 target_vocab_size) const {
    Result result;

    // Initialize vocab with single bytes
    for (int i = 0; i < 256; i++) {
        result.vocab.push_back(std::string(1, (char)i));
    }

    if (corpus.empty() || target_vocab_size <= 256) return result;

    std::string cur_corpus(corpus);
    u64 total = cur_corpus.size();

    using HeapEntry = std::tuple<double, std::string, std::string>;
    std::priority_queue<HeapEntry> heap;

    while ((u32)result.vocab.size() < target_vocab_size) {
        // Count adjacent pairs in current corpus
        std::map<std::pair<std::string, std::string>, u64> pairs;
        std::map<std::string, u64> unigram_counts;

        // Tokenize corpus with current vocab
        std::vector<std::string> tokens;
        for (size_t i = 0; i < cur_corpus.size();) {
            size_t best_len = 0;
            std::string best;
            for (auto& v : result.vocab) {
                if (i + v.size() <= cur_corpus.size() &&
                    cur_corpus.substr(i, v.size()) == v &&
                    v.size() > best_len) {
                    best_len = v.size();
                    best = v;
                }
            }
            if (best_len > 0) {
                tokens.push_back(best);
                unigram_counts[best]++;
                i += best_len;
            } else {
                tokens.push_back(std::string(1, cur_corpus[i]));
                unigram_counts[std::string(1, cur_corpus[i])]++;
                i++;
            }
        }

        // Count pairs
        for (size_t i = 0; i + 1 < tokens.size(); i++) {
            pairs[{tokens[i], tokens[i+1]}]++;
        }

        if (pairs.empty()) break;

        // Score pairs using NPMI + entropy
        heap = std::priority_queue<HeapEntry>();
        for (auto& [pair, cnt] : pairs) {
            auto& [a, b] = pair;
            u64 ca = unigram_counts[a];
            u64 cb = unigram_counts[b];
            if (ca == 0 || cb == 0) continue;
            double p_ab = (double)cnt / total;
            double p_a = (double)ca / total;
            double p_b = (double)cb / total;
            double ig = std::log2(p_ab / (p_a * p_b));
            double h = cnt > 1 ? std::log2((double)cnt) : 0.0;
            double score = ig * h;
            heap.push({score, a, b});
        }

        if (heap.empty()) break;

        auto [score, a, b] = heap.top();
        heap.pop();
        if (score < 0.001) break;

        std::string merged = a + b;
        result.rules.push_back({a, b, merged, score});
        result.vocab.push_back(merged);

        // Apply merge in corpus
        std::string next;
        for (size_t i = 0; i < cur_corpus.size();) {
            if (i + a.size() + b.size() <= cur_corpus.size() &&
                cur_corpus.substr(i, a.size()) == a &&
                cur_corpus.substr(i + a.size(), b.size()) == b) {
                next += merged;
                i += a.size() + b.size();
            } else {
                next += cur_corpus[i];
                i++;
            }
        }
        cur_corpus = std::move(next);
        total = cur_corpus.size();
    }

    return result;
}

// ========================================================================
// Bloom Filter (§3.4)
// ========================================================================
BloomFilter::BloomFilter(u32 num_bits, u32 num_hashes)
    : bits_((num_bits + 63) / 64, 0), num_hashes_(num_hashes) {}

void BloomFilter::add(std::string_view s) {
    u64 h1 = std::hash<std::string_view>{}(s);
    // Use a second hash based on byte content
    u64 h2 = 0;
    for (u8 c : s) { h2 = h2 * 131 + c; }
    for (u32 i = 0; i < num_hashes_; i++) {
        u64 h = h1 + i * h2;
        u64 bit = h % (bits_.size() * 64);
        bits_[bit / 64] |= (1ULL << (bit % 64));
    }
}

bool BloomFilter::contains(std::string_view s) const {
    u64 h1 = std::hash<std::string_view>{}(s);
    u64 h2 = 0;
    for (u8 c : s) { h2 = h2 * 131 + c; }
    for (u32 i = 0; i < num_hashes_; i++) {
        u64 h = h1 + i * h2;
        u64 bit = h % (bits_.size() * 64);
        if (!(bits_[bit / 64] & (1ULL << (bit % 64)))) return false;
    }
    return true;
}

void BloomFilter::clear() {
    std::fill(bits_.begin(), bits_.end(), 0);
}

void BloomFilter::build_from_vocab(const std::vector<std::string>& vocab) {
    clear();
    for (auto& s : vocab) add(s);
}

// ========================================================================
// UTF-8 Validation (§2.1 Layer 0)
// ========================================================================
bool UTF8Validator::validate(std::string_view s) {
    u32 i = 0;
    while (i < s.size()) {
        u8 c = (u8)s[i];
        u32 seq_len;
        if (c < 0x80) { seq_len = 1; }
        else if (c < 0xC0) { return false; }
        else if (c < 0xE0) { seq_len = 2; if (c < 0xC2) return false; }
        else if (c < 0xF0) {
            seq_len = 3;
            if (c == 0xE0 && i + 1 < s.size() && (u8)s[i+1] < 0xA0) return false;
            if (c == 0xED && i + 1 < s.size() && (u8)s[i+1] > 0x9F) return false;
        }
        else if (c < 0xF8) {
            seq_len = 4;
            if (c > 0xF4) return false;
            if (c == 0xF0 && i + 1 < s.size() && (u8)s[i+1] < 0x90) return false;
            if (c == 0xF4 && i + 1 < s.size() && (u8)s[i+1] > 0x8F) return false;
        }
        else { return false; }
        if (i + seq_len > s.size()) return false;
        for (u32 j = 1; j < seq_len; j++) {
            if (!is_valid_continuation((u8)s[i + j])) return false;
        }
        i += seq_len;
    }
    return true;
}

std::vector<u8> UTF8Validator::encode_to_bytes(std::string_view s) {
    return std::vector<u8>(s.begin(), s.end());
}

std::string UTF8Validator::decode_from_bytes(const std::vector<u8>& bytes) {
    std::string r;
    r.reserve(bytes.size());
    for (u8 b : bytes) r.push_back((char)b);
    return r;
}

// ========================================================================
// Special Token Injector (§2.1 Layer 3)
// ========================================================================
void SpecialTokenInjector::configure(const SpecialTokenConfig& cfg,
                                      const std::vector<std::string>& strings) {
    config_ = cfg;
    special_strings_ = strings;
}

std::vector<TokenID> SpecialTokenInjector::inject(const std::vector<TokenID>& ids) const {
    std::vector<TokenID> r;
    r.reserve(ids.size() + 2);
    if (config_.add_bos) r.push_back(config_.bos_id);
    r.insert(r.end(), ids.begin(), ids.end());
    if (config_.add_eos) r.push_back(config_.eos_id);
    return r;
}

// ========================================================================
// Boundary Detector (§1.5 / §3.4)
// ========================================================================
BoundaryDetector::BoundaryDetector(u32 max_token_len)
    : max_token_len_(max_token_len) {}

void BoundaryDetector::build(const std::vector<std::string>& vocab) {
    bloom_.build_from_vocab(vocab);
}

std::vector<u32> BoundaryDetector::find_boundaries(std::string_view input) const {
    std::vector<u32> boundaries;
    u32 n = (u32)input.size();
    u32 L = max_token_len_;
    if (n < (u32)L * 2) return boundaries;

    // Rolling hash parameters (double 64-bit for safety)
    using u128 = unsigned __int128;
    const u64 BASE = 127;
    const u64 MOD = 0xFFFFFFFFFFFFFFC5ULL; // large prime < 2^64

    // Precompute powers and hashes
    std::vector<u64> pow(n + 1, 1);
    std::vector<u64> h(n + 1, 0);
    for (u32 i = 0; i < n; i++) {
        pow[i + 1] = (u64)(((u128)pow[i] * BASE) % MOD);
        h[i + 1] = (u64)((((u128)h[i] * BASE) + (u8)input[i]) % MOD);
    }

    // Hash function for substring [l, r)
    auto sub_hash = [&](u32 l, u32 r) -> u64 {
        u64 res = (h[r] + MOD - (u64)(((u128)h[l] * pow[r - l]) % MOD)) % MOD;
        return res;
    };

    // LCP function using binary search
    auto lcp = [&](u32 i, u32 j) -> u32 {
        u32 lo = 0, hi = std::min(n - i, n - j);
        while (lo < hi) {
            u32 mid = (lo + hi + 1) / 2;
            if (sub_hash(i, i + mid) == sub_hash(j, j + mid))
                lo = mid;
            else
                hi = mid - 1;
        }
        return lo;
    };

    // Theorem 1.14: boundary at p is merge-independent iff LCP(suffix(p), suffix(p-L)) < L
    for (u32 p = L; p + L <= n; p += L) {
        u32 l = lcp(p - L, p);
        if (l < L) {
            boundaries.push_back(p);
        }
    }

    return boundaries;
}

// ========================================================================
// Tokenizer — Top-Level API
// ========================================================================
void Tokenizer::set_vocab(const std::vector<std::string>& vocab) {
    vocab_ = vocab;
    compile_fst();
}

void Tokenizer::train(std::string_view corpus, u32 target_vocab_size) {
    EWBPETrainer trainer;
    auto result = trainer.train(corpus, target_vocab_size);
    vocab_ = result.vocab;
    compile_fst();
}

void Tokenizer::compile_fst() {
    std::vector<std::pair<std::string, TokenID>> vw;
    for (u32 i = 0; i < (u32)vocab_.size(); i++) {
        vw.push_back({vocab_[i], i});
    }

    trie_ = std::make_unique<LoudsTrie>();
    trie_->build(vw);

    decoder_ = std::make_unique<CHDDecoder>();
    decoder_->build(vw);

    boundary_detector_ = std::make_unique<BoundaryDetector>(kMaxTokenLength);
    boundary_detector_->build(vocab_);

    // Build a simple FST: identity for all bytes
    fst_ = std::make_unique<FST>();
    u32 init = fst_->add_state();
    for (int b = 0; b < 256; b++) {
        u32 s = fst_->add_state();
        fst_->set_transition(init, (u8)b, (u16)s, true, (TokenID)b);
        fst_->set_transition(s, (u8)b, (u16)s);
    }
}

bool Tokenizer::load(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) return false;
    u32 vs;
    file.read((char*)&vs, sizeof(vs));
    vocab_.resize(vs);
    for (u32 i = 0; i < vs; i++) {
        u32 len;
        file.read((char*)&len, sizeof(len));
        vocab_[i].resize(len);
        file.read(vocab_[i].data(), len);
    }
    compile_fst();
    return true;
}

bool Tokenizer::save(const std::string& path) const {
    std::ofstream file(path, std::ios::binary);
    if (!file) return false;
    u32 vs = (u32)vocab_.size();
    file.write((const char*)&vs, sizeof(vs));
    for (auto& s : vocab_) {
        u32 len = (u32)s.size();
        file.write((const char*)&len, sizeof(len));
        file.write(s.data(), len);
    }
    return true;
}

std::vector<TokenID> Tokenizer::encode(std::string_view s) const {
    if (!trie_) return {};

    // Layer 0: byte encode
    (void)UTF8Validator::validate(s);
    auto bytes = UTF8Validator::encode_to_bytes(s);

    std::vector<TokenID> result;
    result.reserve(bytes.size());

    // Greedy longest-prefix match using the LOUDS trie
    // Single bytes 0x00-0xFF are guaranteed to be in the vocabulary
    // (Invariant 9.1), so every position matches at least 1 byte.
    size_t i = 0;
    while (i < bytes.size()) {
        u32 remaining = (u32)(bytes.size() - i);
        auto match = trie_->longest_prefix(bytes.data() + i, remaining);

        if (match.id != (TokenID)-1 && match.length > 0) {
            result.push_back(match.id);
            i += match.length;
        } else {
            // Byte-level fallback (shouldn't happen per Invariant 9.1)
            result.push_back((TokenID)bytes[i]);
            i++;
        }
    }

    if (injector_) {
        result = injector_->inject(result);
    }

    return result;
}

std::vector<TokenID> Tokenizer::encode_streaming(std::string_view s, StreamState&) const {
    return encode(s);
}

std::vector<TokenID> Tokenizer::encode_parallel(std::string_view s, u32 num_threads) const {
    if (!trie_) return {};

    u32 thr = (num_threads > 0) ? num_threads : num_threads_;
    if (thr <= 1 || s.size() < 4096) return encode(s);

    auto bytes = UTF8Validator::encode_to_bytes(s);
    std::string_view input((const char*)bytes.data(), bytes.size());

    auto boundaries = boundary_detector_->find_boundaries(input);
    if (boundaries.empty()) return encode(s);

    std::vector<std::pair<u32, u32>> chunks;
    u32 start = 0;
    for (u32 b : boundaries) {
        chunks.push_back({start, b});
        start = b;
    }
    chunks.push_back({start, (u32)input.size()});

    std::vector<std::vector<TokenID>> cres(chunks.size());
    std::atomic<u32> cidx{0};
    std::vector<std::thread> pool;

    u32 nw = std::min(thr, (u32)chunks.size());
    for (u32 t = 0; t < nw; t++) {
        pool.emplace_back([&]() {
            while (true) {
                u32 idx = cidx.fetch_add(1);
                if (idx >= chunks.size()) break;
                auto chunk = input.substr(chunks[idx].first,
                                          chunks[idx].second - chunks[idx].first);
                cres[idx] = encode(chunk);
            }
        });
    }
    for (auto& t : pool) t.join();

    u32 total = 0;
    for (auto& cr : cres) total += (u32)cr.size();
    std::vector<TokenID> result;
    result.reserve(total);
    for (auto& cr : cres) {
        result.insert(result.end(), cr.begin(), cr.end());
    }
    return result;
}

std::string Tokenizer::decode(const std::vector<TokenID>& ids) const {
    if (!decoder_) return {};

    u64 total = 0;
    for (TokenID id : ids) {
        total += decoder_->decode(id).size();
    }

    std::string result;
    result.reserve(total);
    for (TokenID id : ids) {
        auto sv = decoder_->decode(id);
        result.append(sv.data(), sv.size());
    }
    return result;
}

void Tokenizer::set_special_tokens(const SpecialTokenConfig& cfg,
                                    const std::vector<std::string>& ss) {
    special_config_ = cfg;
    special_tokens_ = ss;
    injector_ = std::make_unique<SpecialTokenInjector>();
    injector_->configure(cfg, ss);
}

void Tokenizer::enable_parallel(u32 num_threads) {
    parallel_enabled_ = true;
    num_threads_ = num_threads > 0 ? num_threads : std::thread::hardware_concurrency();
    if (num_threads_ == 0) num_threads_ = 4;
}

void Tokenizer::disable_parallel() {
    parallel_enabled_ = false;
    num_threads_ = 1;
}

u64 Tokenizer::total_ram_bytes() const {
    u64 total = 0;
    if (trie_) total += trie_->byte_size();
    if (decoder_) total += decoder_->byte_size();
    if (fst_) total += fst_->byte_size();
    for (auto& s : vocab_) total += s.size();
    total += vocab_.capacity() * sizeof(std::string);
    return total;
}

// ========================================================================
// Differential Re-tokenization (§6.3)
// ========================================================================
std::vector<TokenID> differential_retokenize(
    const Tokenizer& tok, std::string_view original, std::string_view modified,
    const DiffEdit& edit, u32 ctx)
{
    (void)original;
    u32 start = (edit.offset > ctx) ? edit.offset - ctx : 0;
    u32 end = std::min((u32)modified.size(), edit.offset + (u32)edit.insert.size() + ctx);
    auto chunk = modified.substr(start, end - start);
    return tok.encode(chunk);
}

// ========================================================================
// Domain-Adaptive Vocabulary Slicing (§6.4)
// ========================================================================
LoudsTrie build_domain_trie(const Tokenizer& tok, DomainType domain) {
    std::vector<std::pair<std::string, TokenID>> dv;
    auto full = tok.trie().enumerate();
    for (auto& [s, id] : full) {
        bool include = true;
        if (domain == DomainType::Code) {
            for (u8 c : s) {
                if (c < 0x20 && c != '\n' && c != '\t' && c != '\r') {
                    include = false; break;
                }
            }
        } else if (domain == DomainType::Multilingual) {
            include = false;
            for (u8 c : s) { if (c >= 0x80) { include = true; break; } }
        }
        if (include) dv.push_back({s, id});
    }
    LoudsTrie trie;
    trie.build(dv);
    return trie;
}

} // namespace tokenizer
