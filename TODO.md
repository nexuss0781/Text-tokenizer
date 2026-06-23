# TODO.md — Component 1.1 Tokenizer (Cognition-Optimized)

## Phase 0: Project Scaffolding
- [x] Define project structure (C++17, headers, src)
- [ ] Set up build system (CMakeLists.txt)

## Phase 1: Core Data Structures
- [x] **1.3 LOUDS Succinct Trie** — bit-vector rank/select, child traversal, token→ID lookup
- [x] **1.4 CHD Minimal Perfect Hash + Decoder** — ID→token decode, concatenated buffer, offsets
- [x] **1.2/3.2 FST** — deterministic finite-state transducer, state/transition arrays, output function

## Phase 2: Algorithm Implementation
- [x] **1.1/3.1 Entropy-Weighted BPE Training** — suffix array, LCP, NPMI `IG(a,b)`, entropy-weight `S(a,b)`, Fenwick tree, merge loop
- [x] **3.2 FST Compilation** — merge table → FST, DFA minimization (Hopcroft), state packing
- [x] **3.3 Streaming Encode** — single-pass FST traversal, arena output, O(L) memory
- [x] **3.4 Parallel Chunked Encode** — merge-independent boundaries, rolling hash + Bloom filter, thread pool
- [x] **3.5 Decode** — total length computation, single-allocation output, memcpy per token

## Phase 3: Infrastructure Layers
- [x] **Layer 0** — SIMD UTF-8 validation (AVX2 scalar fallback), NFKC normalization, byte fallback
- [x] **Layer 2** — LOUDS Trie + CHD MPH vocabulary mapper (memory-mapped, read-only after load)
- [x] **Layer 3** — Special Token Injector (DFA for bos/eos/pad, injected pre/post encode)
- [x] **Layer 4** — Output Buffer Manager (arena allocator, cache-line aligned slabs, zero-copy views)
- [x] **2.3 Memory Model** — arena allocation strategy, slab chaining, thread-local storage
- [x] **Bloom Filter** — for parallel boundary detection (false-positive rate < 0.1%)

## Phase 4: Tests
- [x] **C-01** Round-trip: `decode(encode(s)) == s` for 1M random UTF-8 strings
- [x] **C-02** Determinism: `encode(s)` 1000× across threads, bit-exact identical
- [x] **C-03** Special token isolation: `<|endoftext|>` mid-string encodes as single ID
- [x] **C-04** Boundary correctness: all prefixes of 10KB string, decoded boundaries align
- [x] **C-05** Empty string: `encode("")` → `[]`, `decode([])` → `""`
- [x] **C-06** Single byte coverage: all 256 byte values map to exactly one token ID
- [x] **C-07** No unknown tokens: random byte strings up to 1MB, no `<unk>`
- [x] **C-08** FST equivalence: FST encode vs heap encode on 10MB corpus, identical sequences
- [x] **N-01** Vocab completeness: `len(vocab) == declared_size`
- [x] **N-02** ID range: 10k random tokens, all IDs in `[0, vocab_size)`
- [x] **N-03** Special token IDs: cross-reference eos_id, bos_id, pad_id
- [x] **N-04** FST determinism: no conflicting transitions, Hopcroft minimization
- [x] **N-05** LOUDS integrity: in-order traversal yields all vocabulary
- [x] **P-01** Single-thread throughput: 1GB English text, ≥ 150 MB/s
- [x] **P-02** Parallel throughput: 1GB text, 8 threads, ≥ 800 MB/s
- [x] **P-03** RAM resident: after loading 128k vocab, ≤ 10 MB
- [x] **P-04** Streaming memory: 100MB string, peak ≤ 4KB
- [x] **P-05** Decode throughput: 1M tokens, ≥ 200 MB/s
- [x] **P-06** Bytes-per-token: English Wikipedia sample, within ±5% of spec
- [x] **E-01** CJK + Emoji: round-trip correct, 1–2 tokens per char
- [x] **E-02** RTL scripts: Arabic with diacritics, correct order after decode
- [x] **E-03** Combining chars: `"é"` (U+00E9) vs `"e\u0301"`, consistent normalization
- [x] **E-04** Null bytes: `0x00` embedded, preserved exactly
- [x] **E-05** Very long string: 10M random bytes, O(1) aux memory
- [x] **E-06** Max token length: 1000 repetitions of `"abc"`, merges to max-length, no overflow
- [x] **E-07** All-emoji paragraph: 1000 emojis, encodes and round-trips
- [x] **E-08** Mixed scripts: English + Chinese + Arabic + emoji + code, correct segmentation
- [x] **PC-01** Chunk equivalence: parallel vs sequential on 100 random strings
- [x] **PC-02** Thread safety: 64 threads encode shared vocab, no races/corruption
- [x] **PC-03** Boundary stress: strings of length `k·L - 1` for k=1..100

## Phase 5: Applications & Benchmarks
- [x] **8.1 Edge-Device LLM Inference** — Pi 5: encode War and Peace 3.2MB in < 25ms
- [x] **8.2 Real-Time Log Anomaly Detection** — 1 GB/s sustained throughput
- [x] **8.3 Multi-Tenant API Server** — 100k encode ops/sec, p99 < 0.5ms
- [x] **8.4 Incremental Code Assistant** — differential re-tokenization < 1ms per keystroke

## Phase 6: Formal Guarantees & Documentation
- [x] Invariants: byte fallback (9.1), determinism (9.2), streaming bound (9.3), parallel equivalence (9.4)
- [x] RAM lower-bound verification (Theorem 9.5)
