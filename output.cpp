// ==========================================================================
// output.cpp — Component 1.1 Tokenizer C API
//
// Cognition-Optimized BPE Tokenizer: LOUDS trie, CHD decoder, FST engine,
// entropy-weighted training, parallel encode, streaming support.
//
// Usage (single header):
//   #include "output.cpp"
//   TokenizerHandle* tok = tok_create_byte_fallback();
//   int ids[256];
//   int n = tok_encode(tok, "hello world", ids, 256);
//   tok_destroy(tok);
//
// Compile:
//   g++ -std=c++17 -O2 -c output.cpp -o tokenizer_api.o
//   (or include directly as a single translation unit)
// ==========================================================================

#include "core.hpp"

#ifdef __cplusplus
extern "C" {
#endif

// ==========================================================================
// Types
// ==========================================================================

// Opaque handle — all API functions operate through this
typedef struct TokenizerHandle TokenizerHandle;

// Result from tok_encode: returns number of tokens written
typedef struct TokenizerResult {
    int   n;         // number of tokens written
    int   truncated; // 1 if output was truncated, 0 otherwise
} TokenizerResult;

// ==========================================================================
// Lifecycle
// ==========================================================================

// Create tokenizer with byte-level fallback vocabulary (V=256).
// All 256 bytes 0x00-0xFF map to unique token IDs.
TokenizerHandle* tok_create_byte_fallback(void);

// Create tokenizer by loading a .tok vocabulary file.
// Returns NULL if the file cannot be loaded.
TokenizerHandle* tok_create_from_file(const char* path);

// Create tokenizer from an explicit vocabulary array.
// Takes ownership of the data.
//   vocab: array of strings
//   n: number of tokens
TokenizerHandle* tok_create_from_vocab(const char** vocab, int n);

// Destroy tokenizer and free all associated memory.
void tok_destroy(TokenizerHandle* handle);

// ==========================================================================
// Encoding
// ==========================================================================

// Encode a single text string into token IDs.
//   text: input UTF-8 string (null-terminated)
//   ids: output buffer for token IDs
//   max_ids: capacity of output buffer
// Returns {n, truncated} where n is the number of tokens written.
TokenizerResult tok_encode(
    const TokenizerHandle* handle,
    const char* text,
    int* ids,
    int max_ids
);

// Encode text using parallel chunked encoding.
//   num_threads: 0 = use hardware concurrency
TokenizerResult tok_encode_parallel(
    const TokenizerHandle* handle,
    const char* text,
    int* ids,
    int max_ids,
    int num_threads
);

// Encode multiple texts in batch (single-threaded).
//   texts: array of null-terminated strings
//   num_texts: number of strings
//   offsets: output, offsets[i] = token count before text i
//   ids: output buffer for all token IDs (concatenated)
//   max_ids: capacity of output buffer
// Returns total number of tokens written across all texts.
TokenizerResult tok_encode_batch(
    const TokenizerHandle* handle,
    const char** texts,
    int num_texts,
    int* offsets,
    int* ids,
    int max_ids
);

// ==========================================================================
// Decoding
// ==========================================================================

// Decode token IDs back into a UTF-8 string.
//   ids: input token ID array
//   n: number of token IDs
//   text: output buffer for decoded string
//   max_chars: capacity of output buffer
// Returns number of characters written (excluding null terminator).
int tok_decode(
    const TokenizerHandle* handle,
    const int* ids,
    int n,
    char* text,
    int max_chars
);

// ==========================================================================
// Training
// ==========================================================================

// Train a new vocabulary from a corpus using entropy-weighted BPE.
//   corpus: training text (UTF-8 bytes)
//   corpus_len: length of corpus in bytes
//   target_vocab_size: desired vocabulary size (includes 256 byte tokens)
//   handle_out: receives new tokenizer with trained vocabulary
// Returns 0 on success, -1 on failure.
int tok_train(
    const char* corpus,
    int corpus_len,
    int target_vocab_size,
    TokenizerHandle** handle_out
);

// ==========================================================================
// Persistence
// ==========================================================================

// Save tokenizer vocabulary to a .tok file.
// Returns 0 on success, -1 on failure.
int tok_save(const TokenizerHandle* handle, const char* path);

// Load tokenizer vocabulary from a .tok file.
// Returns new handle, or NULL on failure.
TokenizerHandle* tok_load(const char* path);

// ==========================================================================
// Accessors
// ==========================================================================

// Returns vocabulary size (number of tokens).
int tok_vocab_size(const TokenizerHandle* handle);

// Returns total RAM usage in bytes.
long long tok_total_ram_bytes(const TokenizerHandle* handle);

// Returns the token ID for a given string, or -1 if not found.
int tok_lookup(const TokenizerHandle* handle, const char* token_str);

// Decode a single token ID to its string representation.
//   id: token ID
//   out: output buffer
//   max_out: capacity of output buffer
// Returns number of bytes written, or -1 if ID out of range.
int tok_decode_single(const TokenizerHandle* handle, int id, char* out, int max_out);

// ==========================================================================
// FST Management
// ==========================================================================

// Compile the merge FST from the current vocabulary.
// Required for FST-based encoding (used by default after training).
void tok_compile_fst(TokenizerHandle* handle);

// ==========================================================================
// Special Tokens
// ==========================================================================

// Configure special tokens (BOS, EOS, PAD, UNK).
//   bos_id: BOS token ID (-1 to disable)
//   eos_id: EOS token ID (-1 to disable)
//   pad_id: PAD token ID (-1 to disable)
//   unk_id: UNK token ID (-1 to disable)
//   add_bos: if non-zero, prepend BOS on encode
//   add_eos: if non-zero, append EOS on encode
void tok_set_special_tokens(
    TokenizerHandle* handle,
    int bos_id, int eos_id, int pad_id, int unk_id,
    int add_bos, int add_eos
);

// ==========================================================================
// Parallel Configuration
// ==========================================================================

// Enable parallel encoding with given thread count.
//  num_threads: 0 = auto (hardware concurrency)
void tok_enable_parallel(TokenizerHandle* handle, int num_threads);

// Disable parallel encoding (revert to single-threaded FST).
void tok_disable_parallel(TokenizerHandle* handle);

// ==========================================================================
// Streaming
// ==========================================================================

// Opaque streaming state handle.
typedef struct TokenizerStreamState TokenizerStreamState;

// Create a streaming encode session.
// Returns handle to streaming state.
TokenizerStreamState* tok_stream_create(void);

// Encode a chunk of streaming input.
//   state: streaming state (from tok_stream_create)
//   chunk: input bytes (may be partial)
//   chunk_len: length of chunk
//   ids: output buffer for complete tokens
//   max_ids: capacity of output buffer
// Returns number of complete tokens emitted.
TokenizerResult tok_stream_encode(
    TokenizerHandle* handle,
    TokenizerStreamState* state,
    const char* chunk,
    int chunk_len,
    int* ids,
    int max_ids
);

// Flush remaining partial state at end of stream.
// Returns complete tokens from remaining buffer.
TokenizerResult tok_stream_flush(
    TokenizerHandle* handle,
    TokenizerStreamState* state,
    int* ids,
    int max_ids
);

// Destroy streaming state.
void tok_stream_destroy(TokenizerStreamState* state);

#ifdef __cplusplus
} // extern "C"
#endif

// ==========================================================================
// Implementation
// ==========================================================================

struct TokenizerHandle {
    tokenizer::Tokenizer tok;
};

struct TokenizerStreamState {
    tokenizer::Tokenizer::StreamState state;
};

TokenizerHandle* tok_create_byte_fallback(void) {
    auto* h = new TokenizerHandle();
    std::vector<std::string> v;
    v.reserve(256);
    for (int i = 0; i < 256; ++i)
        v.push_back(std::string(1, (char)(uint8_t)i));
    h->tok.set_vocab(v);
    return h;
}

TokenizerHandle* tok_create_from_file(const char* path) {
    auto* h = new TokenizerHandle();
    if (!h->tok.load(path)) {
        delete h;
        return nullptr;
    }
    return h;
}

TokenizerHandle* tok_create_from_vocab(const char** vocab, int n) {
    auto* h = new TokenizerHandle();
    std::vector<std::string> v(n);
    for (int i = 0; i < n; ++i)
        v[i] = vocab[i];
    h->tok.set_vocab(v);
    return h;
}

void tok_destroy(TokenizerHandle* handle) {
    delete handle;
}

TokenizerResult tok_encode(
    const TokenizerHandle* handle,
    const char* text,
    int* ids,
    int max_ids
) {
    TokenizerResult r = {0, 0};
    if (!handle || !text || !ids || max_ids <= 0) return r;

    auto token_vec = handle->tok.encode(text);
    int n = (int)token_vec.size();
    if (n > max_ids) { n = max_ids; r.truncated = 1; }
    for (int i = 0; i < n; ++i) ids[i] = (int)token_vec[i];
    r.n = n;
    return r;
}

TokenizerResult tok_encode_parallel(
    const TokenizerHandle* handle,
    const char* text,
    int* ids,
    int max_ids,
    int num_threads
) {
    TokenizerResult r = {0, 0};
    if (!handle || !text || !ids || max_ids <= 0) return r;

    tokenizer::u32 nt = (num_threads <= 0) ? 0 : (tokenizer::u32)num_threads;
    auto token_vec = handle->tok.encode_parallel(text, nt);
    int n = (int)token_vec.size();
    if (n > max_ids) { n = max_ids; r.truncated = 1; }
    for (int i = 0; i < n; ++i) ids[i] = (int)token_vec[i];
    r.n = n;
    return r;
}

TokenizerResult tok_encode_batch(
    const TokenizerHandle* handle,
    const char** texts,
    int num_texts,
    int* offsets,
    int* ids,
    int max_ids
) {
    TokenizerResult r = {0, 0};
    if (!handle || !texts || !offsets || !ids || max_ids <= 0 || num_texts <= 0)
        return r;

    int total = 0;
    for (int i = 0; i < num_texts; ++i) {
        offsets[i] = total;
        auto token_vec = handle->tok.encode(texts[i]);
        int n = (int)token_vec.size();
        int to_copy = n;
        if (total + to_copy > max_ids) {
            to_copy = max_ids - total;
            r.truncated = 1;
        }
        for (int j = 0; j < to_copy; ++j)
            ids[total + j] = (int)token_vec[j];
        total += to_copy;
        if (r.truncated) break;
    }
    r.n = total;
    return r;
}

int tok_decode(
    const TokenizerHandle* handle,
    const int* ids,
    int n,
    char* text,
    int max_chars
) {
    if (!handle || !ids || !text || max_chars <= 0 || n <= 0) return 0;

    std::vector<tokenizer::TokenID> tids(n);
    for (int i = 0; i < n; ++i) tids[i] = (tokenizer::TokenID)ids[i];

    auto decoded = handle->tok.decode(tids);
    int len = (int)decoded.size();
    if (len >= max_chars) len = max_chars - 1;
    std::memcpy(text, decoded.data(), (size_t)len);
    text[len] = '\0';
    return len;
}

int tok_train(
    const char* corpus,
    int corpus_len,
    int target_vocab_size,
    TokenizerHandle** handle_out
) {
    if (!corpus || corpus_len <= 0 || !handle_out) return -1;

    try {
        auto* h = new TokenizerHandle();
        std::string_view sv(corpus, (size_t)corpus_len);
        h->tok.train(sv, (tokenizer::u32)target_vocab_size);
        *handle_out = h;
        return 0;
    } catch (...) {
        return -1;
    }
}

int tok_save(const TokenizerHandle* handle, const char* path) {
    if (!handle || !path) return -1;
    return handle->tok.save(path) ? 0 : -1;
}

TokenizerHandle* tok_load(const char* path) {
    if (!path) return nullptr;
    return tok_create_from_file(path);
}

int tok_vocab_size(const TokenizerHandle* handle) {
    return handle ? (int)handle->tok.vocab_size() : 0;
}

long long tok_total_ram_bytes(const TokenizerHandle* handle) {
    return handle ? (long long)handle->tok.total_ram_bytes() : 0;
}

int tok_lookup(const TokenizerHandle* handle, const char* token_str) {
    if (!handle || !token_str) return -1;
    auto id = handle->tok.trie().lookup(token_str);
    return id.has_value() ? (int)*id : -1;
}

int tok_decode_single(const TokenizerHandle* handle, int id, char* out, int max_out) {
    if (!handle || !out || max_out <= 0 || id < 0) return -1;
    auto sv = handle->tok.decoder().decode((tokenizer::u32)id);
    int len = (int)sv.size();
    if (len >= max_out) len = max_out - 1;
    std::memcpy(out, sv.data(), (size_t)len);
    out[len] = '\0';
    return len;
}

void tok_compile_fst(TokenizerHandle* handle) {
    if (handle) handle->tok.compile_fst();
}

void tok_set_special_tokens(
    TokenizerHandle* handle,
    int bos_id, int eos_id, int pad_id, int unk_id,
    int add_bos, int add_eos
) {
    if (!handle) return;
    tokenizer::SpecialTokenConfig cfg;
    cfg.bos_id = (tokenizer::TokenID)(bos_id >= 0 ? bos_id : 0);
    cfg.eos_id = (tokenizer::TokenID)(eos_id >= 0 ? eos_id : 0);
    cfg.pad_id = (tokenizer::TokenID)(pad_id >= 0 ? pad_id : 0);
    cfg.unk_id = (tokenizer::TokenID)(unk_id >= 0 ? unk_id : 0);
    cfg.add_bos = add_bos != 0;
    cfg.add_eos = add_eos != 0;
    std::vector<std::string> special_strings;
    handle->tok.set_special_tokens(cfg, special_strings);
}

void tok_enable_parallel(TokenizerHandle* handle, int num_threads) {
    if (handle) handle->tok.enable_parallel((tokenizer::u32)(num_threads > 0 ? num_threads : 0));
}

void tok_disable_parallel(TokenizerHandle* handle) {
    if (handle) handle->tok.disable_parallel();
}

TokenizerStreamState* tok_stream_create(void) {
    return new TokenizerStreamState();
}

TokenizerResult tok_stream_encode(
    TokenizerHandle* handle,
    TokenizerStreamState* state,
    const char* chunk,
    int chunk_len,
    int* ids,
    int max_ids
) {
    TokenizerResult r = {0, 0};
    if (!handle || !state || !ids || max_ids <= 0) return r;
    std::string_view sv(chunk, chunk_len > 0 ? (size_t)chunk_len : 0);
    auto token_vec = handle->tok.encode_streaming(sv, state->state);
    int n = (int)token_vec.size();
    if (n > max_ids) { n = max_ids; r.truncated = 1; }
    for (int i = 0; i < n; ++i) ids[i] = (int)token_vec[i];
    r.n = n;
    return r;
}

TokenizerResult tok_stream_flush(
    TokenizerHandle* handle,
    TokenizerStreamState* state,
    int* ids,
    int max_ids
) {
    TokenizerResult r = {0, 0};
    if (!handle || !state || !ids || max_ids <= 0) return r;
    // Flush by encoding an empty chunk: the streaming encoder
    // will flush any buffered partial token.
    auto token_vec = handle->tok.encode_streaming("", state->state);
    int n = (int)token_vec.size();
    if (n > max_ids) { n = max_ids; r.truncated = 1; }
    for (int i = 0; i < n; ++i) ids[i] = (int)token_vec[i];
    r.n = n;
    return r;
}

void tok_stream_destroy(TokenizerStreamState* state) {
    delete state;
}
