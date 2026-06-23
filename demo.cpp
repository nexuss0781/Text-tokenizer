// demo.cpp — Full tokenizer demonstration
#include "core.hpp"
#include <iostream>

int main() {
    using namespace tokenizer;

    std::cout << "=== Component 1.1 Tokenizer Demo ===\n\n";

    // 1. Build tokenizer with byte-level + multi-byte tokens
    Tokenizer tok;
    std::vector<std::string> v;
    for (int i = 0; i < 256; i++)
        v.push_back(std::string(1, (char)i));
    v.push_back("hello");  // 256
    v.push_back(" ");      // 257
    v.push_back("world");  // 258
    tok.set_vocab(v);

    // 2. LOUDS trie lookup
    std::cout << "LOUDS Trie lookups:\n";
    for (auto s : {"hello", "world", " ", "h", "notfound"}) {
        auto id = tok.trie().lookup(s);
        if (id) std::cout << "  \"" << s << "\" -> ID " << *id << "\n";
        else    std::cout << "  \"" << s << "\" -> not found\n";
    }

    // 3. CHD Decoder
    std::cout << "\nCHD Decoder lookups:\n";
    for (auto id : {256, 258, 257, 104}) {
        auto sv = tok.decoder().decode(id);
        std::cout << "  ID " << id << " -> \"" << std::string(sv) << "\"\n";
    }

    // 4. Encode + Decode
    std::cout << "\nEncode/Decode:\n";
    for (auto input : {"hello world", "hello", "world", "h", ""}) {
        auto ids = tok.encode(input);
        auto dec = tok.decode(ids);
        std::cout << "  \"" << input << "\" -> [";
        for (size_t i = 0; i < ids.size(); i++) {
            if (i) std::cout << ",";
            std::cout << ids[i];
        }
        std::cout << "] -> \"" << dec << "\""
                  << (input == dec ? " ✓" : " ✗") << "\n";
    }

    // 5. RAM usage
    std::cout << "\nRAM: " << tok.total_ram_bytes() / 1024.0 << " KB\n";
    std::cout << "Vocab: " << tok.vocab_size() << " tokens\n";

    // 6. Verify Invariant 9.1: all 256 bytes map to exactly one token
    bool all_bytes_ok = true;
    for (int b = 0; b < 256; b++) {
        std::string s(1, (char)b);
        auto ids = tok.encode(s);
        if (ids.empty()) { all_bytes_ok = false; break; }
    }
    std::cout << "\nInvariant 9.1 (256 byte coverage): "
              << (all_bytes_ok ? "PASS" : "FAIL") << "\n";

    // 7. Verify round-trip for random strings
    std::mt19937 rng(42);
    bool rt_ok = true;
    for (int i = 0; i < 10; i++) {
        std::string s;
        int len = std::uniform_int_distribution<int>(1, 50)(rng);
        for (int j = 0; j < len; j++)
            s.push_back((char)std::uniform_int_distribution<int>(32, 126)(rng));
        auto ids = tok.encode(s);
        auto dec = tok.decode(ids);
        if (dec != s) { rt_ok = false; break; }
    }
    std::cout << "Random round-trip (10 runs): "
              << (rt_ok ? "PASS" : "FAIL") << "\n";

    return 0;
}
