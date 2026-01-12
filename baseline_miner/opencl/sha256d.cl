// SHA256d OpenCL kernel for Baseline mining
// This kernel exactly replicates the CPU mining algorithm from sha256d.c
// 
// Key endianness considerations:
// - SHA256 operates on big-endian 32-bit words internally
// - Header bytes are loaded as big-endian words (bytes [0-3] form word[0], etc.)
// - Nonce is stored little-endian in header byte position 76-79
// - When loading nonce as a word, we byte-swap it to big-endian
// - Target comparison uses big-endian word ordering

// SHA256 round constants
__constant uint K[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5,
    0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
    0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc,
    0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
    0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
    0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3,
    0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5,
    0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
    0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
};

// SHA256 initial hash values
#define H0 0x6a09e667
#define H1 0xbb67ae85
#define H2 0x3c6ef372
#define H3 0xa54ff53a
#define H4 0x510e527f
#define H5 0x9b05688c
#define H6 0x1f83d9ab
#define H7 0x5be0cd19

// Rotate right
#define ROTR(x, n) (((x) >> (n)) | ((x) << (32 - (n))))

// SHA256 functions
#define CH(x, y, z)  (((x) & (y)) ^ (~(x) & (z)))
#define MAJ(x, y, z) (((x) & (y)) ^ ((x) & (z)) ^ ((y) & (z)))
#define BSIG0(x)     (ROTR((x), 2) ^ ROTR((x), 13) ^ ROTR((x), 22))
#define BSIG1(x)     (ROTR((x), 6) ^ ROTR((x), 11) ^ ROTR((x), 25))
#define SSIG0(x)     (ROTR((x), 7) ^ ROTR((x), 18) ^ ((x) >> 3))
#define SSIG1(x)     (ROTR((x), 17) ^ ROTR((x), 19) ^ ((x) >> 10))

// Byte swap for endianness conversion (little-endian to big-endian and vice versa)
#define BSWAP32(x) (((x) >> 24) | (((x) >> 8) & 0xFF00) | (((x) << 8) & 0xFF0000) | ((x) << 24))

// SHA256 compression function operating on 16 words
// state: input/output state (8 words)
// block: 16 big-endian words
void sha256_compress(uint state[8], const uint block[16]) {
    uint w[16];
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        w[i] = block[i];
    }

    uint a = state[0];
    uint b = state[1];
    uint c = state[2];
    uint d = state[3];
    uint e = state[4];
    uint f = state[5];
    uint g = state[6];
    uint h = state[7];

    // First 16 rounds
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        uint t1 = h + BSIG1(e) + CH(e, f, g) + K[i] + w[i];
        uint t2 = BSIG0(a) + MAJ(a, b, c);
        h = g;
        g = f;
        f = e;
        e = d + t1;
        d = c;
        c = b;
        b = a;
        a = t1 + t2;
    }

    // Rounds 16-63 with message schedule expansion
    #pragma unroll
    for (int i = 16; i < 64; i++) {
        int j = i & 15;
        w[j] = w[j] + SSIG0(w[(j + 1) & 15]) + w[(j + 9) & 15] + SSIG1(w[(j + 14) & 15]);
        uint t1 = h + BSIG1(e) + CH(e, f, g) + K[i] + w[j];
        uint t2 = BSIG0(a) + MAJ(a, b, c);
        h = g;
        g = f;
        f = e;
        e = d + t1;
        d = c;
        c = b;
        b = a;
        a = t1 + t2;
    }

    state[0] += a;
    state[1] += b;
    state[2] += c;
    state[3] += d;
    state[4] += e;
    state[5] += f;
    state[6] += g;
    state[7] += h;
}

// Compare hash state against target (both as big-endian words)
// Returns 1 if hash <= target, 0 otherwise
int hash_le_target(const uint hash[8], const uint target0, const uint target1,
                   const uint target2, const uint target3, const uint target4,
                   const uint target5, const uint target6, const uint target7) {
    if (hash[0] < target0) return 1;
    if (hash[0] > target0) return 0;
    if (hash[1] < target1) return 1;
    if (hash[1] > target1) return 0;
    if (hash[2] < target2) return 1;
    if (hash[2] > target2) return 0;
    if (hash[3] < target3) return 1;
    if (hash[3] > target3) return 0;
    if (hash[4] < target4) return 1;
    if (hash[4] > target4) return 0;
    if (hash[5] < target5) return 1;
    if (hash[5] > target5) return 0;
    if (hash[6] < target6) return 1;
    if (hash[6] > target6) return 0;
    if (hash[7] < target7) return 1;
    if (hash[7] > target7) return 0;
    return 1;
}

// Main mining kernel
// midstate: pre-computed SHA256 state after first 64-byte block (8 words)
// block2_base: second block template (16 words, nonce position at [3] will be filled in)
// target: target threshold as 8 big-endian words
// start_nonce: starting nonce value
// results: output buffer for found nonces
// result_hashes: output buffer for corresponding hashes (8 words per result)
// result_count: atomic counter for number of results
// max_results: maximum results to store
__kernel void scan_nonces(
    __constant uint *midstate,
    __constant uint *block2_base,
    __constant uint *target,
    uint start_nonce,
    __global uint *results,
    __global uint *result_hashes,
    __global volatile uint *result_count,
    uint max_results
) {
    uint gid = get_global_id(0);
    uint nonce = start_nonce + gid;
    
    // Check for nonce overflow
    if ((uint)(start_nonce + gid) < start_nonce && gid != 0) {
        return;
    }

    // Prepare block2 with this nonce
    // Block2 layout from CPU code:
    // [0-2]: bytes 64-75 of header (last 12 bytes before nonce)
    // [3]: nonce (byte-swapped from little-endian to big-endian)
    // [4]: 0x80000000 (padding start)
    // [5-14]: zeros
    // [15]: 0x00000280 (length = 80 bytes * 8 = 640 bits)
    uint block2[16];
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        block2[i] = block2_base[i];
    }
    // Nonce is stored little-endian, convert to big-endian word
    block2[3] = BSWAP32(nonce);

    // First SHA256: compress block2 with midstate
    uint state1[8];
    state1[0] = midstate[0];
    state1[1] = midstate[1];
    state1[2] = midstate[2];
    state1[3] = midstate[3];
    state1[4] = midstate[4];
    state1[5] = midstate[5];
    state1[6] = midstate[6];
    state1[7] = midstate[7];
    sha256_compress(state1, block2);

    // Second SHA256: hash the first hash result
    // Block layout for second hash:
    // [0-7]: first hash state words (already big-endian)
    // [8]: 0x80000000 (padding start)
    // [9-14]: zeros
    // [15]: 0x00000100 (length = 32 bytes * 8 = 256 bits)
    uint hash_block[16];
    hash_block[0] = state1[0];
    hash_block[1] = state1[1];
    hash_block[2] = state1[2];
    hash_block[3] = state1[3];
    hash_block[4] = state1[4];
    hash_block[5] = state1[5];
    hash_block[6] = state1[6];
    hash_block[7] = state1[7];
    hash_block[8] = 0x80000000;
    hash_block[9] = 0;
    hash_block[10] = 0;
    hash_block[11] = 0;
    hash_block[12] = 0;
    hash_block[13] = 0;
    hash_block[14] = 0;
    hash_block[15] = 0x00000100;

    // Initialize state2 with SHA256 initial values
    uint state2[8];
    state2[0] = H0;
    state2[1] = H1;
    state2[2] = H2;
    state2[3] = H3;
    state2[4] = H4;
    state2[5] = H5;
    state2[6] = H6;
    state2[7] = H7;
    sha256_compress(state2, hash_block);

    // Check if hash meets target
    // Load target to private memory to avoid address space issues on Apple OpenCL
    if (hash_le_target(state2, target[0], target[1], target[2], target[3],
                       target[4], target[5], target[6], target[7])) {
        // Atomically increment result counter and store result
        uint idx = atomic_inc(result_count);
        if (idx < max_results) {
            results[idx] = nonce;
            // Store hash (8 words)
            #pragma unroll
            for (int i = 0; i < 8; i++) {
                result_hashes[idx * 8 + i] = state2[i];
            }
        }
    }
}

// Kernel to compute midstate from first 64-byte block
// This is run once per job to precompute the midstate
__kernel void compute_midstate(
    __constant uchar *block1_bytes,
    __global uint *midstate_out
) {
    // Load 64 bytes as 16 big-endian words
    uint block1[16];
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        uint b0 = block1_bytes[i * 4];
        uint b1 = block1_bytes[i * 4 + 1];
        uint b2 = block1_bytes[i * 4 + 2];
        uint b3 = block1_bytes[i * 4 + 3];
        block1[i] = (b0 << 24) | (b1 << 16) | (b2 << 8) | b3;
    }

    // Initialize with SHA256 initial values
    uint state[8];
    state[0] = H0;
    state[1] = H1;
    state[2] = H2;
    state[3] = H3;
    state[4] = H4;
    state[5] = H5;
    state[6] = H6;
    state[7] = H7;

    // Compress first block
    sha256_compress(state, block1);

    // Output midstate
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        midstate_out[i] = state[i];
    }
}

// Kernel to prepare block2 base from remaining header bytes
// header_tail: bytes 64-75 of header (12 bytes)
// block2_out: output 16 words for block2 template
__kernel void prepare_block2(
    __constant uchar *header_tail,
    __global uint *block2_out
) {
    // Load bytes 64-75 as big-endian words
    uint w0 = ((uint)header_tail[0] << 24) | ((uint)header_tail[1] << 16) | 
              ((uint)header_tail[2] << 8) | (uint)header_tail[3];
    uint w1 = ((uint)header_tail[4] << 24) | ((uint)header_tail[5] << 16) | 
              ((uint)header_tail[6] << 8) | (uint)header_tail[7];
    uint w2 = ((uint)header_tail[8] << 24) | ((uint)header_tail[9] << 16) | 
              ((uint)header_tail[10] << 8) | (uint)header_tail[11];

    block2_out[0] = w0;
    block2_out[1] = w1;
    block2_out[2] = w2;
    block2_out[3] = 0;  // Nonce placeholder
    block2_out[4] = 0x80000000;  // Padding start
    block2_out[5] = 0;
    block2_out[6] = 0;
    block2_out[7] = 0;
    block2_out[8] = 0;
    block2_out[9] = 0;
    block2_out[10] = 0;
    block2_out[11] = 0;
    block2_out[12] = 0;
    block2_out[13] = 0;
    block2_out[14] = 0;
    block2_out[15] = 0x00000280;  // Length: 80 * 8 = 640 bits
}
