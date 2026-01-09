#include <Python.h>
#include <stdint.h>
#include <string.h>

typedef void (*sha256_block_fn)(uint32_t state[8], const uint8_t data[64]);

typedef struct {
    uint8_t data[64];
    uint32_t datalen;
    uint64_t bitlen;
    uint32_t state[8];
} SHA256_CTX;


#if defined(_MSC_VER)
#include <intrin.h>
#define SHA256D_FORCEINLINE __forceinline
static SHA256D_FORCEINLINE uint32_t sha256d_bswap32(uint32_t x) { return _byteswap_ulong(x); }
static SHA256D_FORCEINLINE uint32_t sha256d_rotr32(uint32_t x, uint32_t n) { return _rotr(x, (int)n); }
#else
#define SHA256D_FORCEINLINE inline __attribute__((always_inline))
static SHA256D_FORCEINLINE uint32_t sha256d_bswap32(uint32_t x) { return __builtin_bswap32(x); }
static SHA256D_FORCEINLINE uint32_t sha256d_rotr32(uint32_t x, uint32_t n) { return (x >> n) | (x << (32U - n)); }
#endif

static SHA256D_FORCEINLINE uint32_t sha256d_load_be32(const uint8_t *p) {
    uint32_t v;
    memcpy(&v, p, sizeof(v));
#if defined(__BYTE_ORDER__) && defined(__ORDER_BIG_ENDIAN__) && (__BYTE_ORDER__ == __ORDER_BIG_ENDIAN__)
    return v;
#else
    return sha256d_bswap32(v);
#endif
}

static SHA256D_FORCEINLINE void sha256d_store_be32(uint8_t *p, uint32_t v) {
#if defined(__BYTE_ORDER__) && defined(__ORDER_BIG_ENDIAN__) && (__BYTE_ORDER__ == __ORDER_BIG_ENDIAN__)
    memcpy(p, &v, sizeof(v));
#else
    v = sha256d_bswap32(v);
    memcpy(p, &v, sizeof(v));
#endif
}

static const uint32_t k[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
};

#define SHA256D_CH(x, y, z) (((x) & (y)) ^ (~(x) & (z)))
#define SHA256D_MAJ(x, y, z) (((x) & (y)) ^ ((x) & (z)) ^ ((y) & (z)))
#define SHA256D_BSIG0(x) (sha256d_rotr32((x), 2) ^ sha256d_rotr32((x), 13) ^ sha256d_rotr32((x), 22))
#define SHA256D_BSIG1(x) (sha256d_rotr32((x), 6) ^ sha256d_rotr32((x), 11) ^ sha256d_rotr32((x), 25))
#define SHA256D_SSIG0(x) (sha256d_rotr32((x), 7) ^ sha256d_rotr32((x), 18) ^ ((x) >> 3))
#define SHA256D_SSIG1(x) (sha256d_rotr32((x), 17) ^ sha256d_rotr32((x), 19) ^ ((x) >> 10))

static SHA256D_FORCEINLINE void sha256_init_state(uint32_t state[8]) {
    state[0] = 0x6a09e667;
    state[1] = 0xbb67ae85;
    state[2] = 0x3c6ef372;
    state[3] = 0xa54ff53a;
    state[4] = 0x510e527f;
    state[5] = 0x9b05688c;
    state[6] = 0x1f83d9ab;
    state[7] = 0x5be0cd19;
}

static SHA256D_FORCEINLINE void sha256_compress_fast_words(uint32_t state[8], const uint32_t block[16]) {
    uint32_t w[16];
    for (int i = 0; i < 16; ++i) {
        w[i] = block[i];
    }

    uint32_t a = state[0];
    uint32_t b = state[1];
    uint32_t c = state[2];
    uint32_t d = state[3];
    uint32_t e = state[4];
    uint32_t f = state[5];
    uint32_t g = state[6];
    uint32_t h = state[7];

    for (int i = 0; i < 16; ++i) {
        const uint32_t t1 = h + SHA256D_BSIG1(e) + SHA256D_CH(e, f, g) + k[i] + w[i];
        const uint32_t t2 = SHA256D_BSIG0(a) + SHA256D_MAJ(a, b, c);
        h = g;
        g = f;
        f = e;
        e = d + t1;
        d = c;
        c = b;
        b = a;
        a = t1 + t2;
    }

    for (int i = 16; i < 64; ++i) {
        const int j = i & 15;
        w[j] = w[j] + SHA256D_SSIG0(w[(j + 1) & 15]) + w[(j + 9) & 15] + SHA256D_SSIG1(w[(j + 14) & 15]);
        const uint32_t t1 = h + SHA256D_BSIG1(e) + SHA256D_CH(e, f, g) + k[i] + w[j];
        const uint32_t t2 = SHA256D_BSIG0(a) + SHA256D_MAJ(a, b, c);
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

static void sha256_compress_fast(uint32_t state[8], const uint8_t data[64]) {
    uint32_t block[16];
    for (int i = 0; i < 16; ++i) {
        block[i] = sha256d_load_be32(data + (size_t)i * 4U);
    }
    sha256_compress_fast_words(state, block);
}

static void sha256_compress_portable(uint32_t state[8], const uint8_t data[64]) {
    uint32_t m[64];
    uint32_t a, b, c, d, e, f, g, h;
    uint32_t t1, t2;

    for (int i = 0; i < 16; ++i) {
        m[i] = (uint32_t)data[i * 4] << 24 | (uint32_t)data[i * 4 + 1] << 16 |
               (uint32_t)data[i * 4 + 2] << 8 | (uint32_t)data[i * 4 + 3];
    }
    for (int i = 16; i < 64; ++i) {
        m[i] = SHA256D_SSIG1(m[i - 2]) + m[i - 7] + SHA256D_SSIG0(m[i - 15]) + m[i - 16];
    }

    a = state[0];
    b = state[1];
    c = state[2];
    d = state[3];
    e = state[4];
    f = state[5];
    g = state[6];
    h = state[7];

    for (int i = 0; i < 64; ++i) {
        t1 = h + SHA256D_BSIG1(e) + SHA256D_CH(e, f, g) + k[i] + m[i];
        t2 = SHA256D_BSIG0(a) + SHA256D_MAJ(a, b, c);
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

static sha256_block_fn g_compress = sha256_compress_fast;
static const char *g_backend = "fast";

static void sha256_transform(SHA256_CTX *ctx, const uint8_t data[64]) {
    g_compress(ctx->state, data);
}

static void sha256_init(SHA256_CTX *ctx) {
    ctx->datalen = 0;
    ctx->bitlen = 0;
    sha256_init_state(ctx->state);
}

static void sha256_update(SHA256_CTX *ctx, const uint8_t data[], size_t len) {
    size_t i = 0;

    if (len == 0) {
        return;
    }

    if (ctx->datalen != 0) {
        const size_t needed = 64U - (size_t)ctx->datalen;
        const size_t take = (len < needed) ? len : needed;
        memcpy(ctx->data + ctx->datalen, data, take);
        ctx->datalen += (uint32_t)take;
        i += take;
        if (ctx->datalen == 64U) {
            sha256_transform(ctx, ctx->data);
            ctx->bitlen += 512U;
            ctx->datalen = 0;
        }
    }

    for (; i + 64U <= len; i += 64U) {
        sha256_transform(ctx, data + i);
        ctx->bitlen += 512U;
    }

    if (i < len) {
        const size_t tail = len - i;
        memcpy(ctx->data, data + i, tail);
        ctx->datalen = (uint32_t)tail;
    }
}

static void sha256_final(SHA256_CTX *ctx, uint8_t hash[32]) {
    uint32_t i = ctx->datalen;

    if (ctx->datalen < 56) {
        ctx->data[i++] = 0x80;
        while (i < 56) {
            ctx->data[i++] = 0x00;
        }
    } else {
        ctx->data[i++] = 0x80;
        while (i < 64) {
            ctx->data[i++] = 0x00;
        }
        sha256_transform(ctx, ctx->data);
        memset(ctx->data, 0, 56);
    }

    ctx->bitlen += ctx->datalen * 8;
    ctx->data[63] = (uint8_t)(ctx->bitlen);
    ctx->data[62] = (uint8_t)(ctx->bitlen >> 8);
    ctx->data[61] = (uint8_t)(ctx->bitlen >> 16);
    ctx->data[60] = (uint8_t)(ctx->bitlen >> 24);
    ctx->data[59] = (uint8_t)(ctx->bitlen >> 32);
    ctx->data[58] = (uint8_t)(ctx->bitlen >> 40);
    ctx->data[57] = (uint8_t)(ctx->bitlen >> 48);
    ctx->data[56] = (uint8_t)(ctx->bitlen >> 56);
    sha256_transform(ctx, ctx->data);

    sha256d_store_be32(hash + 0, ctx->state[0]);
    sha256d_store_be32(hash + 4, ctx->state[1]);
    sha256d_store_be32(hash + 8, ctx->state[2]);
    sha256d_store_be32(hash + 12, ctx->state[3]);
    sha256d_store_be32(hash + 16, ctx->state[4]);
    sha256d_store_be32(hash + 20, ctx->state[5]);
    sha256d_store_be32(hash + 24, ctx->state[6]);
    sha256d_store_be32(hash + 28, ctx->state[7]);
}

static void sha256_state_to_bytes(const uint32_t state[8], uint8_t hash[32]) {
    sha256d_store_be32(hash + 0, state[0]);
    sha256d_store_be32(hash + 4, state[1]);
    sha256d_store_be32(hash + 8, state[2]);
    sha256d_store_be32(hash + 12, state[3]);
    sha256d_store_be32(hash + 16, state[4]);
    sha256d_store_be32(hash + 20, state[5]);
    sha256d_store_be32(hash + 24, state[6]);
    sha256d_store_be32(hash + 28, state[7]);
}

static SHA256D_FORCEINLINE int sha256_hash_le_target_be_words(const uint32_t hash_state[8], const uint32_t target_be_words[8]) {
    for (int i = 0; i < 8; ++i) {
        const uint32_t h = hash_state[i];
        const uint32_t t = target_be_words[i];
        if (h < t) {
            return 1;
        }
        if (h > t) {
            return 0;
        }
    }
    return 1;
}

static void sha256d_bytes(const uint8_t *data, size_t len, uint8_t out[32]) {
    uint8_t hash1[32];
    SHA256_CTX ctx;

    sha256_init(&ctx);
    sha256_update(&ctx, data, len);
    sha256_final(&ctx, hash1);

    sha256_init(&ctx);
    sha256_update(&ctx, hash1, 32);
    sha256_final(&ctx, out);
}

static PyObject *py_sha256d(PyObject *self, PyObject *args) {
    Py_buffer view;
    uint8_t hash2[32];

    if (!PyArg_ParseTuple(args, "y*", &view)) {
        return NULL;
    }

    sha256d_bytes((const uint8_t *)view.buf, (size_t)view.len, hash2);

    PyBuffer_Release(&view);
    return PyBytes_FromStringAndSize((const char *)hash2, 32);
}

static PyObject *py_scan_hashes(PyObject *self, PyObject *args) {
    Py_buffer header_prefix;
    Py_buffer target;
    unsigned long long start_nonce = 0;
    unsigned long long count = 0;
    PyObject *results = NULL;
    unsigned long long i = 0;
    uint8_t hash2[32];
    uint32_t midstate[8];
    uint32_t state1[8];
    uint32_t state2[8];
    uint32_t target_words[8];
    uint32_t block1_words[16];
    uint32_t block2_words[16];
    uint32_t hash_words[16];
    const uint8_t *target_bytes = NULL;
    const uint8_t *prefix_bytes = NULL;

    if (!PyArg_ParseTuple(args, "y*KKy*", &header_prefix, &start_nonce, &count, &target)) {
        return NULL;
    }
    if (header_prefix.len != 76) {
        PyBuffer_Release(&header_prefix);
        PyBuffer_Release(&target);
        PyErr_SetString(PyExc_ValueError, "header_prefix must be 76 bytes");
        return NULL;
    }
    if (target.len != 32) {
        PyBuffer_Release(&header_prefix);
        PyBuffer_Release(&target);
        PyErr_SetString(PyExc_ValueError, "target must be 32 bytes");
        return NULL;
    }
    if (start_nonce >= 0x100000000ULL) {
        PyBuffer_Release(&header_prefix);
        PyBuffer_Release(&target);
        PyErr_SetString(PyExc_ValueError, "start_nonce out of range");
        return NULL;
    }
    if (count > 0x100000000ULL - start_nonce) {
        count = 0x100000000ULL - start_nonce;
    }

    prefix_bytes = (const uint8_t *)header_prefix.buf;
    target_bytes = (const uint8_t *)target.buf;
    for (int j = 0; j < 8; ++j) {
        target_words[j] = sha256d_load_be32(target_bytes + (size_t)j * 4U);
    }

    for (int j = 0; j < 16; ++j) {
        block1_words[j] = sha256d_load_be32(prefix_bytes + (size_t)j * 4U);
    }

    block2_words[0] = sha256d_load_be32(prefix_bytes + 64);
    block2_words[1] = sha256d_load_be32(prefix_bytes + 68);
    block2_words[2] = sha256d_load_be32(prefix_bytes + 72);
    block2_words[3] = 0;
    block2_words[4] = 0x80000000U;
    for (int j = 5; j < 15; ++j) {
        block2_words[j] = 0;
    }
    block2_words[15] = 0x00000280U; /* 80 bytes * 8 */

    for (int j = 0; j < 16; ++j) {
        hash_words[j] = 0;
    }
    hash_words[8] = 0x80000000U;
    hash_words[15] = 0x00000100U; /* 32 bytes * 8 */

    sha256_init_state(midstate);
    sha256_compress_fast_words(midstate, block1_words);

    results = PyList_New(0);
    if (!results) {
        PyBuffer_Release(&header_prefix);
        PyBuffer_Release(&target);
        return NULL;
    }

    for (i = 0; i < count; ++i) {
        uint32_t nonce = (uint32_t)(start_nonce + i);
        block2_words[3] = sha256d_bswap32(nonce);

        state1[0] = midstate[0];
        state1[1] = midstate[1];
        state1[2] = midstate[2];
        state1[3] = midstate[3];
        state1[4] = midstate[4];
        state1[5] = midstate[5];
        state1[6] = midstate[6];
        state1[7] = midstate[7];
        sha256_compress_fast_words(state1, block2_words);

        hash_words[0] = state1[0];
        hash_words[1] = state1[1];
        hash_words[2] = state1[2];
        hash_words[3] = state1[3];
        hash_words[4] = state1[4];
        hash_words[5] = state1[5];
        hash_words[6] = state1[6];
        hash_words[7] = state1[7];

        sha256_init_state(state2);
        sha256_compress_fast_words(state2, hash_words);

        if (sha256_hash_le_target_be_words(state2, target_words)) {
            sha256_state_to_bytes(state2, hash2);
            PyObject *py_nonce = PyLong_FromUnsignedLong(nonce);
            PyObject *py_hash = PyBytes_FromStringAndSize((const char *)hash2, 32);
            PyObject *pair = PyTuple_Pack(2, py_nonce, py_hash);
            Py_DECREF(py_nonce);
            Py_DECREF(py_hash);
            if (!pair || PyList_Append(results, pair) != 0) {
                Py_XDECREF(pair);
                Py_DECREF(results);
                results = NULL;
                break;
            }
            Py_DECREF(pair);
        }
    }

    PyBuffer_Release(&header_prefix);
    PyBuffer_Release(&target);
    return results;
}

static PyMethodDef methods[] = {
    {"sha256d", py_sha256d, METH_VARARGS, "Double SHA-256 hash."},
    {"scan_hashes", py_scan_hashes, METH_VARARGS, "Scan nonces for SHA256d hashes below target."},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef module = {
    PyModuleDef_HEAD_INIT,
    "_sha256d",
    "Portable SHA256d backend.",
    -1,
    methods
};

PyMODINIT_FUNC PyInit__sha256d(void) {
    PyObject *mod = PyModule_Create(&module);
    if (!mod) {
        return NULL;
    }
    PyModule_AddStringConstant(mod, "backend", g_backend);
    return mod;
}
