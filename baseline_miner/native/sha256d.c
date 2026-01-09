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

static uint32_t rotr(uint32_t x, uint32_t n) {
    return (x >> n) | (x << (32 - n));
}

static uint32_t ch(uint32_t x, uint32_t y, uint32_t z) {
    return (x & y) ^ (~x & z);
}

static uint32_t maj(uint32_t x, uint32_t y, uint32_t z) {
    return (x & y) ^ (x & z) ^ (y & z);
}

static uint32_t ep0(uint32_t x) {
    return rotr(x, 2) ^ rotr(x, 13) ^ rotr(x, 22);
}

static uint32_t ep1(uint32_t x) {
    return rotr(x, 6) ^ rotr(x, 11) ^ rotr(x, 25);
}

static uint32_t sig0(uint32_t x) {
    return rotr(x, 7) ^ rotr(x, 18) ^ (x >> 3);
}

static uint32_t sig1(uint32_t x) {
    return rotr(x, 17) ^ rotr(x, 19) ^ (x >> 10);
}

static void sha256_init_state(uint32_t state[8]) {
    state[0] = 0x6a09e667;
    state[1] = 0xbb67ae85;
    state[2] = 0x3c6ef372;
    state[3] = 0xa54ff53a;
    state[4] = 0x510e527f;
    state[5] = 0x9b05688c;
    state[6] = 0x1f83d9ab;
    state[7] = 0x5be0cd19;
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
        m[i] = sig1(m[i - 2]) + m[i - 7] + sig0(m[i - 15]) + m[i - 16];
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
        t1 = h + ep1(e) + ch(e, f, g) + k[i] + m[i];
        t2 = ep0(a) + maj(a, b, c);
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

static sha256_block_fn g_compress = sha256_compress_portable;
static const char *g_backend = "portable";

static void sha256_transform(SHA256_CTX *ctx, const uint8_t data[64]) {
    g_compress(ctx->state, data);
}

static void sha256_init(SHA256_CTX *ctx) {
    ctx->datalen = 0;
    ctx->bitlen = 0;
    sha256_init_state(ctx->state);
}

static void sha256_update(SHA256_CTX *ctx, const uint8_t data[], size_t len) {
    for (size_t i = 0; i < len; ++i) {
        ctx->data[ctx->datalen] = data[i];
        ctx->datalen += 1;
        if (ctx->datalen == 64) {
            sha256_transform(ctx, ctx->data);
            ctx->bitlen += 512;
            ctx->datalen = 0;
        }
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

    for (i = 0; i < 4; ++i) {
        hash[i] = (uint8_t)((ctx->state[0] >> (24 - i * 8)) & 0xff);
        hash[i + 4] = (uint8_t)((ctx->state[1] >> (24 - i * 8)) & 0xff);
        hash[i + 8] = (uint8_t)((ctx->state[2] >> (24 - i * 8)) & 0xff);
        hash[i + 12] = (uint8_t)((ctx->state[3] >> (24 - i * 8)) & 0xff);
        hash[i + 16] = (uint8_t)((ctx->state[4] >> (24 - i * 8)) & 0xff);
        hash[i + 20] = (uint8_t)((ctx->state[5] >> (24 - i * 8)) & 0xff);
        hash[i + 24] = (uint8_t)((ctx->state[6] >> (24 - i * 8)) & 0xff);
        hash[i + 28] = (uint8_t)((ctx->state[7] >> (24 - i * 8)) & 0xff);
    }
}

static void sha256_state_to_bytes(const uint32_t state[8], uint8_t hash[32]) {
    for (int i = 0; i < 4; ++i) {
        hash[i] = (uint8_t)((state[0] >> (24 - i * 8)) & 0xff);
        hash[i + 4] = (uint8_t)((state[1] >> (24 - i * 8)) & 0xff);
        hash[i + 8] = (uint8_t)((state[2] >> (24 - i * 8)) & 0xff);
        hash[i + 12] = (uint8_t)((state[3] >> (24 - i * 8)) & 0xff);
        hash[i + 16] = (uint8_t)((state[4] >> (24 - i * 8)) & 0xff);
        hash[i + 20] = (uint8_t)((state[5] >> (24 - i * 8)) & 0xff);
        hash[i + 24] = (uint8_t)((state[6] >> (24 - i * 8)) & 0xff);
        hash[i + 28] = (uint8_t)((state[7] >> (24 - i * 8)) & 0xff);
    }
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
    uint8_t block1[64];
    uint8_t block2[64];
    uint8_t hash1[32];
    uint8_t hash2[32];
    uint8_t hash_block[64];
    uint32_t midstate[8];
    uint32_t state1[8];
    uint32_t state2[8];
    const uint8_t *target_bytes = NULL;
    const uint8_t *prefix_bytes = NULL;
    uint64_t bitlen = 0;

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
    memcpy(block1, prefix_bytes, 64);
    memset(block2, 0, sizeof(block2));
    memcpy(block2, prefix_bytes + 64, 12);
    block2[16] = 0x80;
    bitlen = 80ULL * 8ULL;
    block2[56] = (uint8_t)(bitlen >> 56);
    block2[57] = (uint8_t)(bitlen >> 48);
    block2[58] = (uint8_t)(bitlen >> 40);
    block2[59] = (uint8_t)(bitlen >> 32);
    block2[60] = (uint8_t)(bitlen >> 24);
    block2[61] = (uint8_t)(bitlen >> 16);
    block2[62] = (uint8_t)(bitlen >> 8);
    block2[63] = (uint8_t)(bitlen);

    memset(hash_block, 0, sizeof(hash_block));
    hash_block[32] = 0x80;
    bitlen = 32ULL * 8ULL;
    hash_block[56] = (uint8_t)(bitlen >> 56);
    hash_block[57] = (uint8_t)(bitlen >> 48);
    hash_block[58] = (uint8_t)(bitlen >> 40);
    hash_block[59] = (uint8_t)(bitlen >> 32);
    hash_block[60] = (uint8_t)(bitlen >> 24);
    hash_block[61] = (uint8_t)(bitlen >> 16);
    hash_block[62] = (uint8_t)(bitlen >> 8);
    hash_block[63] = (uint8_t)(bitlen);

    sha256_init_state(midstate);
    g_compress(midstate, block1);

    target_bytes = (const uint8_t *)target.buf;
    results = PyList_New(0);
    if (!results) {
        PyBuffer_Release(&header_prefix);
        PyBuffer_Release(&target);
        return NULL;
    }

    for (i = 0; i < count; ++i) {
        uint32_t nonce = (uint32_t)(start_nonce + i);
        block2[12] = (uint8_t)(nonce & 0xff);
        block2[13] = (uint8_t)((nonce >> 8) & 0xff);
        block2[14] = (uint8_t)((nonce >> 16) & 0xff);
        block2[15] = (uint8_t)((nonce >> 24) & 0xff);

        memcpy(state1, midstate, sizeof(midstate));
        g_compress(state1, block2);
        sha256_state_to_bytes(state1, hash1);

        memcpy(hash_block, hash1, 32);
        sha256_init_state(state2);
        g_compress(state2, hash_block);
        sha256_state_to_bytes(state2, hash2);

        if (memcmp(hash2, target_bytes, 32) <= 0) {
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
