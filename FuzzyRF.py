"""
Fuzzy Random Forest Classifier
@author jonnyhuck

Notes on this version: 
    - GPU (CUDA/CuPy) required — exits if not available
    - Confusion matrix power posterior adjustment is commented out (in _compute_beta_params_chunked)
    - Params are: log(a), log(b) saved as float16 in Zarr.
"""
import math
import zarr
import cupy as cp
import numpy as np
from time import perf_counter
from sklearn.metrics import confusion_matrix
from zarr.codecs import BloscCodec, BytesCodec
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


# ---------------------------------------------------------------------------
# GPU path — CuPy RawKernel (required)
# ---------------------------------------------------------------------------

_CUDA_KERNEL = r"""
extern "C" __global__
void beta_ppf_f32(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float u,
    float* __restrict__ out,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float ai = a[i], bi = b[i];

    auto lgammaf_d = [](float z) -> float {
        const float c[9] = {
            0.99999999999980993f, 676.520368f, -1259.139217f,
            771.323429f, -176.615029f, 12.507343f,
            -0.138571095f, 9.984369e-6f, 1.505633e-7f
        };
        z -= 1.0f;
        float x = c[0];
        for (int k = 1; k < 9; k++) x += c[k] / (z + k);
        float t = z + 7.5f;
        return 0.5f * logf(6.2831853f) + (z + 0.5f) * logf(t) - t + logf(x);
    };

    float lbab = lgammaf_d(ai) + lgammaf_d(bi) - lgammaf_d(ai + bi);

    auto ibetaf = [&](float x) -> float {
        if (x <= 0.0f) return 0.0f;
        if (x >= 1.0f) return 1.0f;
        float a_ = ai, b_ = bi;
        bool flipped = false;
        if (x > (a_ + 1.0f) / (a_ + b_ + 2.0f)) {
            x = 1.0f - x;
            float tmp = a_; a_ = b_; b_ = tmp;
            flipped = true;
        }
        const float T = 1e-20f;
        float front = expf(logf(x) * a_ + logf(1.0f - x) * b_ - lbab) / a_;
        float f = T, C = T;
        float D = 1.0f - (a_ + b_) * x / (a_ + 1.0f);
        if (fabsf(D) < T) D = T;
        D = 1.0f / D; f = C * D;
        for (int m = 1; m <= 60; m++) {
            float num, delta;
            num = (float)m * (b_ - m) * x / ((a_ + 2*m - 1.0f) * (a_ + 2*m));
            D = 1.0f + num * D; C = 1.0f + num / C;
            if (fabsf(D) < T) D = T; if (fabsf(C) < T) C = T;
            D = 1.0f / D; delta = C * D; f *= delta;
            num = -((a_ + m) * (a_ + b_ + m) * x) / ((a_ + 2*m) * (a_ + 2*m + 1.0f));
            D = 1.0f + num * D; C = 1.0f + num / C;
            if (fabsf(D) < T) D = T; if (fabsf(C) < T) C = T;
            D = 1.0f / D; delta = C * D; f *= delta;
            if (fabsf(delta - 1.0f) < 1e-5f) break;
        }
        float r = front * f;
        return flipped ? 1.0f - r : r;
    };

    float mean = ai / (ai + bi);
    float var  = ai * bi / ((ai + bi) * (ai + bi) * (ai + bi + 1.0f));
    float sd   = sqrtf(fmaxf(var, 1e-12f));
    float t    = (u < 0.5f) ? u : 1.0f - u;
    t = sqrtf(-2.0f * logf(fmaxf(t, 1e-30f)));
    float z = t - (2.515517f + 0.802853f * t + 0.010328f * t * t)
                / (1.0f + 1.432788f * t + 0.189269f * t * t + 0.001308f * t * t * t);
    if (u < 0.5f) z = -z;
    float x = fmaxf(1e-5f, fminf(1.0f - 1e-5f, mean + sd * z));

    for (int iter = 0; iter < 8; iter++) {
        float fx  = ibetaf(x) - u;
        float dfx = expf((ai - 1.0f) * logf(fmaxf(x,        1e-30f))
                       + (bi - 1.0f) * logf(fmaxf(1.0f - x, 1e-30f)) - lbab);
        if (dfx < 1e-30f) break;
        float dx = fx / dfx;
        x = fmaxf(1e-6f, fminf(1.0f - 1e-6f, x - dx));
        if (fabsf(dx) < 1e-5f * x) break;
    }
    out[i] = x;
}
"""

_EXP_KERNEL = r"""
__device__ float half_to_float(unsigned short h) {
    unsigned int sign     = (h >> 15) & 0x1;
    unsigned int exponent = (h >> 10) & 0x1f;
    unsigned int mantissa =  h        & 0x3ff;
    unsigned int f;
    if (exponent == 0) {
        f = sign << 31;
    } else if (exponent == 31) {
        f = (sign << 31) | 0x7f800000 | (mantissa << 13);
    } else {
        f = (sign << 31) | ((exponent + 112) << 23) | (mantissa << 13);
    }
    union { unsigned int i; float f; } u;
    u.i = f;
    return u.f;
}

extern "C" __global__
void exp_f16_to_f32(const unsigned short* __restrict__ log_x,
                    float* __restrict__ out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = expf(half_to_float(log_x[i]));
}
"""

try:

    _gpu_kernel  = cp.RawKernel(_CUDA_KERNEL, 'beta_ppf_f32',   backend='nvrtc')
    _exp_kernel  = cp.RawKernel(_EXP_KERNEL,  'exp_f16_to_f32', backend='nvrtc')
    _GPU_THREADS = 256

    _wa = cp.ones(64, dtype=cp.float32) * 2.0
    _wb = cp.ones(64, dtype=cp.float32) * 3.0
    _wo = cp.empty(64, dtype=cp.float32)
    _gpu_kernel((1,), (_GPU_THREADS,), (_wa, _wb, np.float32(0.5), _wo, 64))
    _wh = cp.zeros(64, dtype=cp.uint16)
    _exp_kernel((1,), (_GPU_THREADS,), (_wh, _wa, 64))
    cp.cuda.Stream.null.synchronize()
    del _wa, _wb, _wo, _wh

    print("FuzzyRF: float32 CUDA kernel ready — GPU path active")

except (ImportError, Exception) as _e:
    print(f"FuzzyRF: GPU not available ({_e}) — exiting")
    exit(1)


# ---------------------------------------------------------------------------
# TRAINER
# ---------------------------------------------------------------------------

class FuzzyRFTrainer:

    def __init__(self, input_data, trees=20, branches=8, chunk_size=1_000_000):
        self.trees      = trees
        self.branches   = branches
        self.chunk_size = chunk_size
        self.rng        = np.random.default_rng()

        stacked_array, X, y = input_data
        stacked_array = stacked_array.astype(np.float32)
        self.rows, self.cols, self.n_bands = stacked_array.shape

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42)

        clf = RandomForestClassifier(n_estimators=self.trees,
                                     max_depth=self.branches,
                                     random_state=42, n_jobs=-1)
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)

        # cm[i,j] = P(predicted=j | true=i), rows sum to 1
        # diagonal cm[k,k] = recall for class k
        # (retained for when CM adjustment is re-enabled)
        self._cm = confusion_matrix(y_test, y_pred, normalize="true").astype(np.float32)

        print("Computing Beta parameters in chunks...")
        self.a, self.b = self._compute_beta_params_chunked(clf, stacked_array)
        self.n_classes  = clf.n_classes_
        del clf, stacked_array, X, y, X_train, X_test, y_train, y_test

    def _compute_beta_params_chunked(self, clf, stacked_array):
        n_samples = self.rows * self.cols
        n_classes = clf.n_classes_
        n_trees   = len(clf.estimators_)
        a_full    = np.zeros((n_classes, n_samples), dtype=np.float32)
        b_full    = np.zeros((n_classes, n_samples), dtype=np.float32)
        X_img     = stacked_array.reshape(n_samples, self.n_bands)

        for start in range(0, n_samples, self.chunk_size):
            end       = min(start + self.chunk_size, n_samples)
            print(f"  pixels {start:,} – {end:,} of {n_samples:,}")
            X_chunk   = X_img[start:end]
            chunk_len = end - start

            mean = np.zeros((chunk_len, n_classes), dtype=np.float32)
            M2   = np.zeros_like(mean)

            for t, tree in enumerate(clf.estimators_):

                P = tree.predict_proba(X_chunk).astype(np.float32)

                # ----------------------------------------------------------------
                # POWER POSTERIOR CM ADJUSTMENT — disabled for baseline test
                # To restore: uncomment the block below
                #
                # ec = np.argmax(P, axis=1)
                # Q  = self._cm[ec]
                # pixel_idx  = np.arange(chunk_len)
                # confidence = P[pixel_idx, ec]
                # recall     = Q[pixel_idx, ec]
                # alpha = np.clip(
                #     (confidence - recall) / np.clip(confidence, 1e-12, None),
                #     0.0, 1.0
                # ).astype(np.float32)
                # tempered = np.clip(Q, 1e-12, 1.0) ** alpha[:, None]
                # P       *= tempered
                # row_sums = P.sum(axis=1, keepdims=True)
                # P[row_sums[:, 0] == 0] = 1.0 / n_classes
                # P /= P.sum(axis=1, keepdims=True)
                # ----------------------------------------------------------------

                # Welford online update
                delta  = P - mean
                mean  += delta / (t + 1)
                M2    += delta * (P - mean)

            var = M2 / max(n_trees - 1, 1)
            a_chunk, b_chunk = self._method_of_moments(mean, var)
            a_full[:, start:end] = a_chunk.T
            b_full[:, start:end] = b_chunk.T

        return (a_full.reshape(n_classes, self.rows, self.cols),
                b_full.reshape(n_classes, self.rows, self.cols))

    @staticmethod
    def _method_of_moments(mean, var):
        eps     = np.float32(1e-6)
        mean    = np.clip(mean, eps, 1.0 - eps)
        max_var = mean * (1.0 - mean) - eps
        var     = np.clip(var, eps, max_var)
        common  = np.maximum(mean * (1.0 - mean) / var - 1.0, eps)
        return (mean * common).astype(np.float32), ((1.0 - mean) * common).astype(np.float32)

    def save(self, zarr_path: str):
        codecs      = [BytesCodec(), BloscCodec(cname="lz4", clevel=5, shuffle="bitshuffle")]
        chunk_shape = (1, 1, min(256, self.rows), min(256, self.cols))

        log_a = np.log(np.clip(self.a, 1e-6, None)).astype(np.float16)
        log_b = np.log(np.clip(self.b, 1e-6, None)).astype(np.float16)
        log_a = np.clip(log_a, np.finfo(np.float16).min, np.finfo(np.float16).max)
        log_b = np.clip(log_b, np.finfo(np.float16).min, np.finfo(np.float16).max)

        store = zarr.open(zarr_path, mode='w',
                          shape=(2, self.n_classes, self.rows, self.cols),
                          chunks=chunk_shape, dtype=np.float16, codecs=codecs)
        store[0] = log_a
        store[1] = log_b
        store.attrs.update({
            'n_classes': int(self.n_classes),
            'rows':      int(self.rows),
            'cols':      int(self.cols),
            'trees':     int(self.trees),
            'branches':  int(self.branches),
            'format':    'log_float16',
        })
        print(f"Saved Beta parameters to {zarr_path}")
        print(f"  Shape: {store.shape}, dtype: {store.dtype}")


# ---------------------------------------------------------------------------
# GENERATOR
# ---------------------------------------------------------------------------

class FuzzyRFGenerator:

    def __init__(self, zarr_path: str):
        self.zarr_path = zarr_path
        self.rng = np.random.default_rng()

        store = zarr.open(zarr_path, mode='r')
        self.n_classes = int(store.attrs['n_classes'])
        self.rows      = int(store.attrs['rows'])
        self.cols      = int(store.attrs['cols'])

        fmt = store.attrs.get('format', 'float32')
        if fmt != 'log_float16':
            print(f"Warning: Zarr store format is '{fmt}', expected 'log_float16'. "
                  f"Retrain with this version of FuzzyRFTrainer to use float16 storage.")

    @classmethod
    def from_trainer(cls, trainer: FuzzyRFTrainer, zarr_path: str):
        trainer.save(zarr_path)
        return cls(zarr_path)

    def get_params(self):
        store = zarr.open(self.zarr_path, mode='r')
        return (np.exp(store[0][:].astype(np.float32)),
                np.exp(store[1][:].astype(np.float32)))

    def mc_draws(self, n_draws: int, vram_gb: float = 12.0):
        """
        Generator: yields (n_classes, rows, cols) float32 landscapes in order.
        Each landscape is a coherent Monte Carlo realisation drawn at a single
        quantile u ~ Uniform(0,1) shared across all pixels (CRN).
        """
        shape    = (self.n_classes, self.rows, self.cols)
        u_values = self.rng.random(n_draws)

        print("mc_draws: GPU path (float16 upload + float32 CUDA kernel)")
        print("Loading log(a)/log(b) from Zarr as float16...")
        t0    = perf_counter()
        store = zarr.open(self.zarr_path, mode='r')
        log_a = store[0][:]
        log_b = store[1][:]
        print(f"  loaded in {perf_counter()-t0:.1f}s  "
              f"({log_a.nbytes/1e9:.1f} GB + {log_b.nbytes/1e9:.1f} GB)")

        # chunk size: peak VRAM = 2*f16 + 3*f32 = 16 bytes per element
        n_elem_per_row = self.n_classes * self.cols
        rows_per_chunk = max(1, int(vram_gb * 1024**3 / (16 * n_elem_per_row)))
        n_chunks       = math.ceil(self.rows / rows_per_chunk)
        print(f"  VRAM: {n_chunks} chunk(s) × {rows_per_chunk} rows")

        # precompute uint16 views of float16 data once — reused every draw
        cpu_chunks = []
        for row_start in range(0, self.rows, rows_per_chunk):
            row_end = min(row_start + rows_per_chunk, self.rows)
            cpu_chunks.append((
                np.ascontiguousarray(
                    log_a[:, row_start:row_end, :].ravel()).view(np.uint16),
                np.ascontiguousarray(
                    log_b[:, row_start:row_end, :].ravel()).view(np.uint16),
                row_start, row_end
            ))
        del log_a, log_b

        print(f"Drawing {n_draws} landscapes ({n_chunks} chunk(s) per draw)...")

        for i in range(n_draws):
            t0     = perf_counter()
            sample = np.empty(shape, dtype=np.float32)
            u_f32  = np.float32(u_values[i])

            for log_a_chunk, log_b_chunk, row_start, row_end in cpu_chunks:
                n_pix  = self.n_classes * (row_end - row_start) * self.cols
                blocks = math.ceil(n_pix / _GPU_THREADS)

                la_gpu = cp.asarray(log_a_chunk)
                lb_gpu = cp.asarray(log_b_chunk)

                a_gpu = cp.empty(n_pix, dtype=cp.float32)
                b_gpu = cp.empty(n_pix, dtype=cp.float32)
                _exp_kernel((blocks,), (_GPU_THREADS,), (la_gpu, a_gpu, n_pix))
                _exp_kernel((blocks,), (_GPU_THREADS,), (lb_gpu, b_gpu, n_pix))
                del la_gpu, lb_gpu

                o_gpu = cp.empty(n_pix, dtype=cp.float32)
                _gpu_kernel(
                    (blocks,), (_GPU_THREADS,),
                    (a_gpu, b_gpu, u_f32, o_gpu, n_pix)
                )
                del a_gpu, b_gpu

                out   = cp.reshape(o_gpu, (self.n_classes, row_end - row_start, self.cols))
                total = out.sum(axis=0, keepdims=True)
                total = cp.where(total == 0, cp.float32(1.0), total)
                out  /= total

                sample[:, row_start:row_end, :] = cp.asnumpy(out)
                del o_gpu, out, total
                cp.get_default_memory_pool().free_all_blocks()

            cp.cuda.Stream.null.synchronize()
            print(f"  draw {i+1}/{n_draws} completed in {perf_counter()-t0:.1f}s")
            yield sample

        del cpu_chunks
        cp.get_default_memory_pool().free_all_blocks()


# ---------------------------------------------------------------------------
# Backwards-compatible alias
# ---------------------------------------------------------------------------

FuzzyRF = FuzzyRFTrainer

if __name__ == "__main__":
    pass