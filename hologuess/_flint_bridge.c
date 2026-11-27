/* The public bridge ABI uses fixed-width integers, independent of FLINT's word size. */
#include <stddef.h>
#include <stdint.h>
#include <flint/nmod_mat.h>
#include <flint/ulong_extras.h>

uint32_t hologuess_flint_abi(void)
{
    return UINT32_C(1);
}

const char *hologuess_flint_header_version(void)
{
    return FLINT_VERSION;
}

const char *hologuess_flint_runtime_version(void)
{
    return flint_version;
}

int hologuess_flint_rref(const int64_t *input, int64_t *output,
                        uint8_t *pivot_mask, uint64_t count,
                        uint64_t rows, uint64_t cols, uint32_t p)
{
    if (p < 2 || p >= (UINT32_C(1) << 31) || !n_is_prime(p))
        return 1;

    /* Validate dimensions for both the FLINT word type and the input buffers. */
    const uint64_t word_max = (UINT64_C(1) << (FLINT_BITS - 1)) - 1;
    if (rows > word_max || cols > word_max || count > SIZE_MAX)
        return 2;
    if (rows != 0 && cols > word_max / rows)
        return 2;
    if (cols && count > SIZE_MAX / cols)
        return 2;
    if (rows && cols && count > (SIZE_MAX / sizeof(int64_t)) / (rows * cols))
        return 2;
    if (!count || !rows || !cols)
        return 0;

    /* Lower the calling thread's budget without resizing the global FLINT pool. */
    const int limit_workers = flint_get_num_threads() > 1;
    const int old_workers = limit_workers ? flint_set_num_workers(0) : 0;
    nmod_mat_t matrix;
    nmod_mat_init(matrix, (slong) rows, (slong) cols, p);
    for (uint64_t k = 0; k < count; ++k) {
        for (slong i = 0; i < (slong) rows; ++i) {
            for (slong j = 0; j < (slong) cols; ++j) {
                int64_t value = *input++;
                if (value < 0 || (uint64_t) value >= p) {
                    value %= (int64_t) p;
                    if (value < 0)
                        value += p;
                }
                nmod_mat_entry(matrix, i, j) = (ulong) value;
            }
        }
        const slong rank = nmod_mat_rref(matrix);
        for (slong i = 0; i < (slong) rows; ++i) {
            int have_pivot = 0;
            for (slong j = 0; j < (slong) cols; ++j) {
                const ulong value = nmod_mat_entry(matrix, i, j);
                *output++ = (int64_t) value;
                if (i < rank && !have_pivot && value != 0) {
                    pivot_mask[j] = 1;
                    have_pivot = 1;
                }
            }
        }
        pivot_mask += cols;
    }
    nmod_mat_clear(matrix);
    if (limit_workers)
        flint_reset_num_workers(old_workers);
    return 0;
}
