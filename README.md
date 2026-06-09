# ternary-checkpoint

**Ternary model checkpointing with 16× compression and integrity verification.**

Neural networks with ternary weights — where each weight is exactly −1, 0, or +1 — are the backbone of efficient edge inference. But checkpointing them as 32-bit floats wastes 30 out of every 32 bits. Every weight *can only be three things*, yet we store it like it could be anything.

`ternary-checkpoint` exploits this with a simple but powerful insight: **2 bits can represent 4 values, and ternary only needs 3**. So we pack 16 trits into a single `u32`, achieving a 16× compression ratio over `float32` — losslessly. Every `-1`, `0`, and `+1` round-trips perfectly through pack → unpack → verify.

## The Insight: Ternary Weights Are Already Compressed in Meaning

A ternary network doesn't have 4,294,967,296 possible values per weight like a float32. It has **3**. The information content per weight is log₂(3) ≈ 1.585 bits. Storing that in 32 bits is 20× over-provisioned.

The packing scheme uses 2 bits per trit:

| Trit | Binary | Meaning |
|------|--------|---------|
| −1 | `00` | Negative connection |
| 0 | `01` | No connection (pruned) |
| +1 | `10` | Positive connection |
| — | `11` | Invalid (unused) |

This wastes 1 of 4 codes per trit (25% overhead), but keeps packing and unpacking to simple bit shifts — no lookup tables, no branching, just `|=` and `>>`. On a 64-bit system, you can even extend to 32 trits per `u64` for further batching.

## Architecture

```
┌─────────────────────────────────────────────────────┐
│              Ternary Weight Matrix                   │
│  [[-1, 0, 1], [1, -1, 0], [0, 1, -1], ...]         │
│  rows × cols trits                                   │
└─────────────────────┬───────────────────────────────┘
                      │
                      ▼
            ┌─────────────────┐
            │  pack_matrix()  │  Flatten → chunk(16) → pack_trits()
            │  16 trits/u32   │
            └────────┬────────┘
                     │
                     ▼
          ┌──────────────────────┐
          │  Compressed Checkpoint│  Vec<u32>, 16× smaller than f32
          │  [0xABCD1234, ...]   │
          └──────────┬───────────┘
                     │
           ┌─────────┴─────────┐
           ▼                   ▼
   ┌───────────────┐   ┌────────────────┐
   │  Verify       │   │  Unpack        │
   │  Integrity    │   │  Matrix        │
   │  (repack +    │   │  (u32 → trits  │
   │   compare)    │   │   → rows)      │
   └───────────────┘   └────────────────┘
```

## Quick Start

```toml
[dependencies]
ternary-checkpoint = "0.1"
```

```rust
use ternary_checkpoint::*;

fn main() {
    // A small ternary weight matrix
    let weights = vec![
        vec![-1, 0, 1],
        vec![1, -1, 0],
        vec![0, 1, -1],
    ];

    // Pack into compressed form (16 trits per u32)
    let compressed = pack_matrix(&weights);
    println!("Packed {} trits into {} u32s", 9, compressed.len());

    // Verify integrity (lossless round-trip)
    assert!(verify_integrity(
        &[−1, 0, 1, 1, −1, 0, 0, 1, −1],
        &compressed,
    ));

    // Unpack back to matrix
    let restored = unpack_matrix(&compressed, 3, 3);
    assert_eq!(weights, restored);
    println!("Round-trip verified: lossless compression!");
}
```

## Tutorial

### Packing Individual Trits

The fundamental operation packs up to 16 trits into a single `u32`:

```rust
use ternary_checkpoint::*;

let trits = vec![-1, 0, 1, -1, 0, 1, -1, 1, 0, 0, 1, -1, 1, -1, 0, 1];
let packed = pack_trits(&trits);  // Vec<u32> with 1 element

let unpacked = unpack_trits(&packed, 16);  // original trits
assert_eq!(trits, unpacked);
```

Fewer than 16 trits works too — the remaining bits are simply zero:

```rust
let small = vec![-1, 0, 1];
let packed = pack_trits(&small);  // still 1 u32
let restored = unpack_trits(&packed, 3);
assert_eq!(small, restored);
```

### Full Weight Matrix Compression

For a real model layer, use `pack_matrix` and `unpack_matrix`:

```rust
// Simulate a 128×64 ternary weight layer
let rows = 128;
let cols = 64;
let total_params = rows * cols;  // 8,192 trits

let matrix: Vec<Vec<Trit>> = (0..rows)
    .map(|r| (0..cols).map(|c| ((r + c) % 3) as i8 - 1).collect())
    .collect();

let compressed = pack_matrix(&matrix);
let restored = unpack_matrix(&compressed, rows, cols);
assert_eq!(matrix, restored);

// Check compression
let original_bytes = total_params * 4;  // float32
let compressed_bytes = compressed.len() * 4;  // u32
let ratio = compression_ratio(original_bytes, compressed_bytes);
println!("Compression ratio: {:.1}×", ratio);  // ~16.0×
```

### Integrity Verification

After saving a checkpoint to disk and loading it back, verify that no corruption occurred:

```rust
let weights: Vec<Trit> = vec![-1, 0, 1, 1, -1, 0, 0, 1, -1];
let compressed = pack_trits(&weights);

// ... save to disk, load back later ...

let loaded_compressed: Vec<u32> = compressed; // simulate load
if verify_integrity(&weights, &loaded_compressed) {
    println!("✓ Checkpoint integrity verified");
} else {
    println!("✗ Checkpoint corrupted!");
}
```

### Weight Pruning with Top-K

Sparsify a weight vector by keeping only the k highest-magnitude connections:

```rust
let mut weights = vec![1, 0, -1, 0, 1, -1, 0, 1];
keep_top_k(&mut weights, 3);

// Only 3 non-zero weights remain, rest are zeroed
let nonzero: Vec<_> = weights.iter().filter(|&&w| w != 0).collect();
assert_eq!(nonzero.len(), 3);
```

This is useful for fine-grained pruning: zero out the least important connections before checkpointing, making the compressed representation more efficient (since zeros pack to `01` just like non-zeros, but downstream sparse matrix ops can skip them).

### Computing Compression Ratio

```rust
// A layer with 1M ternary weights
let original_bytes = 1_000_000 * 4;       // if stored as float32
let compressed_bytes = (1_000_000 / 16) * 4; // packed: 62,500 u32s
let ratio = compression_ratio(original_bytes, compressed_bytes);
assert!((ratio - 16.0).abs() < 0.01);
```

## API Reference

| Function | Description |
|----------|-------------|
| `pack_trits(trits: &[Trit]) -> Vec<u32>` | Pack ≤16 trits into one `u32` |
| `unpack_trits(packed: &[u32], count: usize) -> Vec<Trit>` | Unpack `count` trits from compressed form |
| `pack_matrix(matrix: &[Vec<Trit>]) -> Vec<u32>` | Flatten and pack an entire weight matrix |
| `unpack_matrix(packed: &[u32], rows: usize, cols: usize) -> Vec<Vec<Trit>>` | Unpack back to a row-major matrix |
| `verify_integrity(weights: &[Trit], compressed: &[u32]) -> bool` | Lossless round-trip verification |
| `compression_ratio(original: usize, compressed: usize) -> f64` | Compression ratio (higher = better) |
| `keep_top_k(weights: &mut [Trit], k: usize)` | Zero out all but the k highest-magnitude weights |

| Type | Description |
|------|-------------|
| `Trit` | Alias for `i8`: must be −1, 0, or +1 |

## Ecosystem Role

`ternary-checkpoint` is the **model serialization layer** in the SuperInstance ecosystem:

- **Input:** Ternary weight matrices from training (weights constrained to {-1, 0, +1})
- **Output:** Compact `Vec<u32>` suitable for disk storage, network transfer, or memory-mapped loading
- **Depends on:** [`ternary-types`](https://github.com/SuperInstance/ternary-types) for shared type definitions
- **Complementary to:** [`constraint-schedule`](https://github.com/SuperInstance/constraint-schedule) for scheduling distributed training checkpoints, and [`topo-merge`](https://github.com/SuperInstance/topo-merge) for merging distributed model updates

In a SuperInstance deployment, models are trained with ternary constraints, checkpointed with this crate, transferred between nodes at 16× bandwidth savings, and verified on load — all losslessly.

## Performance

| Operation | Complexity | Notes |
|-----------|-----------|-------|
| `pack_trits` | O(16) | Bit shift + OR per trit |
| `unpack_trits` | O(16) per u32 | Shift + mask per trit |
| `pack_matrix` | O(n) | n = total trits, auto-chunked |
| `unpack_matrix` | O(n) | Symmetric with pack |
| `verify_integrity` | O(n) | Unpack + compare |
| `keep_top_k` | O(n log n) | Sort by magnitude |

No heap allocation in the hot path except the output `Vec`. No external dependencies beyond `ternary-types`.

## License

MIT
