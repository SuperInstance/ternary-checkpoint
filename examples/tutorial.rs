//! A guided walkthrough of ternary checkpoint compression.
//!
//! Run with: cargo run --example tutorial

use ternary_checkpoint::*;

fn main() {
    println!("=== ternary-checkpoint Tutorial ===\n");

    // --- Step 1: Basic trit packing ---
    println!("Step 1: Packing individual trits\n");

    let trits = vec![-1_i8, 0, 1, -1, 0, 1, -1, 1, 0, 0, 1, -1, 1, -1, 0, 1];
    println!("  Input: {} trits", trits.len());

    let packed = pack_trits(&trits);
    println!("  Packed into {} u32(s)", packed.len());
    println!("  Binary: {:032b}\n", packed[0]);

    let unpacked = unpack_trits(&packed, trits.len());
    assert_eq!(trits, unpacked);
    println!("  Unpacked: {:?}", unpacked);
    println!("  ✓ Lossless round-trip!\n");

    // --- Step 2: Partial packing (fewer than 16 trits) ---
    println!("Step 2: Partial packing\n");

    let small = vec![-1_i8, 0, 1];
    let packed_small = pack_trits(&small);
    println!("  {} trits → {} u32", small.len(), packed_small.len());

    let restored = unpack_trits(&packed_small, 3);
    assert_eq!(small, restored);
    println!("  Restored: {:?}\n", restored);

    // --- Step 3: Matrix packing ---
    println!("Step 3: Full weight matrix compression\n");

    let weights = vec![
        vec![-1_i8, 0, 1, -1],
        vec![1_i8, -1, 0, 1],
        vec![0_i8, 1, -1, 0],
    ];
    println!("  Matrix: {}×{} = {} trits", weights.len(), weights[0].len(), 12);

    let compressed = pack_matrix(&weights);
    println!("  Compressed into {} u32(s)", compressed.len());

    let restored_matrix = unpack_matrix(&compressed, 3, 4);
    assert_eq!(weights, restored_matrix);
    println!("  ✓ Matrix round-trip verified!\n");

    // --- Step 4: Integrity verification ---
    println!("Step 4: Integrity verification\n");

    let flat_weights: Vec<Trit> = vec![-1, 0, 1, 1, -1, 0, 0, 1, -1];
    let compressed_weights = pack_trits(&flat_weights);

    if verify_integrity(&flat_weights, &compressed_weights) {
        println!("  ✓ Checkpoint integrity verified\n");
    }

    // Simulate corruption
    let mut corrupted = compressed_weights.clone();
    corrupted[0] ^= 1; // flip one bit
    if !verify_integrity(&flat_weights, &corrupted) {
        println!("  ✗ Corrupted checkpoint detected!\n");
    }

    // --- Step 5: Compression ratio ---
    println!("Step 5: Compression ratios\n");

    let scenarios = [
        ("Tiny layer", 64, 64),
        ("Medium layer", 256, 128),
        ("Large layer", 1024, 512),
        ("Embedding", 10000, 256),
    ];

    for (name, rows, cols) in scenarios {
        let total_trits = rows * cols;
        let original_bytes = total_trits * 4; // float32
        let compressed_bytes = ((total_trits + 15) / 16) * 4; // packed u32s
        let ratio = compression_ratio(original_bytes, compressed_bytes);
        println!(
            "  {}: {}×{} ({} trits) → {:.1}× compression",
            name, rows, cols, total_trits, ratio
        );
    }
    println!();

    // --- Step 6: Top-K pruning ---
    println!("Step 6: Weight pruning with top-K\n");

    let mut weights = vec![1_i8, 0, -1, 0, 1, -1, 0, 1];
    println!("  Before: {:?}", weights);
    keep_top_k(&mut weights, 3);
    println!("  After keep_top_k(3): {:?}", weights);

    let nonzero: Vec<_> = weights.iter().filter(|&&w| w != 0).collect();
    println!("  Non-zero: {} weights\n", nonzero.len());

    // --- Step 7: Simulate a model checkpoint ---
    println!("Step 7: Full model checkpoint simulation\n");

    let layers = [
        ("input_hidden", 784, 128),
        ("hidden_1", 128, 64),
        ("hidden_2", 64, 32),
        ("output", 32, 10),
    ];

    let mut total_original = 0usize;
    let mut total_compressed = 0usize;

    for (name, rows, cols) in layers {
        // Generate deterministic ternary weights
        let matrix: Vec<Vec<Trit>> = (0..rows)
            .map(|r| {
                (0..cols)
                    .map(|c| (((r * 7 + c * 13) % 3) as i8) - 1)
                    .collect()
            })
            .collect();

        let compressed = pack_matrix(&matrix);
        let restored = unpack_matrix(&compressed, rows, cols);
        assert_eq!(matrix, restored);

        let orig_bytes = rows * cols * 4;
        let comp_bytes = compressed.len() * 4;
        total_original += orig_bytes;
        total_compressed += comp_bytes;

        println!(
            "  {}: {} params → {} bytes (from {} bytes)",
            name,
            rows * cols,
            comp_bytes,
            orig_bytes
        );
    }

    let ratio = compression_ratio(total_original, total_compressed);
    println!(
        "\n  Total: {} bytes → {} bytes ({:.1}× compression)",
        total_original, total_compressed, ratio
    );
    println!("  ✓ All layers verified lossless!\n");

    println!("=== Tutorial complete! ===");
}
