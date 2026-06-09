//! Advanced usage: checkpointing a multi-layer ternary network with verification.
//!
//! Demonstrates saving and loading a realistic ternary model with multiple
//! layers, integrity checks, and compression analysis.
//!
//! Run with: cargo run --example advanced

use ternary_checkpoint::*;

fn main() {
    println!("=== Advanced: Multi-Layer Ternary Checkpointing ===\n");

    // Simulate a ternary BERT-like model for edge deployment
    // Model: embedding + 4 transformer layers + classification head

    println!("Model: TernaryMiniBERT\n");

    let layers: Vec<(&str, usize, usize)> = vec![
        ("token_embedding", 30522, 128),   // vocabulary × hidden
        ("position_embedding", 512, 128),  // max_seq × hidden
        ("layer_0_attention_q", 128, 128),
        ("layer_0_attention_k", 128, 128),
        ("layer_0_attention_v", 128, 128),
        ("layer_0_ffn_1", 128, 512),       // expansion
        ("layer_0_ffn_2", 512, 128),       // projection back
        ("layer_1_attention_q", 128, 128),
        ("layer_1_attention_k", 128, 128),
        ("layer_1_attention_v", 128, 128),
        ("layer_1_ffn_1", 128, 512),
        ("layer_1_ffn_2", 512, 128),
        ("classifier", 128, 2),            // binary classification head
    ];

    // Generate and checkpoint all layers
    struct CheckpointLayer {
        name: String,
        rows: usize,
        cols: usize,
        original_weights: Vec<Vec<Trit>>,
        compressed: Vec<u32>,
    }

    let mut checkpoints: Vec<CheckpointLayer> = Vec::new();
    let mut total_original_bytes = 0usize;
    let mut total_compressed_bytes = 0usize;

    println!("Checkpointing {} layers:\n", layers.len());

    for (name, rows, cols) in &layers {
        // Generate pseudo-random ternary weights
        let matrix: Vec<Vec<Trit>> = (0..*rows)
            .map(|r| {
                (0..*cols)
                    .map(|c| {
                        // Deterministic hash-like generation
                        let v = (r.wrapping_mul(2654435761) ^ c.wrapping_mul(2246822519)) % 3;
                        (v as i8) - 1
                    })
                    .collect()
            })
            .collect();

        let compressed = pack_matrix(&matrix);

        let orig_bytes = rows * cols * 4; // float32 baseline
        let comp_bytes = compressed.len() * 4;
        total_original_bytes += orig_bytes;
        total_compressed_bytes += comp_bytes;

        checkpoints.push(CheckpointLayer {
            name: name.to_string(),
            rows: *rows,
            cols: *cols,
            original_weights: matrix,
            compressed,
        });

        println!(
            "  {:<25} {:>8} params  {:>10} → {:>8} bytes",
            format!("{}:", name),
            rows * cols,
            orig_bytes,
            comp_bytes,
        );
    }

    println!();

    // Total compression stats
    let ratio = compression_ratio(total_original_bytes, total_compressed_bytes);
    println!("--- Compression Summary ---\n");
    println!(
        "  Original (float32):  {} bytes ({:.1} MB)",
        total_original_bytes,
        total_original_bytes as f64 / 1_048_576.0
    );
    println!(
        "  Compressed (packed): {} bytes ({:.1} MB)",
        total_compressed_bytes,
        total_compressed_bytes as f64 / 1_048_576.0
    );
    println!("  Compression ratio:   {:.1}×", ratio);
    println!(
        "  Space saved:         {} bytes ({:.1} MB)\n",
        total_original_bytes - total_compressed_bytes,
        (total_original_bytes - total_compressed_bytes) as f64 / 1_048_576.0,
    );

    // Simulate save → load → verify cycle
    println!("--- Simulating Save/Load Cycle ---\n");

    // "Save" — in reality, you'd write compressed to disk
    let saved: Vec<(&str, Vec<u32>, usize, usize)> = checkpoints
        .iter()
        .map(|c| (c.name.as_str(), c.compressed.clone(), c.rows, c.cols))
        .collect();

    // "Load" — read from disk and verify
    let mut all_verified = true;
    for checkpoint in &checkpoints {
        let saved_layer = saved.iter().find(|(n, _, _, _)| *n == checkpoint.name).unwrap();

        // Unpack and verify
        let restored = unpack_matrix(&saved_layer.1, checkpoint.rows, checkpoint.cols);

        // Flatten for integrity check
        let original_flat: Vec<Trit> = checkpoint
            .original_weights
            .iter()
            .flat_map(|r| r.iter().copied())
            .collect();

        let verified = verify_integrity(&original_flat, &saved_layer.1);
        let matrix_match = checkpoint.original_weights == restored;

        if !verified || !matrix_match {
            println!("  ✗ {} — integrity check FAILED!", checkpoint.name);
            all_verified = false;
        }
    }

    if all_verified {
        println!("  ✓ All {} layers verified — lossless round-trip confirmed", checkpoints.len());
    }
    println!();

    // Analyze weight distribution
    println!("--- Weight Distribution Analysis ---\n");
    for checkpoint in &checkpoints {
        let total: usize = checkpoint.original_weights.iter().map(|r| r.len()).sum();
        let neg: usize = checkpoint
            .original_weights
            .iter()
            .flat_map(|r| r.iter())
            .filter(|&&w| w == -1)
            .count();
        let zero: usize = checkpoint
            .original_weights
            .iter()
            .flat_map(|r| r.iter())
            .filter(|&&w| w == 0)
            .count();
        let pos: usize = checkpoint
            .original_weights
            .iter()
            .flat_map(|r| r.iter())
            .filter(|&&w| w == 1)
            .count();

        println!(
            "  {:<25} -1:{:>5} ({:>4.1}%)  0:{:>5} ({:>4.1}%)  +1:{:>5} ({:>4.1}%)",
            format!("{}:", checkpoint.name),
            neg,
            neg as f64 / total as f64 * 100.0,
            zero,
            zero as f64 / total as f64 * 100.0,
            pos,
            pos as f64 / total as f64 * 100.0,
        );
    }
    println!();

    // Pruning experiment: what if we keep only top-50% weights?
    println!("--- Pruning Experiment (top-50%) ---\n");

    for checkpoint in &checkpoints {
        let mut weights: Vec<Trit> = checkpoint
            .original_weights
            .iter()
            .flat_map(|r| r.iter().copied())
            .collect();

        let total = weights.len();
        let k = total / 2;
        let pre_nonzero: usize = weights.iter().filter(|&&w| w != 0).count();

        keep_top_k(&mut weights, k);

        let post_nonzero: usize = weights.iter().filter(|&&w| w != 0).count();
        let compression_hint = total as f64 / (total - post_nonzero).max(1) as f64;

        println!(
            "  {:<25} {} → {} non-zero ({:.0}% sparsity, ~{:.1}× sparse speedup)",
            format!("{}:", checkpoint.name),
            pre_nonzero,
            post_nonzero,
            (1.0 - post_nonzero as f64 / total as f64) * 100.0,
            compression_hint,
        );
    }
    println!();

    println!("=== Advanced demo complete! ===");
}
