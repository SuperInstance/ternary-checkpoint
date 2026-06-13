use crate::pack::extract_trit;

/// Unpack packed u32 data back to f32 values, applying optional scaling factors.
///
/// If no scaling factors are provided, outputs are -1.0, 0.0, or +1.0.
/// With scaling, each value is multiplied by its corresponding scale factor.
pub fn unpack(packed: &[u32], num_trits: usize) -> Vec<f32> {
    unpack_with_scales(packed, num_trits, None)
}

/// Unpack with per-tensor scaling (single scale factor applied to all values).
pub fn unpack_with_scale(packed: &[u32], num_trits: usize, scale: f32) -> Vec<f32> {
    let mut result = Vec::with_capacity(num_trits);
    for i in 0..num_trits {
        let word_idx = i / 16;
        let bit_pos = i % 16;
        let trit = extract_trit(packed[word_idx], bit_pos);
        result.push(trit as f32 * scale);
    }
    result
}

/// Unpack with optional per-channel scaling factors.
///
/// If `per_channel_scales` is Some, each trit is multiplied by its corresponding
/// scale factor. If None, raw ternary values (-1.0, 0.0, +1.0) are returned.
pub fn unpack_with_scales(
    packed: &[u32],
    num_trits: usize,
    per_channel_scales: Option<&[f32]>,
) -> Vec<f32> {
    let mut result = Vec::with_capacity(num_trits);

    for i in 0..num_trits {
        let word_idx = i / 16;
        let bit_pos = i % 16;
        let trit = extract_trit(packed[word_idx], bit_pos);
        let scale = per_channel_scales
            .map(|s| s.get(i).copied().unwrap_or(1.0))
            .unwrap_or(1.0);
        result.push(trit as f32 * scale);
    }

    result
}

/// Unpack packed data to raw i8 ternary values (-1, 0, +1).
pub fn unpack_to_ternary(packed: &[u32], num_trits: usize) -> Vec<i8> {
    let mut result = Vec::with_capacity(num_trits);
    for i in 0..num_trits {
        let word_idx = i / 16;
        let bit_pos = i % 16;
        result.push(extract_trit(packed[word_idx], bit_pos));
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pack::pack;

    #[test]
    fn test_unpack_empty() {
        let result = unpack(&[], 0);
        assert!(result.is_empty());
    }

    #[test]
    fn test_round_trip_all_ones() {
        let weights: Vec<f32> = vec![1.0; 32];
        let packed = pack(&weights);
        let unpacked = unpack(&packed, weights.len());
        for (i, &v) in unpacked.iter().enumerate() {
            assert!((v - 1.0f32).abs() < 1e-6, "mismatch at {}: got {}", i, v);
        }
    }

    #[test]
    fn test_round_trip_mixed() {
        let weights: Vec<f32> = vec![1.0, -1.0, 0.05, -0.5, 2.0, -3.0, 0.01, 0.1];
        let packed = pack(&weights);
        let unpacked = unpack(&packed, weights.len());
        // Expected ternary: +1, -1, 0, -1, +1, -1, 0, 0
        let expected = vec![1.0, -1.0, 0.0, -1.0, 1.0, -1.0, 0.0, 0.0];
        for (i, (&u, &e)) in unpacked.iter().zip(expected.iter()).enumerate() {
            assert!((u - e).abs() < 1e-6, "mismatch at {}: got {} expected {}", i, u, e);
        }
    }

    #[test]
    fn test_unpack_with_scale() {
        let weights: Vec<f32> = vec![1.0, -1.0, 0.05];
        let packed = pack(&weights);
        let unpacked = unpack_with_scale(&packed, 3, 2.5);
        let expected = vec![2.5, -2.5, 0.0];
        for (i, (&u, &e)) in unpacked.iter().zip(expected.iter()).enumerate() {
            assert!((u - e).abs() < 1e-6, "mismatch at {}: got {} expected {}", i, u, e);
        }
    }

    #[test]
    fn test_unpack_with_per_channel_scales() {
        let weights: Vec<f32> = vec![1.0, -1.0, 0.05, 1.0];
        let packed = pack(&weights);
        let scales = vec![1.0, 2.0, 3.0, 0.5];
        let unpacked = unpack_with_scales(&packed, 4, Some(&scales));
        let expected = vec![1.0, -2.0, 0.0, 0.5];
        for (i, (&u, &e)) in unpacked.iter().zip(expected.iter()).enumerate() {
            assert!((u - e).abs() < 1e-6, "mismatch at {}: got {} expected {}", i, u, e);
        }
    }

    #[test]
    fn test_unpack_to_ternary() {
        let weights: Vec<f32> = vec![1.0, -1.0, 0.05];
        let packed = pack(&weights);
        let ternary = unpack_to_ternary(&packed, 3);
        assert_eq!(ternary, vec![1i8, -1i8, 0i8]);
    }

    #[test]
    fn test_round_trip_large() {
        let weights: Vec<f32> = (0..1000).map(|i| {
            match i % 3 {
                0 => 1.5f32,
                1 => -2.0f32,
                _ => 0.01f32,
            }
        }).collect();
        let packed = pack(&weights);
        let unpacked = unpack(&packed, weights.len());
        for (i, &v) in unpacked.iter().enumerate() {
            let expected = match i % 3 {
                0 => 1.0f32,
                1 => -1.0f32,
                _ => 0.0f32,
            };
            assert!((v - expected).abs() < 1e-6, "mismatch at {}", i);
        }
    }
}
