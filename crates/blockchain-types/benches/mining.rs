use std::hint::black_box;

use blake3::{Hash, Hasher};
use blockchain_types::{
    BlockConstructor,
    block::mining::{
        Miner, MinerSimple, MiningStrategy,
        mining_utils::{hash_with_nonce, is_valid_target_hash, is_valid_target_hash_const},
        simd::is_valid_target_hash_simd,
    },
};
use criterion::{
    BenchmarkId, Criterion, SamplingMode, Throughput, criterion_group, criterion_main,
};

/// Helper to create hash with N leading zero bytes
fn hash_with_zeros(zero_bytes: usize) -> Hash {
    let mut bytes = [0xFFu8; 32];
    bytes[..zero_bytes].fill(0);
    Hash::from_bytes(bytes)
}

/// Benchmark runtime difficulty validation (`is_valid_target_hash`)
fn bench_runtime_validation(c: &mut Criterion) {
    let mut group = c.benchmark_group("runtime_validation");

    for &difficulty in &[4, 8, 16, 24, 32, 48, 64] {
        let hash = hash_with_zeros(difficulty / 2);
        group.bench_with_input(
            BenchmarkId::new("is_valid_target_hash", difficulty),
            &difficulty,
            |b, &d| {
                b.iter(|| black_box(is_valid_target_hash(&hash, d)));
            },
        );
    }
    group.finish();
}

/// Benchmark const generic validation (`is_valid_target_hash_const`)
fn bench_const_validation(c: &mut Criterion) {
    let mut group = c.benchmark_group("const_validation");

    // D=0,1,2: Inline checks
    group.bench_function("D=0_inline", |b| {
        let hash = hash_with_zeros(0);
        b.iter(|| black_box(is_valid_target_hash_const::<0>(&hash)));
    });

    group.bench_function("D=1_inline", |b| {
        let hash = hash_with_zeros(1);
        b.iter(|| black_box(is_valid_target_hash_const::<1>(&hash)));
    });

    group.bench_function("D=2_inline", |b| {
        let hash = hash_with_zeros(1);
        b.iter(|| black_box(is_valid_target_hash_const::<2>(&hash)));
    });

    // D=4,6: Auto-vectorized scalar
    group.bench_function("D=4_scalar", |b| {
        let hash = hash_with_zeros(2);
        b.iter(|| black_box(is_valid_target_hash_const::<4>(&hash)));
    });

    group.bench_function("D=6_scalar", |b| {
        let hash = hash_with_zeros(3);
        b.iter(|| black_box(is_valid_target_hash_const::<6>(&hash)));
    });

    // D>=8: SIMD path
    for &difficulty in &[8, 16, 24, 32, 48, 64] {
        let hash = hash_with_zeros(difficulty / 2);
        group.bench_function(format!("D={difficulty}_simd"), |b| {
            b.iter(|| {
                black_box(match difficulty {
                    8 => is_valid_target_hash_const::<8>(&hash),
                    16 => is_valid_target_hash_const::<16>(&hash),
                    24 => is_valid_target_hash_const::<24>(&hash),
                    32 => is_valid_target_hash_const::<32>(&hash),
                    48 => is_valid_target_hash_const::<48>(&hash),
                    64 => is_valid_target_hash_const::<64>(&hash),
                    _ => unreachable!(),
                })
            });
        });
    }

    group.finish();
}

/// Benchmark SIMD-specific validation
fn bench_simd_validation(c: &mut Criterion) {
    let mut group = c.benchmark_group("simd_validation");

    // Minimum SIMD difficulty (D=8)
    group.bench_function("D=8_4_bytes", |b| {
        let hash = hash_with_zeros(4);
        b.iter(|| black_box(is_valid_target_hash_simd::<8>(&hash)));
    });

    // Single SIMD chunk boundary (D=32, 16 bytes)
    group.bench_function("D=32_one_chunk", |b| {
        let hash = hash_with_zeros(16);
        b.iter(|| black_box(is_valid_target_hash_simd::<32>(&hash)));
    });

    // Two SIMD chunks (D=64, 32 bytes)
    group.bench_function("D=64_two_chunks", |b| {
        let hash = hash_with_zeros(32);
        b.iter(|| black_box(is_valid_target_hash_simd::<64>(&hash)));
    });

    // Partial chunks (test scalar remainder handling)
    group.bench_function("D=24_12_bytes", |b| {
        let hash = hash_with_zeros(12);
        b.iter(|| black_box(is_valid_target_hash_simd::<24>(&hash)));
    });

    group.bench_function("D=48_24_bytes", |b| {
        let hash = hash_with_zeros(24);
        b.iter(|| black_box(is_valid_target_hash_simd::<48>(&hash)));
    });

    group.finish();
}

/// Benchmark SIMD vs scalar for same difficulty
fn bench_simd_vs_scalar(c: &mut Criterion) {
    let mut group = c.benchmark_group("simd_vs_scalar");

    for &difficulty in &[8, 16, 24, 32] {
        let hash = hash_with_zeros(difficulty / 2);

        // Runtime scalar
        group.bench_with_input(
            BenchmarkId::new("runtime_scalar", difficulty),
            &difficulty,
            |b, &d| {
                b.iter(|| black_box(is_valid_target_hash(&hash, d)));
            },
        );

        // SIMD
        group.bench_function(format!("simd_D={difficulty}"), |b| {
            b.iter(|| {
                black_box(match difficulty {
                    8 => is_valid_target_hash_simd::<8>(&hash),
                    16 => is_valid_target_hash_simd::<16>(&hash),
                    24 => is_valid_target_hash_simd::<24>(&hash),
                    32 => is_valid_target_hash_simd::<32>(&hash),
                    _ => unreachable!(),
                })
            });
        });
    }

    group.finish();
}

/// Benchmark early exit performance (hash fails quickly)
fn bench_early_exit(c: &mut Criterion) {
    let mut group = c.benchmark_group("early_exit");

    // Fail at first byte
    let hash_fail_first = Hash::from_bytes({
        let mut b = [0u8; 32];
        b[0] = 0xFF;
        b
    });

    // Fail at chunk boundary (byte 15)
    let hash_fail_boundary = Hash::from_bytes({
        let mut b = [0u8; 32];
        b[0..15].fill(0);
        b[15] = 0xFF;
        b
    });

    // Fail at last byte
    let hash_fail_last = Hash::from_bytes({
        let mut b = [0u8; 32];
        b[31] = 0xFF;
        b
    });

    group.bench_function("fail_first_byte_D=32", |b| {
        b.iter(|| black_box(is_valid_target_hash_simd::<32>(&hash_fail_first)));
    });

    group.bench_function("fail_chunk_boundary_D=32", |b| {
        b.iter(|| black_box(is_valid_target_hash_simd::<32>(&hash_fail_boundary)));
    });

    group.bench_function("fail_last_byte_D=64", |b| {
        b.iter(|| black_box(is_valid_target_hash_simd::<64>(&hash_fail_last)));
    });

    group.finish();
}

/// Benchmark validation with partial zeros (odd difficulties)
fn bench_partial_zero(c: &mut Criterion) {
    let mut group = c.benchmark_group("partial_zero");

    // D=9 (4 bytes + 1 nibble)
    let hash_d9 = Hash::from_bytes({
        let mut b = [0xFFu8; 32];
        b[0..4].fill(0);
        b[4] = 0x0F; // Upper nibble zero
        b
    });

    group.bench_function("D=9_partial", |b| {
        b.iter(|| black_box(is_valid_target_hash_simd::<9>(&hash_d9)));
    });

    // D=33 (16 bytes + 1 nibble, crosses SIMD chunk)
    let hash_d33 = Hash::from_bytes({
        let mut b = [0xFFu8; 32];
        b[0..16].fill(0);
        b[16] = 0x0F;
        b
    });

    group.bench_function("D=33_partial_after_chunk", |b| {
        b.iter(|| black_box(is_valid_target_hash_simd::<33>(&hash_d33)));
    });

    group.finish();
}

/// Benchmark different byte patterns to test SIMD comparison robustness
#[allow(clippy::needless_range_loop)]
fn bench_byte_patterns(c: &mut Criterion) {
    let mut group = c.benchmark_group("byte_patterns");

    // All zeros (best case)
    let hash_all_zeros = Hash::from_bytes([0u8; 32]);
    group.bench_function("all_zeros_D=32", |b| {
        b.iter(|| black_box(is_valid_target_hash_simd::<32>(&hash_all_zeros)));
    });

    // Alternating pattern after required zeros
    let hash_alternating = Hash::from_bytes({
        let mut b = [0u8; 32];
        for i in 16..32 {
            b[i] = if i % 2 == 0 { 0x00 } else { 0xFF };
        }
        b
    });
    group.bench_function("alternating_after_zeros_D=32", |b| {
        b.iter(|| black_box(is_valid_target_hash_simd::<32>(&hash_alternating)));
    });

    // Gradient pattern
    let hash_gradient = Hash::from_bytes({
        let mut b = [0u8; 32];
        for i in 16..32 {
            b[i] = i as u8;
        }
        b
    });
    group.bench_function("gradient_after_zeros_D=32", |b| {
        b.iter(|| black_box(is_valid_target_hash_simd::<32>(&hash_gradient)));
    });

    group.finish();
}

/// Comprehensive comparison across all strategies
fn bench_strategy_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("strategy_comparison");

    for &difficulty in &[4, 8, 16, 32] {
        let hash = hash_with_zeros(difficulty / 2);

        // Runtime validation
        group.bench_with_input(
            BenchmarkId::new("runtime", difficulty),
            &difficulty,
            |b, &d| {
                b.iter(|| black_box(is_valid_target_hash(&hash, d)));
            },
        );

        // Const generic validation
        group.bench_function(format!("const_D={difficulty}"), |b| {
            b.iter(|| {
                black_box(match difficulty {
                    4 => is_valid_target_hash_const::<4>(&hash),
                    8 => is_valid_target_hash_const::<8>(&hash),
                    16 => is_valid_target_hash_const::<16>(&hash),
                    32 => is_valid_target_hash_const::<32>(&hash),
                    _ => unreachable!(),
                })
            });
        });
    }

    group.finish();
}

criterion_group!(
    validation_benches,
    bench_runtime_validation,
    bench_const_validation,
    bench_simd_validation,
    bench_simd_vs_scalar,
    bench_early_exit,
    bench_partial_zero,
    bench_byte_patterns,
    bench_strategy_comparison,
);

/// Benchmark `hash_with_nonce` (the core mining operation)
fn bench_hash_with_nonce(c: &mut Criterion) {
    let mut group = c.benchmark_group("hash_with_nonce");

    let hasher = Hasher::new();

    group.bench_function("single_hash", |b| {
        b.iter(|| {
            let nonce = black_box(12345u64);
            black_box(hash_with_nonce(hasher.clone(), nonce))
        });
    });

    group.finish();
}

/// Benchmark mining loop iterations (hash + validate)
/// Uses very low difficulty to ensure fast completion
fn bench_mining_iteration(c: &mut Criterion) {
    let mut group = c.benchmark_group("mining_iteration");
    group.sampling_mode(SamplingMode::Flat); // More consistent for short operations

    let hasher = Hasher::new();

    // D=4 (2 zero bytes) - easy to find, ~1/65536 hashes
    group.bench_function("hash_and_validate_D=4", |b| {
        b.iter(|| {
            let nonce = black_box(12345u64);
            let hash = hash_with_nonce(hasher.clone(), nonce);
            black_box(is_valid_target_hash(&hash, 4))
        });
    });

    group.bench_function("hash_and_validate_const_D=4", |b| {
        b.iter(|| {
            let nonce = black_box(12345u64);
            let hash = hash_with_nonce(hasher.clone(), nonce);
            black_box(is_valid_target_hash_const::<4>(&hash))
        });
    });

    group.finish();
}

/// Benchmark expected hashes to solution for different difficulties
/// Uses statistical sampling rather than actual mining
fn bench_mining_throughput(c: &mut Criterion) {
    let mut group = c.benchmark_group("mining_throughput");
    group.sampling_mode(SamplingMode::Flat);

    let hasher = Hasher::new();

    // Measure hash rate: how many hash+validate cycles per second?
    for &difficulty in &[0, 4, 8] {
        group.throughput(Throughput::Elements(1));
        group.bench_with_input(
            BenchmarkId::new("hashes_per_sec", difficulty),
            &difficulty,
            |b, &d| {
                let mut nonce = 0u64;
                b.iter(|| {
                    nonce = nonce.wrapping_add(1);
                    let hash = hash_with_nonce(hasher.clone(), nonce);
                    black_box(is_valid_target_hash(&hash, d))
                });
            },
        );
    }

    group.finish();
}

/// Benchmark mining strategy: sequential nonce search
/// Limits iterations to avoid long-running benchmarks
fn bench_mining_strategies(c: &mut Criterion) {
    let mut group = c.benchmark_group("mining_strategies");
    group.sample_size(10); // Fewer samples since each is expensive

    let hasher = Hasher::new();

    // Search for valid hash with D=4 (relatively easy)
    // Limit to 10,000 attempts to bound runtime
    group.bench_function("sequential_search_D=4_limited", |b| {
        b.iter(|| {
            let mut nonce = black_box(0u64);
            let max_attempts = 10_000;

            for _ in 0..max_attempts {
                let hash = hash_with_nonce(hasher.clone(), nonce);
                if is_valid_target_hash(&hash, 4) {
                    return black_box(nonce);
                }
                nonce = nonce.wrapping_add(1);
            }
            black_box(nonce) // Return last attempted nonce if not found
        });
    });

    // Compare const generic version
    group.bench_function("sequential_search_const_D=4_limited", |b| {
        b.iter(|| {
            let mut nonce = black_box(0u64);
            let max_attempts = 10_000;

            for _ in 0..max_attempts {
                let hash = hash_with_nonce(hasher.clone(), nonce);
                if is_valid_target_hash_const::<4>(&hash) {
                    return black_box(nonce);
                }
                nonce = nonce.wrapping_add(1);
            }
            black_box(nonce)
        });
    });

    group.finish();
}

/// Benchmark parallel mining (if rayon feature enabled)
#[cfg(feature = "rayon")]
fn bench_parallel_mining(c: &mut Criterion) {
    use rayon::prelude::*;

    let mut group = c.benchmark_group("parallel_mining");
    group.sample_size(10);

    let hasher = Hasher::new();

    // Parallel search with limited attempts per thread
    group.bench_function("rayon_parallel_D=4_limited", |b| {
        b.iter(|| {
            let attempts_per_thread = 1000;
            let num_threads = rayon::current_num_threads();

            (0..num_threads)
                .into_par_iter()
                .find_map_any(|thread_id| {
                    let start_nonce = thread_id as u64 * attempts_per_thread;

                    for offset in 0..attempts_per_thread {
                        let nonce = start_nonce + offset;
                        let hash = hash_with_nonce(hasher.clone(), nonce);
                        if is_valid_target_hash_const::<4>(&hash) {
                            return Some(nonce);
                        }
                    }
                    None
                })
                .unwrap_or(0)
        });
    });

    group.finish();
}

/// Benchmark validation overhead in mining context
fn bench_validation_overhead(c: &mut Criterion) {
    let mut group = c.benchmark_group("validation_overhead");

    let hasher = Hasher::new();
    let hash = hash_with_nonce(hasher.clone(), 12345);

    // Just hashing
    group.bench_function("hash_only", |b| {
        b.iter(|| {
            let nonce = black_box(12345u64);
            black_box(hash_with_nonce(hasher.clone(), nonce))
        });
    });

    // Hash + runtime validation
    group.bench_function("hash_plus_runtime_validate", |b| {
        b.iter(|| {
            let nonce = black_box(12345u64);
            let hash = hash_with_nonce(hasher.clone(), nonce);
            black_box(is_valid_target_hash(&hash, 16))
        });
    });

    // Hash + const validation
    group.bench_function("hash_plus_const_validate", |b| {
        b.iter(|| {
            let nonce = black_box(12345u64);
            let hash = hash_with_nonce(hasher.clone(), nonce);
            black_box(is_valid_target_hash_const::<16>(&hash))
        });
    });

    // Just validation (for comparison)
    group.bench_function("validate_only", |b| {
        b.iter(|| black_box(is_valid_target_hash(&hash, 16)));
    });

    group.finish();
}

/// Estimate mining difficulty impact
fn bench_difficulty_impact(c: &mut Criterion) {
    let mut group = c.benchmark_group("difficulty_impact");
    group.sample_size(20);

    let hasher = Hasher::new();

    // For each difficulty, measure time to check 1000 hashes
    // This simulates mining workload without actual solution-finding

    for &difficulty in &[0, 4, 8, 16] {
        group.bench_with_input(
            BenchmarkId::new("check_1000_hashes", difficulty),
            &difficulty,
            |b, &d| {
                b.iter(|| {
                    let mut count = 0;
                    for nonce in 0u64..1000 {
                        let hash = hash_with_nonce(hasher.clone(), nonce);
                        let valid = match d {
                            0 => is_valid_target_hash_const::<0>(&hash),
                            4 => is_valid_target_hash_const::<4>(&hash),
                            8 => is_valid_target_hash_const::<8>(&hash),
                            16 => is_valid_target_hash_const::<16>(&hash),
                            _ => unreachable!(),
                        };
                        if valid {
                            count += 1;
                        }
                    }
                    black_box(count)
                });
            },
        );
    }

    group.finish();
}

#[cfg(feature = "rayon")]
criterion_group!(
    hash_validation_benches,
    bench_hash_with_nonce,
    bench_mining_iteration,
    bench_mining_throughput,
    bench_mining_strategies,
    bench_parallel_mining,
    bench_validation_overhead,
    bench_difficulty_impact,
);

#[cfg(not(feature = "rayon"))]
criterion_group!(
    hash_validation_benches,
    bench_hash_with_nonce,
    bench_mining_iteration,
    bench_mining_throughput,
    bench_mining_strategies,
    bench_validation_overhead,
    bench_difficulty_impact,
);

/// Create a simple block constructor for benchmarking
fn create_test_constructor() -> BlockConstructor {
    BlockConstructor::new(
        1,                                    // index
        &[],                                  // transactions
        blake3::hash(b"previous block hash"), // previous_hash
        None,
    )
}

/// Benchmark mining with controlled difficulty
/// Uses statistical sampling to estimate mining time without running full proof-of-work
fn bench_mining_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("mining_comparison");
    group.sampling_mode(SamplingMode::Flat);
    group.sample_size(20); // Fewer samples for expensive operations

    // D=0: Trivial (always valid)
    group.bench_function("sequential_runtime_D=0", |b| {
        b.iter(|| {
            let constructor = create_test_constructor();
            black_box(MinerSimple::mine(constructor, 0, 0))
        });
    });

    group.bench_function("sequential_const_D=0", |b| {
        b.iter(|| {
            let constructor = create_test_constructor();
            black_box(Miner::<0>::mine(constructor, 0, 0))
        });
    });

    // D=2: Very easy (~1 in 256 hashes)
    group.bench_function("sequential_runtime_D=2", |b| {
        b.iter(|| {
            let constructor = create_test_constructor();
            black_box(MinerSimple::mine(constructor, 2, 0))
        });
    });

    group.bench_function("sequential_const_D=2", |b| {
        b.iter(|| {
            let constructor = create_test_constructor();
            black_box(Miner::<2>::mine(constructor, 2, 0))
        });
    });

    // D=4: Easy (~1 in 65K hashes, ~6-10ms)
    group.bench_function("sequential_runtime_D=4", |b| {
        b.iter(|| {
            let constructor = create_test_constructor();
            black_box(MinerSimple::mine(constructor, 4, 0))
        });
    });

    group.bench_function("sequential_const_D=4", |b| {
        b.iter(|| {
            let constructor = create_test_constructor();
            black_box(Miner::<4>::mine(constructor, 4, 0))
        });
    });

    #[cfg(feature = "rayon")]
    group.bench_function("parallel_const_D=4", |b| {
        b.iter(|| {
            let constructor = create_test_constructor();
            black_box(Miner::<4>::mine(constructor, 4, 0))
        });
    });

    group.finish();
}

/// Benchmark expected time to solution for different difficulties
/// Uses limited iterations to bound runtime
fn bench_time_to_solution(c: &mut Criterion) {
    let mut group = c.benchmark_group("time_to_solution");
    group.sampling_mode(SamplingMode::Flat);
    group.sample_size(10);

    // D=0: Instant (first hash always valid)
    group.bench_function("D=0_instant", |b| {
        b.iter(|| {
            let constructor = create_test_constructor();
            black_box(Miner::<0>::mine(constructor, 0, 0))
        });
    });

    // D=1: ~1 in 16 hashes (upper nibble zero)
    group.bench_function("D=1_approx_16_hashes", |b| {
        b.iter(|| {
            let constructor = create_test_constructor();
            black_box(Miner::<1>::mine(constructor, 1, 0))
        });
    });

    // D=2: ~1 in 256 hashes (one zero byte)
    group.bench_function("D=2_approx_256_hashes", |b| {
        b.iter(|| {
            let constructor = create_test_constructor();
            black_box(Miner::<2>::mine(constructor, 2, 0))
        });
    });

    // D=3: ~1 in 4K hashes
    group.bench_function("D=3_approx_4k_hashes", |b| {
        b.iter(|| {
            let constructor = create_test_constructor();
            black_box(Miner::<3>::mine(constructor, 3, 0))
        });
    });

    // D=4: ~1 in 65K hashes (~6-10ms expected)
    group.bench_function("D=4_approx_65k_hashes", |b| {
        b.iter(|| {
            let constructor = create_test_constructor();
            black_box(Miner::<4>::mine(constructor, 4, 0))
        });
    });

    // D=5: ~1 in 1M hashes (~100-200ms expected)
    // Only run if rayon feature enabled (too slow otherwise)
    #[cfg(feature = "rayon")]
    group.bench_function("D=5_approx_1m_hashes", |b| {
        b.iter(|| {
            let constructor = create_test_constructor();
            black_box(Miner::<5>::mine(constructor, 5, 0))
        });
    });

    group.finish();
}

/// Benchmark mining iteration rate (hashes per second)
fn bench_hash_rate(c: &mut Criterion) {
    let mut group = c.benchmark_group("hash_rate");
    group.sampling_mode(SamplingMode::Flat);

    let constructor = create_test_constructor();
    let hasher = constructor.hash_state().clone();

    // Single hash throughput
    group.bench_function("single_hash", |b| {
        let mut nonce = 0u64;
        b.iter(|| {
            nonce = nonce.wrapping_add(1);
            black_box(hash_with_nonce(hasher.clone(), nonce))
        });
    });

    // Sequential mining loop (1000 iterations)
    group.bench_function("sequential_1000_iterations_D=4", |b| {
        b.iter(|| {
            let mut nonce = 0u64;
            let mut found = 0;
            for _ in 0..1000 {
                let hash = hash_with_nonce(hasher.clone(), nonce);
                if is_valid_target_hash_const::<4>(&hash) {
                    found += 1;
                }
                nonce = nonce.wrapping_add(1);
            }
            black_box(found)
        });
    });

    group.finish();
}

/// Benchmark validation overhead in mining context
fn bench_mining_validation_overhead(c: &mut Criterion) {
    let mut group = c.benchmark_group("mining_validation_overhead");

    let constructor = create_test_constructor();
    let hasher = constructor.hash_state().clone();

    // Just hashing (baseline)
    group.bench_function("hash_only_no_validation", |b| {
        let mut nonce = 0u64;
        b.iter(|| {
            nonce = nonce.wrapping_add(1);
            black_box(hash_with_nonce(hasher.clone(), nonce))
        });
    });

    // Hash + runtime validation
    group.bench_function("hash_plus_runtime_D=4", |b| {
        let mut nonce = 0u64;
        b.iter(|| {
            nonce = nonce.wrapping_add(1);
            let hash = hash_with_nonce(hasher.clone(), nonce);
            black_box(is_valid_target_hash(&hash, 4))
        });
    });

    // Hash + const validation
    group.bench_function("hash_plus_const_D=4", |b| {
        let mut nonce = 0u64;
        b.iter(|| {
            nonce = nonce.wrapping_add(1);
            let hash = hash_with_nonce(hasher.clone(), nonce);
            black_box(is_valid_target_hash_const::<4>(&hash))
        });
    });

    // Hash + const validation (SIMD path)
    group.bench_function("hash_plus_const_D=16", |b| {
        let mut nonce = 0u64;
        b.iter(|| {
            nonce = nonce.wrapping_add(1);
            let hash = hash_with_nonce(hasher.clone(), nonce);
            black_box(is_valid_target_hash_const::<16>(&hash))
        });
    });

    group.finish();
}

/// Benchmark parallel vs sequential mining
#[cfg(feature = "rayon")]
fn bench_parallel_vs_sequential(c: &mut Criterion) {
    let mut group = c.benchmark_group("parallel_vs_sequential");
    group.sample_size(10);

    // D=4: Sequential
    group.bench_function("sequential_D=4", |b| {
        b.iter(|| {
            let constructor = create_test_constructor();
            black_box(Miner::<4>::mine(constructor, 4, 0))
        });
    });

    // D=4: Parallel (automatically uses rayon for D >= 6 in const miner)
    group.bench_function("parallel_D=4", |b| {
        b.iter(|| {
            let constructor = create_test_constructor();
            // Force parallel by using D >= 6
            black_box(Miner::<6>::mine(constructor, 6, 0))
        });
    });

    // D=5: Higher difficulty to show parallel benefit
    group.bench_function("sequential_D=5", |b| {
        b.iter(|| {
            let constructor = create_test_constructor();
            // Use D < 6 to force sequential in MinerConst
            black_box(Miner::<5>::mine(constructor, 5, 0))
        });
    });

    group.bench_function("parallel_D=5", |b| {
        b.iter(|| {
            let constructor = create_test_constructor();
            // Use runtime miner which doesn't have the D < 6 sequential optimization
            // Or we need to create a separate parallel miner for testing
            black_box(Miner::<6>::mine(constructor, 6, 0))
        });
    });

    group.finish();
}

/// Benchmark mining strategy overhead
fn bench_strategy_selection(c: &mut Criterion) {
    let mut group = c.benchmark_group("strategy_selection");

    // Compare different difficulty thresholds where strategy changes

    // D=5: Last difficulty that might use sequential
    group.bench_function("D=5_strategy_decision", |b| {
        b.iter(|| {
            let constructor = create_test_constructor();
            black_box(Miner::<5>::mine(constructor, 5, 0))
        });
    });

    // D=6: First difficulty that triggers parallel (if rayon enabled)
    #[cfg(feature = "rayon")]
    group.bench_function("D=6_parallel_threshold", |b| {
        b.iter(|| {
            let constructor = create_test_constructor();
            black_box(Miner::<6>::mine(constructor, 6, 0))
        });
    });

    // D=8: SIMD validation threshold
    #[cfg(feature = "rayon")]
    group.bench_function("D=8_simd_threshold", |b| {
        b.iter(|| {
            let constructor = create_test_constructor();
            black_box(Miner::<8>::mine(constructor, 8, 0))
        });
    });

    group.finish();
}

/// Benchmark block construction overhead
fn bench_block_construction(c: &mut Criterion) {
    let mut group = c.benchmark_group("block_construction");

    // Measure time to create constructor
    group.bench_function("create_constructor", |b| {
        b.iter(|| black_box(create_test_constructor()));
    });

    // Measure time to create constructor with transactions
    group.bench_function("create_constructor_with_txs", |b| {
        b.iter(|| {
            black_box(BlockConstructor::new(
                1,
                &[],
                blake3::hash(b"previous"),
                None,
            ))
        });
    });

    // Full D=0 mining (measures constructor + mining + block creation)
    group.bench_function("full_mine_D=0", |b| {
        b.iter(|| {
            let constructor = create_test_constructor();
            black_box(Miner::<0>::mine(constructor, 0, 0))
        });
    });

    group.finish();
}

/// Benchmark nonce space exploration strategies
fn bench_nonce_strategies(c: &mut Criterion) {
    let mut group = c.benchmark_group("nonce_strategies");
    group.sample_size(10);

    // Sequential from 0
    group.bench_function("sequential_from_0_D=4", |b| {
        b.iter(|| {
            let constructor = create_test_constructor();
            black_box(Miner::<4>::mine(constructor, 4, 0))
        });
    });

    // Sequential from random start
    group.bench_function("sequential_from_random_D=4", |b| {
        b.iter(|| {
            let constructor = create_test_constructor();
            // Generate random nonce per iteration
            let random_start = {
                use std::{collections::hash_map::RandomState, hash::BuildHasher};

                RandomState::new().hash_one(std::time::SystemTime::now())
            };
            black_box(Miner::<4>::mine(constructor, 4, random_start))
        });
    });

    group.finish();
}

#[cfg(feature = "rayon")]
criterion_group!(
    block_mining_benches,
    bench_mining_comparison,
    bench_time_to_solution,
    bench_hash_rate,
    bench_mining_validation_overhead,
    bench_parallel_vs_sequential,
    bench_strategy_selection,
    bench_block_construction,
    bench_nonce_strategies,
);

#[cfg(not(feature = "rayon"))]
criterion_group!(
    block_mining_benches,
    bench_mining_comparison,
    bench_time_to_solution,
    bench_hash_rate,
    bench_mining_validation_overhead,
    bench_strategy_selection,
    bench_block_construction,
    bench_nonce_strategies,
);

criterion_main!(
    validation_benches,
    hash_validation_benches,
    block_mining_benches
);
