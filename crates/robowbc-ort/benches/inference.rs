//! Benchmarks for single-inference latency on ONNX models.
//!
//! Run with: `cargo bench -p robowbc-ort`
//!
//! Requires test model fixtures in `tests/fixtures/`. Generate them with:
//! ```sh
//! python3 tests/fixtures/gen_test_models.py
//! ```

use criterion::{criterion_group, criterion_main, Criterion};
use robowbc_ort::OrtBackend;
use std::path::PathBuf;

fn fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures")
}

fn bench_identity_inference(c: &mut Criterion) {
    let model_path = fixture_dir().join("test_identity.onnx");
    if !model_path.exists() {
        eprintln!(
            "skipping identity benchmark: model not found at {}",
            model_path.display()
        );
        return;
    }

    let mut backend = OrtBackend::from_file(&model_path).expect("model should load");
    let input_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];

    c.bench_function("identity_1x4", |b| {
        b.iter(|| {
            backend
                .run(&[("input", &input_data, &[1, 4])])
                .expect("inference should succeed");
        });
    });
}

fn bench_relu_inference(c: &mut Criterion) {
    let model_path = fixture_dir().join("test_relu.onnx");
    if !model_path.exists() {
        eprintln!(
            "skipping relu benchmark: model not found at {}",
            model_path.display()
        );
        return;
    }

    let mut backend = OrtBackend::from_file(&model_path).expect("model should load");
    let input_data: Vec<f32> = vec![-3.0, -1.0, 0.0, 1.0, 2.0, 5.0, -0.5, 10.0];

    c.bench_function("relu_1x8", |b| {
        b.iter(|| {
            backend
                .run(&[("input", &input_data, &[1, 8])])
                .expect("inference should succeed");
        });
    });
}

criterion_group!(benches, bench_identity_inference, bench_relu_inference);
criterion_main!(benches);
