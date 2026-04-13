use std::{
    env,
    error::Error,
    fs::File,
    io::{self, BufWriter, Write},
    path::{Path, PathBuf},
};

const ENV_DYLIB_PATH: &str = "ROBOWBC_ORT_DYLIB_PATH";
const DIST_NAME: &str = "onnxruntime-linux-x64-1.24.2";
const DIST_URL: &str =
    "https://github.com/microsoft/onnxruntime/releases/download/v1.24.2/onnxruntime-linux-x64-1.24.2.tgz";
const DYLIB_RELATIVE_PATH: &str = "lib/libonnxruntime.so.1.24.2";

fn main() {
    println!("cargo:rerun-if-env-changed={ENV_DYLIB_PATH}");

    if let Some(path) = env::var_os(ENV_DYLIB_PATH) {
        println!(
            "cargo:rustc-env={ENV_DYLIB_PATH}={}",
            PathBuf::from(path).display()
        );
        return;
    }

    if !matches_target("linux", "x86_64") {
        return;
    }

    match ensure_official_runtime() {
        Ok(path) => println!("cargo:rustc-env={ENV_DYLIB_PATH}={}", path.display()),
        Err(err) => println!("cargo:warning=failed to prepare ONNX Runtime shared library: {err}"),
    }
}

fn matches_target(target_os: &str, target_arch: &str) -> bool {
    env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok(target_os)
        && env::var("CARGO_CFG_TARGET_ARCH").as_deref() == Ok(target_arch)
}

fn ensure_official_runtime() -> Result<PathBuf, Box<dyn Error>> {
    let out_dir = PathBuf::from(env::var("OUT_DIR")?);
    let dist_dir = out_dir.join(DIST_NAME);
    let dylib_path = dist_dir.join(DYLIB_RELATIVE_PATH);
    if dylib_path.is_file() {
        return Ok(dylib_path);
    }

    let archive_path = out_dir.join(format!("{DIST_NAME}.tgz"));
    if !archive_path.is_file() {
        download_archive(&archive_path)?;
    }

    extract_archive(&archive_path, &out_dir)?;

    if dylib_path.is_file() {
        Ok(dylib_path)
    } else {
        Err(format!(
            "expected ONNX Runtime shared library at {} after extraction",
            dylib_path.display()
        )
        .into())
    }
}

fn download_archive(archive_path: &Path) -> Result<(), Box<dyn Error>> {
    let response = ureq::get(DIST_URL).call()?;
    let mut reader = response.into_body().into_reader();
    let mut writer = BufWriter::new(File::create(archive_path)?);
    io::copy(&mut reader, &mut writer)?;
    writer.flush()?;
    Ok(())
}

fn extract_archive(archive_path: &Path, out_dir: &Path) -> Result<(), Box<dyn Error>> {
    let archive = File::open(archive_path)?;
    let decoder = flate2::read::GzDecoder::new(archive);
    let mut archive = tar::Archive::new(decoder);
    archive.unpack(out_dir)?;
    Ok(())
}
