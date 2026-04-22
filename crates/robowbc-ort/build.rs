use std::{
    env,
    error::Error,
    fs::{self, File},
    io::{self, BufWriter, Write},
    path::{Path, PathBuf},
};

const ENV_DYLIB_PATH: &str = "ROBOWBC_ORT_DYLIB_PATH";
const DIST_NAME: &str = "onnxruntime-linux-x64-1.24.2";
const DIST_URL: &str =
    "https://github.com/microsoft/onnxruntime/releases/download/v1.24.2/onnxruntime-linux-x64-1.24.2.tgz";
const DYLIB_RELATIVE_PATH: &str = "lib/libonnxruntime.so.1.24.2";
const PROVIDERS_SHARED_RELATIVE_PATH: &str = "lib/libonnxruntime_providers_shared.so";

fn main() {
    println!("cargo:rerun-if-env-changed={ENV_DYLIB_PATH}");

    if env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("linux") {
        panic!("robowbc-ort only supports Linux targets");
    }

    if let Some(path) = env::var_os(ENV_DYLIB_PATH) {
        println!(
            "cargo:rustc-env={ENV_DYLIB_PATH}={}",
            PathBuf::from(path).display()
        );
        return;
    }

    if env::var("CARGO_CFG_TARGET_ARCH").as_deref() != Ok("x86_64") {
        return;
    }

    match ensure_official_runtime() {
        Ok(path) => println!("cargo:rustc-env={ENV_DYLIB_PATH}={}", path.display()),
        Err(err) => println!("cargo:warning=failed to prepare ONNX Runtime shared library: {err}"),
    }
}

fn ensure_official_runtime() -> Result<PathBuf, Box<dyn Error>> {
    let out_dir = PathBuf::from(env::var("OUT_DIR")?);
    let dist_dir = out_dir.join(DIST_NAME);
    let dylib_path = dist_dir.join(DYLIB_RELATIVE_PATH);
    if runtime_is_complete(&dist_dir) {
        return Ok(dylib_path);
    }

    if dist_dir.exists() {
        fs::remove_dir_all(&dist_dir).map_err(|e| {
            format!(
                "failed to remove incomplete ONNX Runtime extraction at {}: {e}",
                dist_dir.display()
            )
        })?;
    }

    let archive_path = out_dir.join(format!("{DIST_NAME}.tgz"));
    if !archive_path.is_file() {
        download_archive(&archive_path)?;
    }

    if let Err(err) = extract_archive(&archive_path, &out_dir) {
        if dist_dir.exists() {
            fs::remove_dir_all(&dist_dir).map_err(|cleanup_err| {
                format!(
                    "failed to clean incomplete ONNX Runtime extraction at {} after extract error ({err}): {cleanup_err}",
                    dist_dir.display()
                )
            })?;
        }
        if archive_path.exists() {
            fs::remove_file(&archive_path).map_err(|cleanup_err| {
                format!(
                    "failed to remove corrupted ONNX Runtime archive {} after extract error ({err}): {cleanup_err}",
                    archive_path.display()
                )
            })?;
        }
        download_archive(&archive_path)?;
        extract_archive(&archive_path, &out_dir).map_err(|retry_err| {
            format!(
                "failed to extract ONNX Runtime archive {} ({err}); retry after re-download also failed: {retry_err}",
                archive_path.display()
            )
        })?;
    }

    if runtime_is_complete(&dist_dir) {
        Ok(dylib_path)
    } else {
        Err(format!(
            "expected ONNX Runtime shared libraries at {} and {} after extraction",
            dist_dir.join(DYLIB_RELATIVE_PATH).display(),
            dist_dir.join(PROVIDERS_SHARED_RELATIVE_PATH).display()
        )
        .into())
    }
}

fn runtime_is_complete(dist_dir: &Path) -> bool {
    dist_dir.join(DYLIB_RELATIVE_PATH).is_file()
        && dist_dir.join(PROVIDERS_SHARED_RELATIVE_PATH).is_file()
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
