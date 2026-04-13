use std::{env, process::Command};

fn main() {
    println!("cargo:rerun-if-env-changed=PYO3_PYTHON");

    if env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("linux") {
        return;
    }

    if let Some(libdir) = python_libdir() {
        println!("cargo:rustc-link-arg=-Wl,-rpath,{libdir}");
    }
}

fn python_libdir() -> Option<String> {
    let python = env::var("PYO3_PYTHON").unwrap_or_else(|_| String::from("python3"));
    let output = Command::new(python)
        .args([
            "-c",
            "import sysconfig; print(sysconfig.get_config_var('LIBDIR') or '')",
        ])
        .output()
        .ok()?;

    if !output.status.success() {
        return None;
    }

    let libdir = String::from_utf8(output.stdout).ok()?;
    let libdir = libdir.trim();
    if libdir.is_empty() {
        None
    } else {
        Some(libdir.to_owned())
    }
}
